//! `cargo xtask disposable-target` — the typed port of the retired
//! `scripts/run-in-disposable-cargo-target.sh`.
//!
//! Runs one foreground build journey in an owned Cargo target and removes
//! that target afterward. High-churn acceptance and release graphs remain
//! useful through sccache without accumulating linked binaries and symbols
//! in the repository.
//!
//! Every observable contract of the shell runner is preserved verbatim so
//! local habits, hooks, and Continuous Integration steps survive the move
//! unchanged: the `[cargo-artifact-lifecycle] lane=... status=...` log
//! lines (including the five-second progress lines), the exit codes (2 for
//! usage errors, 1 for runtime failures, 129/130/143 for HUP/INT/TERM),
//! the `.astronomical-disposable-cargo-target-v1:LANE` marker file,
//! nested-lane reuse, the rename-to-`.removing` ownership handoff, and the
//! three-second grace then kill escalation of the child process group.
//!
//! This module owns the command-line surface and the pure logic (argument
//! parsing, lane-name rules, marker identity, lane-root validation, `du`
//! output accounting, wrapper selection), all unit-tested beside the
//! source; the process and signal supervision lives in the `lifecycle`
//! child module. The `du -sk` measurement stays an external, killable
//! process on purpose: interruption during measurement must leave the
//! owned target retained (and testable through a PATH shim), which a
//! native walk could not guarantee.

#![allow(unsafe_code)]
// The workspace denies unsafe code everywhere else; the timestamp and
// write-permission probes below are the sanctioned edges for this module
// (the lifecycle child module carries its own allowance for signals).

mod cleanup;
mod lifecycle;
use crate::process_signals as signal_edge;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus};

const LIFECYCLE_MARKER_NAME: &str = ".astronomical-disposable-cargo-target";
const LIFECYCLE_MARKER_VERSION: &str = "astronomical-disposable-cargo-target-v1";

/// The resolved repository root is exported to the journey so a runner
/// invoked inside an existing lane rediscovers the same root the outer
/// runner used. The shell runner conveyed this through its own script
/// path; a compiled binary can be copied into lane targets, so the root
/// travels explicitly.
pub(crate) const DISPOSABLE_REPOSITORY_ROOT_ENV: &str = "ASTRONOMICAL_DISPOSABLE_REPOSITORY_ROOT";

// ── Pure helpers (unit-tested beside this source) ─────────────────────

pub(crate) fn is_valid_lane_name(lane_name: &str) -> bool {
    !lane_name.is_empty()
        && lane_name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && !lane_name.starts_with('-')
        && !lane_name.ends_with('-')
}

pub(crate) fn marker_identity(lane_name: &str) -> String {
    format!("{LIFECYCLE_MARKER_VERSION}:{lane_name}")
}

/// `du -sk` reports total allocated storage in kibibytes as the first
/// whitespace-separated token of its first output line; the lifecycle log
/// carries that value converted to bytes.
pub(crate) fn allocated_bytes_from_du_output(du_output: &str) -> Result<u64, ()> {
    let kibibytes_text = du_output
        .lines()
        .next()
        .and_then(|first_line| first_line.split_whitespace().next())
        .unwrap_or("");
    if kibibytes_text.is_empty() || !kibibytes_text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(());
    }
    let allocated_kibibytes: u64 = kibibytes_text.parse().map_err(|_| ())?;
    allocated_kibibytes.checked_mul(1024).ok_or(())
}

pub(crate) fn lane_root_string_is_acceptable(lane_root: &str) -> bool {
    !lane_root.is_empty()
        && lane_root != "/"
        && lane_root != "."
        && lane_root != ".."
        && lane_root.starts_with('/')
}

pub(crate) fn select_rustc_wrapper(
    rustc_wrapper_environment_value: Option<&str>,
    sccache_is_available: bool,
) -> Option<String> {
    match rustc_wrapper_environment_value.filter(|value| !value.is_empty()) {
        Some(configured_wrapper) => Some(configured_wrapper.to_owned()),
        None if sccache_is_available => Some("sccache".to_owned()),
        None => None,
    }
}

pub(crate) fn exit_code_from_status(exit_status: &ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    exit_status
        .code()
        .unwrap_or_else(|| 128 + exit_status.signal().unwrap_or(0))
}

// ── Entry point ────────────────────────────────────────────────────────

pub fn run(repository_root: &Path, arguments: &[String]) -> ExitCode {
    let journey = match parse_arguments(arguments) {
        Ok(journey) => journey,
        Err(usage_error) => return report_usage_error(usage_error),
    };
    // The storage measurement is the only external tool the typed runner
    // still spawns; failing fast here matches the shell runner's
    // require_command gate before any target directory is created.
    if !command_is_available("du") {
        print_error("required command is unavailable: du");
        return ExitCode::from(1);
    }
    if std::env::var("ASTRONOMICAL_CARGO_TARGET_LIFECYCLE").as_deref() == Ok("disposable") {
        return run_nested_command(repository_root, &journey);
    }
    lifecycle::run_disposable_command(repository_root, journey)
}

#[derive(Debug)]
pub(crate) struct LaneJourney {
    pub(crate) lane_name: String,
    pub(crate) command: Vec<String>,
}

#[derive(Debug)]
pub(crate) enum UsageError {
    PrintUsage,
    LaneName(String),
    MissingCommand,
}

pub(crate) fn parse_arguments(arguments: &[String]) -> Result<LaneJourney, UsageError> {
    if arguments.len() < 4 || arguments[0] != "--lane" {
        return Err(UsageError::PrintUsage);
    }
    let lane_name = arguments[1].clone();
    if !is_valid_lane_name(&lane_name) {
        return Err(UsageError::LaneName(lane_name));
    }
    if arguments[2] != "--" {
        return Err(UsageError::PrintUsage);
    }
    let command = arguments[3..].to_vec();
    if command.is_empty() {
        return Err(UsageError::MissingCommand);
    }
    Ok(LaneJourney { lane_name, command })
}

fn print_usage() {
    print_output_line(
        "Usage: cargo xtask disposable-target --lane NAME -- COMMAND [ARGUMENT ...]",
        false,
    );
    print_output_line("", false);
    print_output_line(
        "Runs a foreground command from the repository root with a unique",
        false,
    );
    print_output_line(
        "CARGO_TARGET_DIR, then removes only that marker-owned directory.",
        false,
    );
}

fn report_usage_error(usage_error: UsageError) -> ExitCode {
    match usage_error {
        UsageError::PrintUsage => print_usage(),
        UsageError::LaneName(lane_name) => print_error(&format!(
            "lane name must contain lowercase letters, digits, and internal hyphens only: {lane_name}"
        )),
        UsageError::MissingCommand => print_error("a foreground command is required"),
    }
    ExitCode::from(2)
}

fn print_error(message: &str) {
    print_output_line(&format!("Error: {message}"), true);
}

fn print_output_line(text: &str, to_standard_error: bool) {
    // Line-buffered writes with an explicit flush keep the interleaving of
    // lifecycle lines and errors stable when both streams share one log.
    if to_standard_error {
        let mut standard_error = std::io::stderr().lock();
        let _ = writeln!(standard_error, "{text}");
        let _ = standard_error.flush();
    } else {
        let mut standard_output = std::io::stdout().lock();
        let _ = writeln!(standard_output, "{text}");
        let _ = standard_output.flush();
    }
}

fn lifecycle_line(lane_name: &str, status_text: &str, to_standard_error: bool) {
    print_output_line(
        &format!("[cargo-artifact-lifecycle] lane={lane_name} status={status_text}"),
        to_standard_error,
    );
}

fn command_is_available(command_name: &str) -> bool {
    let search_paths = std::env::var("PATH").unwrap_or_default();
    for search_directory in search_paths.split(':') {
        if search_directory.is_empty() {
            continue;
        }
        let candidate = Path::new(search_directory).join(command_name);
        let is_executable_file = std::fs::metadata(&candidate)
            .ok()
            .map(|candidate_metadata| {
                use std::os::unix::fs::PermissionsExt;
                candidate_metadata.is_file() && candidate_metadata.permissions().mode() & 0o111 != 0
            })
            .unwrap_or(false);
        if is_executable_file {
            return true;
        }
    }
    false
}

fn unix_epoch_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since_epoch| since_epoch.as_secs())
        .unwrap_or(0)
}

/// Local timestamp in the shell runner's `date '+%Y-%m-%dT%H:%M:%S%z'`
/// shape, so log readers and greps written against the shell output keep
/// parsing the start line unchanged.
fn local_timestamp() -> String {
    let epoch_seconds = unix_epoch_seconds() as libc::time_t;
    let mut broken_down_time: libc::tm = unsafe { std::mem::zeroed() };
    let time_conversion_succeeded =
        unsafe { !libc::localtime_r(&epoch_seconds, &mut broken_down_time).is_null() };
    if !time_conversion_succeeded {
        return epoch_seconds.to_string();
    }
    let gmtoff_seconds = broken_down_time.tm_gmtoff;
    let offset_sign = if gmtoff_seconds < 0 { '-' } else { '+' };
    let offset_minutes_total = gmtoff_seconds.unsigned_abs() / 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}{:02}{:02}",
        broken_down_time.tm_year + 1900,
        broken_down_time.tm_mon + 1,
        broken_down_time.tm_mday,
        broken_down_time.tm_hour,
        broken_down_time.tm_min,
        broken_down_time.tm_sec,
        offset_sign,
        offset_minutes_total / 60,
        offset_minutes_total % 60
    )
}

fn validate_owned_target(target_directory: &Path, expected_lane_name: &str) -> Result<(), String> {
    let target_is_owned_directory = std::fs::symlink_metadata(target_directory)
        .map(|target_metadata| target_metadata.is_dir())
        .unwrap_or(false);
    if !target_is_owned_directory {
        return Err(format!(
            "refusing to use a missing, non-directory, or symbolic-link Cargo target: {}",
            target_directory.display()
        ));
    }
    let marker_path = target_directory.join(LIFECYCLE_MARKER_NAME);
    let marker_is_regular_file = std::fs::symlink_metadata(&marker_path)
        .map(|marker_metadata| marker_metadata.is_file())
        .unwrap_or(false);
    if !marker_is_regular_file {
        return Err(format!(
            "refusing to use Cargo target without an owned lifecycle marker: {}",
            target_directory.display()
        ));
    }
    let observed_marker_identity = std::fs::read_to_string(&marker_path).unwrap_or_default();
    if observed_marker_identity != marker_identity(expected_lane_name) {
        return Err(format!(
            "refusing Cargo target with an unexpected lifecycle marker: {}",
            target_directory.display()
        ));
    }
    Ok(())
}

// ── Nested reuse ───────────────────────────────────────────────────────

fn run_nested_command(repository_root: &Path, journey: &LaneJourney) -> ExitCode {
    let existing_target_directory = std::env::var("CARGO_TARGET_DIR").unwrap_or_default();
    let existing_lane_name = std::env::var("ASTRONOMICAL_CARGO_TARGET_LANE").unwrap_or_default();
    if existing_target_directory.is_empty() || existing_lane_name.is_empty() {
        print_error("nested disposable lifecycle is missing its target ownership");
        return ExitCode::from(1);
    }
    let existing_target_path = PathBuf::from(&existing_target_directory);
    if let Err(refusal_message) = validate_owned_target(&existing_target_path, &existing_lane_name)
    {
        print_error(&refusal_message);
        return ExitCode::from(1);
    }
    lifecycle_line(
        &journey.lane_name,
        &format!(
            "reused owner_lane={existing_lane_name} target_directory={}",
            existing_target_path.display()
        ),
        false,
    );
    let nested_exit_status = Command::new(&journey.command[0])
        .args(&journey.command[1..])
        .current_dir(repository_root)
        .status();
    match nested_exit_status {
        Ok(nested_exit_status) => ExitCode::from(exit_code_from_status(&nested_exit_status) as u8),
        Err(launch_error) => {
            print_error(&format!(
                "could not run nested disposable command: {launch_error}"
            ));
            ExitCode::from(1)
        }
    }
}

// ── Lane root and target creation ──────────────────────────────────────

fn select_lane_root() -> Result<PathBuf, String> {
    let configured_lane_root = std::env::var("ASTRONOMICAL_CARGO_LANE_ROOT")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::env::var("TMPDIR")
                .ok()
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "/tmp".to_owned());
    if !Path::new(&configured_lane_root).is_dir() {
        return Err(format!(
            "Cargo lane root must be an existing directory: {configured_lane_root}"
        ));
    }
    let canonical_lane_root =
        std::fs::canonicalize(&configured_lane_root).map_err(|canonicalize_error| {
            format!(
                "Cargo lane root must be an existing directory: {configured_lane_root} ({canonicalize_error})"
            )
        })?;
    let lane_root_text = canonical_lane_root
        .to_string_lossy()
        .trim_end_matches('/')
        .to_owned();
    if !lane_root_string_is_acceptable(&lane_root_text) {
        return Err(format!(
            "Cargo lane root must be an absolute non-root directory: {lane_root_text}"
        ));
    }
    let lane_root_path = PathBuf::from(&lane_root_text);
    let lane_root_is_plain_directory = std::fs::symlink_metadata(&lane_root_path)
        .map(|lane_root_metadata| lane_root_metadata.is_dir())
        .unwrap_or(false);
    if !lane_root_is_plain_directory || !path_is_writable(&lane_root_path) {
        return Err(format!(
            "Cargo lane root must be an existing writable directory: {lane_root_text}"
        ));
    }
    Ok(lane_root_path)
}

fn path_is_writable(path: &Path) -> bool {
    use std::ffi::CString;
    let Some(path_text) = path.to_str() else {
        return false;
    };
    let Ok(path_bytes) = CString::new(path_text) else {
        return false;
    };
    unsafe { libc::access(path_bytes.as_ptr(), libc::W_OK) == 0 }
}

fn create_marker_owned_target(lane_root: &Path, lane_name: &str) -> std::io::Result<PathBuf> {
    // mktemp-style unique directory: lowercase alphanumeric suffix, created
    // exclusively so concurrent lanes can never collide.
    let mut entropy = unix_epoch_seconds() ^ ((std::process::id() as u64) << 24);
    for _ in 0..64 {
        entropy = entropy
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let suffix: String = (0..6)
            .map(|digit_index| {
                let digit = (entropy >> (digit_index * 5)) as usize % 36;
                if digit < 10 {
                    (b'0' + digit as u8) as char
                } else {
                    (b'a' + (digit - 10) as u8) as char
                }
            })
            .collect();
        let candidate = lane_root.join(format!("astronomical-cargo-{lane_name}.{suffix}"));
        match std::fs::create_dir(&candidate) {
            Ok(()) => {
                if let Err(marker_error) = std::fs::write(
                    candidate.join(LIFECYCLE_MARKER_NAME),
                    marker_identity(lane_name),
                ) {
                    // A directory without its marker is unowned and can never
                    // be cleaned by the lifecycle, so it must not survive a
                    // failed marker write.
                    let _ = std::fs::remove_dir(&candidate);
                    return Err(marker_error);
                }
                return Ok(candidate);
            }
            Err(creation_error) if creation_error.kind() == std::io::ErrorKind::AlreadyExists => {
                continue;
            }
            Err(creation_error) => return Err(creation_error),
        }
    }
    Err(std::io::Error::other(
        "64 lane-directory attempts all collided",
    ))
}

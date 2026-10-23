//! `cargo xtask scan-retired-native-output` — the read-only Phase 0 probe
//! for issue #1486: counts the retired complete CMake native-output
//! directories that still linger in Cargo targets and cache checkouts.
//! Phase 1 (retiring the shell cleaner and its contract) landed behind this
//! probe's zero-count gate; the probe now guards Phase 2, the eventual
//! removal of the transitional build-script cleanup in
//! `crates/runtime-integration/build_legacy_native_output.rs`. The probe
//! never removes anything.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

const LEGACY_DIRECTORY_NAME: &str = "mlx-c-runtime-build";
const REMOVAL_DIRECTORY_PREFIX: &str = ".mlx-c-runtime-build.astronomical-removing-";
const RUNTIME_BUILD_PREFIX: &str = "astronomical-runtime-integration-";
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
/// A filesystem root can never be a Cargo target worth scanning; refusing
/// it keeps a misconfigured `--target-directory /` from walking the disk.
const REFUSED_SCAN_ROOTS: [&str; 3] = ["/", ".", ".."];

// ── Pure helpers (unit-tested beside this source) ─────────────────────

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ScanArguments {
    Scan {
        target_directory_override: Option<String>,
        extra_scan_roots: Vec<String>,
        fail_on_found: bool,
    },
    /// `--help` or `-h`: print the usage and exit successfully.
    PrintHelp,
    PrintUsage,
}

pub(crate) fn parse_scan_arguments(arguments: &[String]) -> ScanArguments {
    let mut target_directory_override: Option<String> = None;
    let mut extra_scan_roots: Vec<String> = Vec::new();
    let mut fail_on_found_was_seen = false;
    let mut argument_index = 0;
    while argument_index < arguments.len() {
        let argument = arguments[argument_index].as_str();
        match argument {
            "--help" | "-h" => return ScanArguments::PrintHelp,
            "--target-directory" => {
                let Some(path_argument) = arguments.get(argument_index + 1) else {
                    return ScanArguments::PrintUsage;
                };
                // A following flag is a caller mistake, not a path: consuming
                // it would surface later as a confusing missing-directory
                // error instead of a usage error.
                if target_directory_override.is_some() || path_argument.starts_with('-') {
                    return ScanArguments::PrintUsage;
                }
                target_directory_override = Some(path_argument.clone());
                argument_index += 2;
            }
            "--fail-on-found" => {
                // The same strictness as a duplicated --target-directory: a
                // repeated flag is a caller mistake, not a silent no-op.
                if fail_on_found_was_seen {
                    return ScanArguments::PrintUsage;
                }
                fail_on_found_was_seen = true;
                argument_index += 1;
            }
            _ if argument.starts_with('-') => return ScanArguments::PrintUsage,
            _ => {
                extra_scan_roots.push(argument.to_owned());
                argument_index += 1;
            }
        }
    }
    ScanArguments::Scan {
        target_directory_override,
        extra_scan_roots,
        fail_on_found: fail_on_found_was_seen,
    }
}

/// Classifies one directory entry structurally: a retired candidate is a
/// directory whose own name is the legacy name (or the interrupted-removal
/// prefix), whose parent is `out/`, whose grandparent is a
/// `astronomical-runtime-integration-*` build-script generation, and whose
/// great-grandparent is `build/`. Component-based matching keeps the rule
/// readable without glob translation.
pub(crate) fn is_retired_candidate_directory(parent_directory: &Path, entry_name: &OsStr) -> bool {
    let entry_name_text = entry_name.to_string_lossy();
    let name_matches = entry_name_text == LEGACY_DIRECTORY_NAME
        || entry_name_text.starts_with(REMOVAL_DIRECTORY_PREFIX);
    if !name_matches {
        return false;
    }
    if parent_directory.file_name().as_deref() != Some(OsStr::new("out")) {
        return false;
    }
    let Some(generation_directory) = parent_directory.parent() else {
        return false;
    };
    let generation_matches = generation_directory
        .file_name()
        .is_some_and(|generation_name| {
            generation_name
                .to_string_lossy()
                .starts_with(RUNTIME_BUILD_PREFIX)
        });
    if !generation_matches {
        return false;
    }
    generation_directory
        .parent()
        .and_then(|build_directory| build_directory.file_name())
        .is_some_and(|build_name| build_name == OsStr::new("build"))
}

pub(crate) fn scan_root_is_refused(scan_root: &str) -> bool {
    REFUSED_SCAN_ROOTS.contains(&scan_root)
}

// ── Entry point ────────────────────────────────────────────────────────

pub fn run(repository_root: &Path, arguments: &[String]) -> ExitCode {
    match parse_scan_arguments(arguments) {
        ScanArguments::PrintHelp => {
            print_scan_usage();
            ExitCode::SUCCESS
        }
        ScanArguments::PrintUsage => {
            print_scan_usage();
            ExitCode::from(2)
        }
        ScanArguments::Scan {
            target_directory_override,
            extra_scan_roots,
            fail_on_found,
        } => run_scan(
            repository_root,
            target_directory_override,
            extra_scan_roots,
            fail_on_found,
        ),
    }
}

fn print_scan_usage() {
    print_output_line(
        "Usage: cargo xtask scan-retired-native-output [--target-directory PATH] [--fail-on-found] [EXTRA_SCAN_ROOT ...]",
        false,
    );
}

fn run_scan(
    repository_root: &Path,
    target_directory_override: Option<String>,
    extra_scan_roots: Vec<String>,
    fail_on_found: bool,
) -> ExitCode {
    let target_directory = match target_directory_override {
        Some(target_override) => PathBuf::from(target_override),
        None => match resolve_cargo_target_directory(repository_root) {
            Ok(resolved_target_directory) => resolved_target_directory,
            Err(resolution_error) => {
                print_output_line(&format!("Error: {resolution_error}"), true);
                return ExitCode::from(1);
            }
        },
    };

    let configured_scan_roots: Vec<String> = [target_directory.to_string_lossy().into_owned()]
        .into_iter()
        .chain(extra_scan_roots)
        .collect();
    let mut canonical_scan_roots: Vec<PathBuf> = Vec::new();
    for scan_root in &configured_scan_roots {
        if scan_root_is_refused(scan_root) {
            print_output_line(
                &format!(
                    "Error: refusing to scan a filesystem root, current directory, or parent directory as a Cargo target: {scan_root}"
                ),
                true,
            );
            return ExitCode::from(1);
        }
        let scan_root_path = PathBuf::from(scan_root);
        // Canonicalize before the refusal check: Path does not normalize, so
        // spelling variants of the filesystem root ("/.", "//", symlinked
        // roots) would otherwise bypass the string comparison. The canonical
        // form also keys the duplicate-root deduplication below.
        let Ok(canonical_scan_root) = scan_root_path.canonicalize() else {
            print_output_line(
                &format!(
                    "Error: scan root must be an existing directory: {}",
                    scan_root_path.display()
                ),
                true,
            );
            return ExitCode::from(1);
        };
        if scan_root_is_refused(&canonical_scan_root.to_string_lossy()) {
            print_output_line(
                &format!(
                    "Error: refusing to scan a filesystem root as a Cargo target: {}",
                    canonical_scan_root.display()
                ),
                true,
            );
            return ExitCode::from(1);
        }
        if canonical_scan_roots.contains(&canonical_scan_root) {
            // An overlapping or repeated scan root would count the same
            // retired directories twice; the set keyed on the canonical
            // form is the deduplication authority.
            continue;
        }
        canonical_scan_roots.push(canonical_scan_root);
    }

    let scan_started_at = Instant::now();
    let mut next_progress_at = scan_started_at + PROGRESS_INTERVAL;
    // Global totals key candidates canonically so overlapping scan roots
    // can never count one physical retired directory twice in the summary
    // the retirement gate reads.
    let mut counted_retired_canonical_paths: BTreeSet<PathBuf> = BTreeSet::new();
    let mut total_retired_bytes: u64 = 0;
    let mut total_retired_directories: u64 = 0;
    let mut total_unreadable_directories: u64 = 0;
    for scan_root in &canonical_scan_roots {
        let scan_summary =
            scan_root_for_retired_candidates(scan_root, scan_started_at, &mut next_progress_at);
        total_unreadable_directories += scan_summary.unreadable_directories;
        print_output_line(
            &format!(
                "[retired-native-scan] root={} directories_walked={} unreadable_directories={} found={} allocated_bytes={}",
                scan_root.display(),
                scan_summary.directories_walked,
                scan_summary.unreadable_directories,
                scan_summary.retired_directories.len(),
                scan_summary.retired_allocated_bytes
            ),
            false,
        );
        for retired_directory in &scan_summary.retired_directories {
            print_output_line(
                &format!("[retired-native-scan] found path={}", retired_directory.display()),
                false,
            );
        }
        for retired_directory in &scan_summary.retired_directories {
            let Ok(canonical_retired_directory) = retired_directory.canonicalize() else {
                continue;
            };
            if !counted_retired_canonical_paths.insert(canonical_retired_directory) {
                continue;
            }
            total_retired_directories += 1;
            total_retired_bytes += allocated_bytes_under(retired_directory);
        }
    }

    // The summary line stays in raw bytes exactly like every other
    // lifecycle log; the trailing human figure converts with the Apple
    // marketing convention 1 GB = 1,073,741,824 bytes (2^30).
    let total_retired_gb_whole = total_retired_bytes / (1024 * 1024 * 1024);
    print_output_line(
        &format!(
            "[retired-native-scan] status=summary found={total_retired_directories} unreadable_directories={total_unreadable_directories} allocated_bytes={total_retired_bytes} allocated_gb={total_retired_gb_whole} elapsed_seconds={}",
            scan_started_at.elapsed().as_secs()
        ),
        false,
    );
    if fail_on_found && total_retired_directories > 0 {
        print_output_line(&gate_error_message(total_retired_directories), true);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// The operator guidance a tripped `--fail-on-found` gate prints: the
/// surviving removal paths are the build script's automatic clearing of the
/// current generation plus a deliberate whole-target `cargo clean` for
/// stale generations (see CONTRIBUTING.md and issue #1486).
pub(crate) fn gate_error_message(retired_directory_count: u64) -> String {
    format!(
        "Error: {retired_directory_count} retired native-output directories remain; \
         remove them with a deliberate whole-target `cargo clean` (the build script \
         clears the current generation automatically; see issue #1486)"
    )
}

fn print_output_line(output_text: &str, to_standard_error: bool) {
    // Line-buffered writes with an explicit flush keep the interleaving of
    // scan lines, progress lines, and errors stable when both streams
    // share one log.
    if to_standard_error {
        let mut standard_error = std::io::stderr().lock();
        let _ = writeln!(standard_error, "{output_text}");
        let _ = standard_error.flush();
    } else {
        let mut standard_output = std::io::stdout().lock();
        let _ = writeln!(standard_output, "{output_text}");
        let _ = standard_output.flush();
    }
}

pub(crate) struct ScanSummary {
    pub(crate) directories_walked: u64,
    pub(crate) unreadable_directories: u64,
    pub(crate) retired_directories: Vec<PathBuf>,
    pub(crate) retired_allocated_bytes: u64,
}

fn resolve_cargo_target_directory(repository_root: &Path) -> Result<PathBuf, String> {
    let metadata_output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(repository_root)
        .output()
        .map_err(|metadata_error| format!("could not run cargo metadata: {metadata_error}"))?;
    if !metadata_output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&metadata_output.stderr)
        ));
    }
    let metadata_json: serde_json::Value = serde_json::from_slice(&metadata_output.stdout)
        .map_err(|metadata_parse_error| {
            format!("could not parse cargo metadata output: {metadata_parse_error}")
        })?;
    let target_directory_text = metadata_json["target_directory"]
        .as_str()
        .ok_or_else(|| "cargo metadata carried no target_directory".to_owned())?;
    Ok(PathBuf::from(target_directory_text))
}

pub(crate) fn scan_root_for_retired_candidates(
    scan_root: &Path,
    scan_started_at: Instant,
    next_progress_at: &mut Instant,
) -> ScanSummary {
    let mut directories_walked: u64 = 0;
    let mut unreadable_directories: u64 = 0;
    let mut retired_directories: Vec<PathBuf> = Vec::new();
    let mut retired_allocated_bytes: u64 = 0;
    let mut walk_frontier: Vec<PathBuf> = vec![scan_root.to_path_buf()];
    while let Some(visited_directory) = walk_frontier.pop() {
        directories_walked += 1;
        if Instant::now() >= *next_progress_at {
            *next_progress_at = Instant::now() + PROGRESS_INTERVAL;
            // Progress line honors the repository rule that any batch job
            // reports progress on five-second intervals; the walked count
            // doubles as an ETA signal for slow targets.
            print_output_line(
                &format!(
                    "[retired-native-scan] status=progress directories_walked={directories_walked} elapsed_seconds={}",
                    scan_started_at.elapsed().as_secs()
                ),
                true,
            );
        }
        let Ok(directory_entries) = std::fs::read_dir(&visited_directory) else {
            // A permission-denied subtree must shrink the measurement
            // visibly rather than silently, so the summary carries the
            // count and the retirement gate can distrust a scan with holes.
            unreadable_directories += 1;
            continue;
        };
        for directory_entry in directory_entries.flatten() {
            let entry_type = match directory_entry.file_type() {
                Ok(entry_type) => entry_type,
                Err(_) => continue,
            };
            // Symlinks are never followed and never classified: the shell
            // cleaner refuses them for the same reason.
            if entry_type.is_symlink() {
                continue;
            }
            if entry_type.is_dir() {
                let entry_path = directory_entry.path();
                if is_retired_candidate_directory(&visited_directory, &directory_entry.file_name())
                {
                    retired_allocated_bytes += allocated_bytes_under(&entry_path);
                    retired_directories.push(entry_path);
                    // A retired candidate is counted whole; its subtree is
                    // not scanned for further candidates.
                    continue;
                }
                walk_frontier.push(entry_path);
            }
        }
    }
    retired_directories.sort();
    ScanSummary {
        directories_walked,
        unreadable_directories,
        retired_directories,
        retired_allocated_bytes,
    }
}

/// Total allocated storage (st_blocks × 512, the same accounting `du -sk`
/// reports) for one retired directory and its subtree. Unlike `du`, a
/// hard-linked inode reachable through several directory entries is
/// counted once per entry; for retired CMake output that difference is
/// noise, and the measurement only ever compares counts with themselves.
pub(crate) fn allocated_bytes_under(retired_directory: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    let mut total_allocated_bytes = std::fs::symlink_metadata(retired_directory)
        .map(|directory_metadata| directory_metadata.blocks() * 512)
        .unwrap_or(0);
    let mut subtree_frontier: Vec<PathBuf> = vec![retired_directory.to_path_buf()];
    while let Some(visited_directory) = subtree_frontier.pop() {
        let Ok(directory_entries) = std::fs::read_dir(&visited_directory) else {
            continue;
        };
        for directory_entry in directory_entries.flatten() {
            let Ok(entry_metadata) = directory_entry.metadata() else {
                continue;
            };
            total_allocated_bytes += entry_metadata.blocks() * 512;
            if entry_metadata.is_dir() {
                subtree_frontier.push(directory_entry.path());
            }
        }
    }
    total_allocated_bytes
}

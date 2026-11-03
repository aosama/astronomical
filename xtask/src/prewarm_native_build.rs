//! `cargo xtask prewarm-native-build --profile NAME [--profile NAME ...]` —
//! builds the pinned native MLX runtime into the machine-wide native build
//! store before any Cargo compilation, so later build scripts find it warm
//! and never invoke CMake. Typed port of the retired shell prewarm script,
//! with the same `[prewarm-native-build]` lines.
//!
//! Cargo captures build-script output, so the native build tool streams its
//! progress into a file instead. The shell port ran `tail -f` beside the
//! build to surface it; here the same loop that waits for the build relays
//! new progress lines, so there is no second process to clean up.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use astronomical_native_identity::NativeBuildProfile;

const STATUS_PREFIX: &str = "[prewarm-native-build]";
const NATIVE_BUILD_PROGRESS_FILE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE";
const PROGRESS_POLL_INTERVAL: Duration = Duration::from_millis(250);

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let native_build_profiles = parse_profiles(arguments)?;
    if !repository_root
        .join("crates/runtime-integration/native")
        .is_dir()
    {
        return Err(format!(
            "{STATUS_PREFIX} Error: repository root does not contain the native runtime sources: {}",
            repository_root.display()
        ));
    }
    let prewarm_started_at = Instant::now();
    let progress_file_path = std::env::temp_dir().join(format!(
        "astronomical-native-build-prewarm.{}",
        std::process::id()
    ));
    let prewarm_result =
        prewarm_profiles(repository_root, &native_build_profiles, &progress_file_path);
    let _ = std::fs::remove_file(&progress_file_path);
    match &prewarm_result {
        Ok(()) => println!(
            "{STATUS_PREFIX} status=success elapsed_seconds={}",
            prewarm_started_at.elapsed().as_secs()
        ),
        Err(_) => println!(
            "{STATUS_PREFIX} status=failed elapsed_seconds={}",
            prewarm_started_at.elapsed().as_secs()
        ),
    }
    prewarm_result
}

/// Profiles in the order given; at least one is required and each must be
/// one the native build store understands.
pub(crate) fn parse_profiles(arguments: &[String]) -> Result<Vec<NativeBuildProfile>, String> {
    let mut native_build_profiles = Vec::new();
    let mut argument_position = 0;
    while argument_position < arguments.len() {
        if arguments[argument_position] != "--profile" {
            return Err(prewarm_usage(&format!(
                "unsupported argument: {}",
                arguments[argument_position]
            )));
        }
        let Some(profile_text) = arguments.get(argument_position + 1) else {
            return Err(prewarm_usage(
                "--profile requires a native build profile name",
            ));
        };
        match NativeBuildProfile::from_profile_text(profile_text) {
            Ok(native_build_profile) => native_build_profiles.push(native_build_profile),
            Err(_) => {
                return Err(prewarm_usage(&format!(
                    "unsupported profile: {profile_text}"
                )));
            }
        }
        argument_position += 2;
    }
    if native_build_profiles.is_empty() {
        return Err(prewarm_usage(
            "missing required argument --profile <profile-name>",
        ));
    }
    Ok(native_build_profiles)
}

fn prewarm_usage(problem: &str) -> String {
    format!(
        "{STATUS_PREFIX} Error: {problem}\n{STATUS_PREFIX} Error: usage: cargo xtask prewarm-native-build --profile <profile-name> [--profile <profile-name> ...]\n{STATUS_PREFIX} Error: supported profiles: core, core+memory-contract"
    )
}

fn prewarm_profiles(
    repository_root: &Path,
    native_build_profiles: &[NativeBuildProfile],
    progress_file_path: &Path,
) -> Result<(), String> {
    if let Err(create_error) = File::create(progress_file_path) {
        return Err(format!(
            "{STATUS_PREFIX} Error: could not create the progress file: {create_error}"
        ));
    }
    let mut profile_names = Vec::new();
    for native_build_profile in native_build_profiles {
        profile_names.push(native_build_profile.as_profile_text());
    }
    println!(
        "{STATUS_PREFIX} status=start profiles={}",
        profile_names.join(" ")
    );
    provision_bindgen_headers(repository_root)?;
    let mut progress_relay = ProgressRelay::new(progress_file_path.to_path_buf());
    for native_build_profile in native_build_profiles {
        build_native_profile(repository_root, *native_build_profile, &mut progress_relay)?;
    }
    Ok(())
}

/// The MLX-C bindings crate's build script resolves its headers from this
/// extraction. The step is idempotent: a warm cache verifies in about a
/// second and never touches the network.
fn provision_bindgen_headers(repository_root: &Path) -> Result<(), String> {
    println!("{STATUS_PREFIX} status=provision-bindgen-headers start");
    let provision_started_at = Instant::now();
    let provision_status = Command::new("sh")
        .arg(repository_root.join("scripts/provision-bindgen-headers.sh"))
        .current_dir(repository_root)
        .status();
    match provision_status {
        Ok(exit_status) if exit_status.success() => {
            println!(
                "{STATUS_PREFIX} status=provision-bindgen-headers success elapsed_seconds={}",
                provision_started_at.elapsed().as_secs()
            );
            Ok(())
        }
        _ => Err(format!(
            "{STATUS_PREFIX} Error: bindgen header provisioning failed; run cargo xtask bootstrap-native-dependencies if the archive cache is empty"
        )),
    }
}

fn build_native_profile(
    repository_root: &Path,
    native_build_profile: NativeBuildProfile,
    progress_relay: &mut ProgressRelay,
) -> Result<(), String> {
    let profile_name = native_build_profile.as_profile_text();
    let profile_started_at = Instant::now();
    println!("{STATUS_PREFIX} status=native-build start profile={profile_name}");
    let spawn_result = Command::new("cargo")
        .args([
            "run",
            "-p",
            "astronomical-native-build-tool",
            "--",
            "--profile",
            profile_name,
        ])
        .arg("--repository-root")
        .arg(repository_root)
        .current_dir(repository_root)
        .env(
            NATIVE_BUILD_PROGRESS_FILE_VARIABLE,
            &progress_relay.progress_file_path,
        )
        .spawn();
    let native_build_process = match spawn_result {
        Ok(native_build_process) => native_build_process,
        Err(spawn_error) => {
            return Err(format!(
                "{STATUS_PREFIX} Error: could not start the native build tool: {spawn_error}"
            ));
        }
    };
    wait_relaying_progress(native_build_process, progress_relay, profile_name)?;
    println!(
        "{STATUS_PREFIX} status=native-build success profile={profile_name} elapsed_seconds={}",
        profile_started_at.elapsed().as_secs()
    );
    Ok(())
}

fn wait_relaying_progress(
    mut native_build_process: Child,
    progress_relay: &mut ProgressRelay,
    profile_name: &str,
) -> Result<(), String> {
    loop {
        progress_relay.relay_new_lines();
        match native_build_process.try_wait() {
            Ok(Some(exit_status)) => {
                progress_relay.relay_new_lines();
                if exit_status.success() {
                    return Ok(());
                }
                return Err(format!(
                    "{STATUS_PREFIX} Error: native build failed for profile {profile_name} with {exit_status}"
                ));
            }
            Ok(None) => std::thread::sleep(PROGRESS_POLL_INTERVAL),
            Err(wait_error) => {
                return Err(format!(
                    "{STATUS_PREFIX} Error: could not wait for the native build: {wait_error}"
                ));
            }
        }
    }
}

/// Prints whatever the native build tool appended to its progress file
/// since the last call. A read failure only loses telemetry, never the
/// build, matching the tool's own best-effort reporting.
pub(crate) struct ProgressRelay {
    progress_file_path: PathBuf,
    relayed_byte_count: u64,
}

impl ProgressRelay {
    pub(crate) fn new(progress_file_path: PathBuf) -> Self {
        Self {
            progress_file_path,
            relayed_byte_count: 0,
        }
    }

    pub(crate) fn relay_new_lines(&mut self) {
        let new_progress_text = self.read_new_text();
        if !new_progress_text.is_empty() {
            print!("{new_progress_text}");
        }
    }

    pub(crate) fn read_new_text(&mut self) -> String {
        let Ok(mut progress_file) = File::open(&self.progress_file_path) else {
            return String::new();
        };
        if progress_file
            .seek(SeekFrom::Start(self.relayed_byte_count))
            .is_err()
        {
            return String::new();
        }
        let mut new_progress_bytes = Vec::new();
        if progress_file.read_to_end(&mut new_progress_bytes).is_err() {
            return String::new();
        }
        self.relayed_byte_count += new_progress_bytes.len() as u64;
        String::from_utf8_lossy(&new_progress_bytes).into_owned()
    }
}

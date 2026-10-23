//! The sandboxed fixture environment for the disposable-target journeys:
//! the per-test repository and lane-root directories, the executable
//! fixture writer, the sanitized runner command, and the file/log reading
//! helpers the journeys assert through.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::process::RUNNER_DEADLINE_SECONDS;

pub(crate) struct DisposableSandbox {
    pub(crate) root_directory: PathBuf,
    pub(crate) repository_directory: PathBuf,
    pub(crate) lane_root_directory: PathBuf,
}

impl DisposableSandbox {
    pub(crate) fn install() -> DisposableSandbox {
        // Tests run in parallel threads inside one process, so the process
        // id alone cannot name the sandbox root: a sibling test's cleanup
        // would wipe this test's repository mid-journey.
        static SANDBOX_SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let sandbox_sequence = SANDBOX_SEQUENCE.fetch_add(1, Ordering::SeqCst);
        let root_directory = std::env::temp_dir().join(format!(
            "astronomical-disposable-journeys-{}-{sandbox_sequence}",
            std::process::id()
        ));
        let repository_directory = root_directory.join("repository");
        let lane_root_directory = root_directory.join("cargo-lanes");
        std::fs::create_dir_all(&repository_directory)
            .expect("the sandbox repository should be creatable");
        std::fs::create_dir_all(&lane_root_directory)
            .expect("the sandbox lane root should be creatable");
        DisposableSandbox {
            // Canonicalized so path assertions match the runner's own
            // canonical lane paths (on macOS TMPDIR is the /var/folders
            // symlink form while the runner resolves /private/var/...).
            root_directory: canonicalized(&root_directory),
            repository_directory: canonicalized(&repository_directory),
            lane_root_directory: canonicalized(&lane_root_directory),
        }
    }

    pub(crate) fn write_executable(&self, relative_path: &str, body: &str) -> PathBuf {
        let executable_path = self.root_directory.join(relative_path);
        if let Some(parent_directory) = executable_path.parent() {
            std::fs::create_dir_all(parent_directory)
                .expect("the executable's parent directory should be creatable");
        }
        std::fs::write(&executable_path, body).expect("the sandbox executable should be writable");
        make_executable(&executable_path);
        executable_path
    }
}

impl Drop for DisposableSandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root_directory);
    }
}

pub(crate) fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path)
        .expect("the sandbox executable metadata should be readable")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions)
        .expect("the sandbox executable should be made executable");
}

pub(crate) fn file_text(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()))
}

pub(crate) fn canonicalized(path: &Path) -> PathBuf {
    std::fs::canonicalize(path)
        .unwrap_or_else(|error| panic!("{} should canonicalize: {error}", path.display()))
}

pub(crate) fn sanitized_runner_command(sandbox: &DisposableSandbox) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .arg("--repository-root")
        .arg(&sandbox.repository_directory)
        .arg("disposable-target")
        .env_remove("ASTRONOMICAL_CARGO_TARGET_LIFECYCLE")
        .env_remove("ASTRONOMICAL_CARGO_TARGET_LANE")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("ASTRONOMICAL_DISPOSABLE_REPOSITORY_ROOT")
        .env_remove("RUSTC_WRAPPER")
        .env("ASTRONOMICAL_CARGO_LANE_ROOT", &sandbox.lane_root_directory)
        .current_dir(&sandbox.root_directory);
    command
}

pub(crate) fn journey_log_text(journey_output: &Output) -> String {
    let mut combined_log_text = String::from_utf8_lossy(&journey_output.stdout).into_owned();
    combined_log_text.push_str(&String::from_utf8_lossy(&journey_output.stderr));
    combined_log_text
}

pub(crate) fn run_journey(
    sandbox: &DisposableSandbox,
    lane_name: &str,
    command: &Path,
) -> (Output, String) {
    let journey_output = sanitized_runner_command(sandbox)
        .arg("--lane")
        .arg(lane_name)
        .arg("--")
        .arg(command)
        .output()
        .expect("the disposable-target journey should run");
    let log_text = journey_log_text(&journey_output);
    (journey_output, log_text)
}

pub(crate) fn capture_fixture_body() -> String {
    // Takes the record path as its first argument so one executable serves
    // both the outer journeys and nested-lane reuse (the retired shell
    // fixture took the record path the same way).
    "#!/bin/sh\n\
     set -eu\n\
     record_path=\"${1:?a record path is required}\"\n\
     printf '%s\\n' \"${CARGO_TARGET_DIR:?disposable Cargo target is required}\" > \"$record_path\"\n\
     printf '%s\\n' \"${RUSTC_WRAPPER:-none}\" > \"$record_path.rustc-wrapper\"\n\
     printf '%s\\n' \"$PWD\" > \"$record_path.working-directory\"\n\
     mkdir -p \"${CARGO_TARGET_DIR}/debug\"\n\
     printf '%s\\n' artifact > \"${CARGO_TARGET_DIR}/debug/fixture\"\n"
        .to_owned()
}

pub(crate) fn assert_lane_root_is_empty(lane_root_directory: &Path) {
    let lane_entries: Vec<String> = std::fs::read_dir(lane_root_directory)
        .expect("the lane root should be listable")
        .map(|lane_entry| {
            lane_entry
                .expect("the lane entry should be readable")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert!(
        lane_entries.is_empty(),
        "the disposable Cargo lane root retained artifacts: {lane_entries:?}"
    );
}

pub(crate) fn assert_path_is_absent(path: &Path) {
    assert!(
        !path.exists() && !path.is_symlink(),
        "expected the lifecycle-owned path to be absent: {}",
        path.display()
    );
}

pub(crate) fn wait_for_file(path: &Path) {
    let wait_deadline = Instant::now() + Duration::from_secs(RUNNER_DEADLINE_SECONDS);
    while !path.exists() {
        if Instant::now() > wait_deadline {
            panic!(
                "{} did not appear within {RUNNER_DEADLINE_SECONDS}s",
                path.display()
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

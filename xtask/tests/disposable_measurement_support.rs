//! Helpers for the interrupted-measurement journey of the disposable-target
//! tests: a `du` shim that blocks the runner inside storage measurement, the
//! journey spawn that puts that shim first on `PATH`, and the lookup of the
//! one target directory the runner must retain. Kept out of the journey file
//! so that file stays within the source-size limit.

use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};

use crate::disposable_journey_support::sandbox::{
    DisposableSandbox, make_executable, sanitized_runner_command,
};

/// Writes a `du` that reports a size, announces readiness through
/// `cleanup_ready_path`, then sleeps: the runner is therefore parked inside
/// its storage measurement when the test interrupts it.
pub(crate) fn write_blocking_du_shim(shim_directory: &Path, cleanup_ready_path: &Path) {
    std::fs::create_dir_all(shim_directory).expect("the shim directory should be creatable");
    let du_shim_path = shim_directory.join("du");
    std::fs::write(
        &du_shim_path,
        format!(
            "#!/bin/sh\n\
             set -eu\n\
             printf '4\\t%s\\n' \"$2\"\n\
             printf '%s\\n' ready > \"{cleanup_ready}\"\n\
             exec sleep 30\n",
            cleanup_ready = cleanup_ready_path.display()
        ),
    )
    .expect("the du shim should be writable");
    make_executable(&du_shim_path);
}

pub(crate) fn spawn_cleanup_signal_journey(
    sandbox: &DisposableSandbox,
    shim_directory: &Path,
    fixture_path: &Path,
    record_path: &Path,
) -> Child {
    let mut command_builder = sanitized_runner_command(sandbox);
    command_builder
        .arg("--lane")
        .arg("cleanup-signal")
        .arg("--")
        .arg(fixture_path)
        .arg(record_path)
        .env(
            "PATH",
            format!(
                "{}:{}",
                shim_directory.display(),
                std::env::var("PATH").unwrap_or_default()
            ),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command_builder
        .spawn()
        .expect("the cleanup-signal journey should spawn")
}

pub(crate) fn retained_handoff_target_directories(
    lane_root_directory: &Path,
    original_target: &str,
) -> Vec<PathBuf> {
    let original_target_name = Path::new(original_target)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    std::fs::read_dir(lane_root_directory)
        .expect("the lane root should be listable after retention")
        .map(|lane_entry| {
            lane_entry
                .expect("the lane entry should be readable")
                .path()
        })
        .filter(|lane_entry_path| {
            lane_entry_path
                .file_name()
                .map(|entry_name| {
                    entry_name
                        .to_string_lossy()
                        .starts_with(&format!("{original_target_name}.removing."))
                })
                .unwrap_or(false)
        })
        .collect()
}

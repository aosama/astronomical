//! Remembers that a kept program was proven to sit wholly on the Neural Engine.
//!
//! Proving placement means asking Core ML for a compute plan, which compiles the
//! program a second time: about a second per large layer, 34 s across a 9B model's
//! 56 programs. On programs macOS already held, that request gave no usable plan for
//! 17 of the 56 on one reload, and each of those was then removed and built again.
//! Placement depends on the program, the Mac and the macOS build, so a record names
//! the last two and is removed whenever its program is, and one proof then serves
//! every later load.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::host_description::{mac_hardware_model, macos_build_version, macos_product_version};

const PLACEMENT_RECORD_EXTENSION: &str = "placement-verified";

/// Whether a kept program's load must prove its placement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PlacementCheck {
    Required,
    AlreadyProven,
}

/// Beside the program and not inside it, because macOS keys its own compiled copy
/// to the program's folder and that folder must stay as it was compiled.
fn placement_record_path(kept_program_path: &Path) -> PathBuf {
    kept_program_path.with_extension(PLACEMENT_RECORD_EXTENSION)
}

/// The macOS build and Mac a proof holds for. An update or a move to another Mac,
/// by a migration or a restored backup, can change what the Neural Engine accepts.
pub(crate) fn placement_environment_of(
    macos_product_version: &str,
    macos_build_version: &str,
    mac_hardware_model: &str,
) -> String {
    format!("macOS {macos_product_version} ({macos_build_version}) on {mac_hardware_model}")
}

/// This process's environment, or `None` when the system would not describe itself,
/// in which case nothing is recorded and every load proves placement.
pub(crate) fn current_placement_environment() -> Option<&'static str> {
    static PLACEMENT_ENVIRONMENT: OnceLock<Option<String>> = OnceLock::new();
    PLACEMENT_ENVIRONMENT
        .get_or_init(describe_current_placement_environment)
        .as_deref()
}

fn describe_current_placement_environment() -> Option<String> {
    Some(placement_environment_of(
        macos_product_version()?,
        macos_build_version()?,
        mac_hardware_model()?,
    ))
}

/// A record counts only when it names exactly this environment.
pub(crate) fn placement_check_for(
    kept_program_path: &Path,
    placement_environment: Option<&str>,
) -> PlacementCheck {
    let Some(placement_environment) = placement_environment else {
        return PlacementCheck::Required;
    };
    match fs::read_to_string(placement_record_path(kept_program_path)) {
        Ok(recorded_environment)
            if !placement_environment.is_empty()
                && recorded_environment.trim() == placement_environment =>
        {
            PlacementCheck::AlreadyProven
        }
        _ => PlacementCheck::Required,
    }
}

/// A record that cannot be written costs the next load one more placement check and
/// nothing else, so it is logged and never fails the load.
pub(crate) fn record_placement_verified(kept_program_path: &Path, placement_environment: &str) {
    if let Err(write_error) = fs::write(
        placement_record_path(kept_program_path),
        placement_environment,
    ) {
        tracing::warn!(error = %write_error, "could not record a Neural Engine program's placement");
    }
}

/// Called before a program is compiled or removed, never after, so a record can
/// only describe the program folder it was written for. The worker can be stopped
/// between any two steps, and a record that outlived its program would vouch for
/// whatever was compiled to that path next.
pub(crate) fn forget_placement_verified(kept_program_path: &Path) {
    let record_path = placement_record_path(kept_program_path);
    if record_path.exists()
        && let Err(removal_error) = fs::remove_file(&record_path)
    {
        tracing::warn!(error = %removal_error, "could not remove a Neural Engine program's placement record");
    }
}

#[cfg(test)]
#[path = "neural_engine_program_placement_record_tests.rs"]
mod tests;

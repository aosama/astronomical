//! Resolves which loopback instance this CLI belongs to from the executable
//! path. Users only have Stable; Development is the cargo / Development.app
//! identity. There is no user-facing channel flag.

use std::path::{Path, PathBuf};

use astronomical_config::AstronomicalRuntimeInstance;

/// Bundle folder names stamped by the macOS app assembler.
const STABLE_APP_BUNDLE_NAME: &str = "Astronomical.app";
const DEVELOPMENT_APP_BUNDLE_NAME: &str = "Astronomical Development.app";

/// Infers runtime identity from where this binary lives.
///
/// The input is canonicalized first because the installed CLI is reached
/// through a symlink (`/usr/local/bin/astronomical`), and `std::env::current_exe()`
/// reports the link, not the bundle path it points into. Inferring the instance
/// from the un-dereferenced link would classify every installed invocation as
/// Development. A nonexistent path (unit tests, hypothetical layouts) falls
/// back to the input as given.
///
/// `Astronomical Development.app` is checked first because that path also
/// contains the Stable bundle name as a substring.
#[must_use]
pub fn runtime_instance_from_executable_path(
    executable_path: &Path,
) -> AstronomicalRuntimeInstance {
    let canonicalized_executable_path: PathBuf =
        std::fs::canonicalize(executable_path).unwrap_or_else(|_| executable_path.to_path_buf());
    let executable_path_text = canonicalized_executable_path.to_string_lossy();
    if executable_path_text.contains(DEVELOPMENT_APP_BUNDLE_NAME) {
        AstronomicalRuntimeInstance::Development
    } else if executable_path_text.contains(STABLE_APP_BUNDLE_NAME) {
        AstronomicalRuntimeInstance::Stable
    } else {
        AstronomicalRuntimeInstance::Development
    }
}

/// Loopback instance this binary serves: its own channel, only.
///
/// Strict channel isolation is a user decision (2026-10-11): a Development
/// CLI must never route traffic to the Stable daily driver, and a Stable CLI
/// must never reach a Development daemon. When the own-channel daemon is down
/// the verbs refuse with "Start Astronomical first." rather than silently
/// attaching to the other channel.
#[must_use]
pub fn candidate_instances(
    preferred_instance: AstronomicalRuntimeInstance,
) -> [AstronomicalRuntimeInstance; 1] {
    [preferred_instance]
}

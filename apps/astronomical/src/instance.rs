//! Resolves which loopback instance this CLI belongs to from the executable
//! path. Users only have Stable; Development is the cargo / Development.app
//! identity. There is no user-facing channel flag.

use std::path::Path;

use astronomical_config::AstronomicalRuntimeInstance;

/// Bundle folder names stamped by the macOS app assembler.
const STABLE_APP_BUNDLE_NAME: &str = "Astronomical.app";
const DEVELOPMENT_APP_BUNDLE_NAME: &str = "Astronomical Development.app";

/// Infers runtime identity from where this binary lives.
///
/// `Astronomical Development.app` is checked first because that path also
/// contains the Stable bundle name as a substring.
#[must_use]
pub fn runtime_instance_from_executable_path(
    executable_path: &Path,
) -> AstronomicalRuntimeInstance {
    let executable_path_text = executable_path.to_string_lossy();
    if executable_path_text.contains(DEVELOPMENT_APP_BUNDLE_NAME) {
        AstronomicalRuntimeInstance::Development
    } else if executable_path_text.contains(STABLE_APP_BUNDLE_NAME) {
        AstronomicalRuntimeInstance::Stable
    } else {
        AstronomicalRuntimeInstance::Development
    }
}

/// Loopback instances to try for this binary, most preferred first.
///
/// A user installs one app, so the binary's own channel normally wins. The
/// other channel is kept as a fallback because an unpackaged developer build
/// belongs to Development while the only running app may be Stable; refusing in
/// that case reads as "Start Astronomical first." even though one is running.
#[must_use]
pub fn candidate_instances(
    preferred_instance: AstronomicalRuntimeInstance,
) -> [AstronomicalRuntimeInstance; 2] {
    match preferred_instance {
        AstronomicalRuntimeInstance::Stable => [
            AstronomicalRuntimeInstance::Stable,
            AstronomicalRuntimeInstance::Development,
        ],
        AstronomicalRuntimeInstance::Development => [
            AstronomicalRuntimeInstance::Development,
            AstronomicalRuntimeInstance::Stable,
        ],
    }
}

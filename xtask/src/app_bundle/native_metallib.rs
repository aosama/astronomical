//! Locates the MLX AOT metallib in the machine-wide native build store.
//!
//! The shell packager computed the native identity by spawning
//! `cargo run -p xtask -- native-fingerprint` from inside an xtask-driven
//! build; this calls the identity library in-process instead. The metallib
//! is copied into the bundle, never linked, because the store lives under
//! ~/Library/Caches, which macOS may purge while the app is installed.

use std::path::{Path, PathBuf};

use astronomical_native_identity::probes::ToolchainOverrides;
use astronomical_native_identity::{
    FingerprintRequest, IdentityScope, NativeBuildProfile, native_build_identity,
};

const NATIVE_BUILD_STORE_SCHEMA_VERSION_PATH: &str =
    "crates/native-build-orchestration/src/native-build-store-schema-version";
const NATIVE_BUILD_STORE_RELATIVE_TO_HOME: &str = "Library/Caches/Astronomical/native-builds";

pub(crate) fn packaged_metallib_source(
    repository_root: &Path,
    host_target_triple: &str,
    home_directory: &Path,
) -> Result<PathBuf, String> {
    let schema_version = native_store_schema_version(repository_root)?;
    let fingerprint_request = FingerprintRequest {
        repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: ToolchainOverrides {
            target_triple: Some(host_target_triple.to_owned()),
            ..ToolchainOverrides::none()
        },
    };
    let native_identity = match native_build_identity(&fingerprint_request) {
        Ok(native_identity) => native_identity,
        Err(identity_error) => {
            return Err(format!("native build identity failed: {identity_error}"));
        }
    };
    Ok(metallib_store_path(
        home_directory,
        schema_version,
        &native_identity,
    ))
}

pub(crate) fn metallib_store_path(
    home_directory: &Path,
    schema_version: u32,
    native_identity: &str,
) -> PathBuf {
    home_directory
        .join(NATIVE_BUILD_STORE_RELATIVE_TO_HOME)
        .join(format!("v{schema_version}"))
        .join("entries")
        .join(native_identity)
        .join("share/mlx/mlx.metallib")
}

fn native_store_schema_version(repository_root: &Path) -> Result<u32, String> {
    let schema_version_path = repository_root.join(NATIVE_BUILD_STORE_SCHEMA_VERSION_PATH);
    let schema_version_text = match std::fs::read_to_string(&schema_version_path) {
        Ok(schema_version_text) => schema_version_text,
        Err(read_error) => {
            return Err(format!(
                "native build store schema version is unavailable: {read_error}"
            ));
        }
    };
    match schema_version_text.trim().parse::<u32>() {
        Ok(schema_version) => Ok(schema_version),
        Err(_) => Err("native build store schema version must be an unsigned integer".to_owned()),
    }
}

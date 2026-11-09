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

use crate::native_store_paths::{native_store_entry_directory, native_store_schema_version};

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
    native_store_entry_directory(home_directory, schema_version, native_identity)
        .join("share/mlx/mlx.metallib")
}

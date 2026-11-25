//! Finds the patched MLX and MLX-C source trees `cargo xtask
//! provision-bindgen-headers` extracts, for native code that compiles against
//! MLX's own C++ headers.
//!
//! The native build store publishes the MLX-C headers only, because the Rust
//! bindings need nothing else. Native code that defines an MLX primitive
//! includes MLX's C++ headers, and those are in the provisioned extraction:
//! the pinned archives with the same patches the native build applies, keyed
//! by the source-only native build identity, so a pin or patch edit points
//! here at a new tree and never at a stale one.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use native_identity::probes::ToolchainOverrides;
use native_identity::{
    FingerprintRequest, IdentityScope, NativeBuildProfile, native_build_identity,
};

use crate::native_compile::NativeRuntimeBuildInputs;

const PROVISIONED_SOURCES_SUBDIRECTORY: &str = "bindgen-headers";
const COMPLETION_MARKER_FILE_NAME: &str = "complete";
const PROVISIONING_COMMAND: &str = "cargo xtask provision-bindgen-headers";
const PINS_DIRECTORY: &str = "third-party/pins";
/// How the MLX pin states the deployment target the native build passes to
/// CMake, up to the opening quote of its value.
const DEPLOYMENT_TARGET_SETTING: &str = "set(ASTRONOMICAL_MLX_MACOS_DEPLOYMENT_TARGET \"";

/// The two patched source trees a native compile adds to its include path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvisionedMlxSources {
    /// Holds `mlx/array.h`, `mlx/primitives.h` and the rest of MLX's C++.
    pub mlx_source_directory: PathBuf,
    /// Holds `mlx/c/array.h` and MLX-C's private conversion headers.
    pub mlx_c_source_directory: PathBuf,
}

/// The provisioned trees for the pins and patches of `repository_root`.
///
/// # Errors
/// Fails when the identity cannot be computed, when no dependency directory
/// can be resolved, or when the extraction for this identity is not complete;
/// the last names the command that provisions it.
pub fn provisioned_mlx_sources(
    repository_root: &Path,
) -> Result<ProvisionedMlxSources, Box<dyn Error>> {
    let source_identity = native_build_identity(&FingerprintRequest {
        repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::SourceOnly,
        overrides: ToolchainOverrides::none(),
    })
    .map_err(|identity_error| format!("native build identity failed: {identity_error}"))?;
    let native_build_inputs = NativeRuntimeBuildInputs::from_environment();
    let dependencies_directory = native_build_inputs
        .astro_native_build_cache_dependencies_directory()
        .ok_or("no native dependency directory: set HOME")?;
    provisioned_mlx_sources_under(dependencies_directory, &source_identity)
}

/// The trees of one identity under one dependency directory.
///
/// # Errors
/// Fails when that identity's extraction has no completion marker.
pub fn provisioned_mlx_sources_under(
    dependencies_directory: &Path,
    source_identity: &str,
) -> Result<ProvisionedMlxSources, Box<dyn Error>> {
    let extraction_directory = dependencies_directory
        .join(PROVISIONED_SOURCES_SUBDIRECTORY)
        .join(source_identity);
    if !extraction_directory
        .join(COMPLETION_MARKER_FILE_NAME)
        .is_file()
    {
        return Err(format!(
            "the patched MLX sources for this checkout's pins and patches are not provisioned under {}; run `{PROVISIONING_COMMAND}`",
            extraction_directory.display()
        )
        .into());
    }
    Ok(ProvisionedMlxSources {
        mlx_source_directory: extraction_directory.join("mlx-src"),
        mlx_c_source_directory: extraction_directory.join("mlx_c-src"),
    })
}

/// The macOS deployment target the pinned MLX is built for.
///
/// MLX's headers choose code paths by it, so native code compiled against
/// them is built for the same one. It is read from the pin that sets it and
/// never restated, so a pin bump cannot leave the two apart.
///
/// # Errors
/// Fails when the pins cannot be read or none of them sets the target.
pub fn pinned_mlx_macos_deployment_target(
    repository_root: &Path,
) -> Result<String, Box<dyn Error>> {
    let pins_directory = repository_root.join(PINS_DIRECTORY);
    for pin_entry in fs::read_dir(&pins_directory)? {
        let pin_path = pin_entry?.path();
        if pin_path
            .extension()
            .and_then(|extension| extension.to_str())
            != Some("cmake")
        {
            continue;
        }
        let pin_text = fs::read_to_string(&pin_path)?;
        if let Some(deployment_target) = deployment_target_in(&pin_text) {
            return Ok(deployment_target.to_owned());
        }
    }
    Err(format!(
        "no pin under {} sets the MLX macOS deployment target",
        pins_directory.display()
    )
    .into())
}

fn deployment_target_in(pin_text: &str) -> Option<&str> {
    for pin_line in pin_text.lines() {
        if let Some(quoted_value) = pin_line.trim().strip_prefix(DEPLOYMENT_TARGET_SETTING) {
            return quoted_value.split('"').next();
        }
    }
    None
}

#[cfg(test)]
#[path = "provisioned_mlx_sources_tests.rs"]
mod provisioned_mlx_sources_tests;

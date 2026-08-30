//! Locates the MLX AOT metallib without depending on a purgeable cache.
//!
//! A shipped app must keep working after macOS or a cleaner deletes
//! `~/Library/Caches`. The build-produced kernels therefore live beside the worker
//! inside the app bundle. Unpackaged `cargo run` may still use the compile-time
//! native-store copy, but that path is a fallback, not the packaged default.

use std::path::{Path, PathBuf};

use crate::MlxRuntimeError;

const BUNDLED_METALLIB_RELATIVE_PATH: &str = "../Resources/share/mlx/mlx.metallib";

/// Chooses the metallib a worker should open: environment override, then the
/// app-bundle copy, then an unpackaged native-store file when one exists.
pub fn resolve_mlx_metallib_path(
    environment_override_path: Option<PathBuf>,
    current_executable_path: &Path,
    unpackaged_native_store_metallib_path: Option<&Path>,
) -> Result<PathBuf, MlxRuntimeError> {
    if let Some(environment_override_path) = environment_override_path {
        return require_absolute_metallib_path(environment_override_path);
    }

    let bundled_metallib_path = bundled_mlx_metallib_path(current_executable_path)?;
    if is_regular_non_symlink_file(&bundled_metallib_path) {
        return require_absolute_metallib_path(bundled_metallib_path);
    }

    if let Some(unpackaged_native_store_metallib_path) = unpackaged_native_store_metallib_path {
        if is_regular_non_symlink_file(unpackaged_native_store_metallib_path) {
            return require_absolute_metallib_path(
                unpackaged_native_store_metallib_path.to_path_buf(),
            );
        }
    }

    Err(MlxRuntimeError::InvalidMetallibPath {
        description: format!(
            "MLX AOT metallib was not found next to the app executable at {bundled_metallib_path:?}. Reinstall Astronomical, or rebuild native artifacts for an unpackaged cargo run."
        ),
    })
}

fn bundled_mlx_metallib_path(current_executable_path: &Path) -> Result<PathBuf, MlxRuntimeError> {
    let Some(executable_directory) = current_executable_path.parent() else {
        return Err(MlxRuntimeError::InvalidMetallibPath {
            description: format!(
                "current executable has no parent directory: {current_executable_path:?}"
            ),
        });
    };
    Ok(executable_directory.join(BUNDLED_METALLIB_RELATIVE_PATH))
}

fn require_absolute_metallib_path(metallib_path: PathBuf) -> Result<PathBuf, MlxRuntimeError> {
    if !metallib_path.is_absolute() {
        return Err(MlxRuntimeError::InvalidMetallibPath {
            description: format!("path must be absolute: {metallib_path:?}"),
        });
    }
    Ok(metallib_path)
}

fn is_regular_non_symlink_file(candidate_path: &Path) -> bool {
    candidate_path
        .symlink_metadata()
        .map(|path_metadata| path_metadata.is_file())
        .unwrap_or(false)
}

//! Locates the MLX AOT (Ahead-Of-Time compiled) Metal library without
//! depending on a purgeable cache.
//!
//! # Why this module exists
//!
//! MLX needs its Metal kernels compiled into a `.metallib` archive before any
//! graphics-processor work can start. Two facts about macOS make locating
//! that archive a policy decision rather than a trivial path lookup:
//!
//! 1. `~/Library/Caches` is purgeable. macOS, cleanup tools, and users
//!    regularly delete its contents (issue #521). A shipped app that resolved
//!    its kernels from the cache would break after the first cleanup.
//! 2. An unpackaged `cargo run` worker has no app bundle. Development runs
//!    must still find a metallib — the one the build script published beside
//!    the cargo target directory is workspace-local and survives cache purges.
//!
//! The resolution order below encodes those constraints: the app-bundle copy
//! is the packaged default, the compile-time native-store copy is the
//! development fallback, and an environment override exists for directed
//! experiments. Whichever candidate wins, the caller (see
//! `mlx_runtime::metallib`) independently validates its size and SHA-256
//! digest before MLX opens it, so a tampered or truncated archive can never
//! reach the graphics processor even if it sits at the right path.
//!
//! # Why this module is not behind the `mlx` feature
//!
//! The rules are pure path logic with no native dependency, and the integration
//! test suite (which runs on machines without a compiled native image)
//! asserts every branch: bundled preference, environment override, symlink
//! rejection, and the user-facing failure message.

use std::path::{Path, PathBuf};

use crate::MlxRuntimeError;

/// Where a packaged macOS app keeps the metallib relative to the worker
/// executable. `Contents/MacOS/<worker>` resolves to
/// `Contents/Resources/share/mlx/mlx.metallib`, matching the layout the
/// app-builder copies from the native build store.
const BUNDLED_METALLIB_RELATIVE_PATH: &str = "../Resources/share/mlx/mlx.metallib";

/// Chooses the metallib a worker should open: environment override, then the
/// app-bundle copy, then an unpackaged native-store file when one exists.
///
/// # Arguments
///
/// - `environment_override_path` — the value of the worker's metallib
///   environment override, when set. Absolute paths win outright so an
///   experiment can pin a specific kernel archive.
/// - `current_executable_path` — the running worker executable; anchors the
///   app-bundle lookup so the answer follows the binary that will load it.
/// - `unpackaged_native_store_metallib_path` — the metallib path the Cargo
///   build script baked in at compile time (see
///   `mlx_runtime::metallib::compiled_metallib_path`). Development fallback,
///   never the packaged default.
///
/// # Errors
///
/// Returns [`MlxRuntimeError::InvalidMetallibPath`] when no candidate exists
/// — the message tells app users to reinstall and development users to
/// rebuild native artifacts, because those are the two populations that hit
/// this path — or when an override is not absolute.
#[must_use]
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

    if let Some(unpackaged_native_store_metallib_path) = unpackaged_native_store_metallib_path
        && is_regular_non_symlink_file(unpackaged_native_store_metallib_path)
    {
        return require_absolute_metallib_path(unpackaged_native_store_metallib_path.to_path_buf());
    }

    Err(MlxRuntimeError::InvalidMetallibPath {
        description: format!(
            "MLX AOT metallib was not found next to the app executable at {bundled_metallib_path:?}. Reinstall Astronomical, or rebuild native artifacts for an unpackaged cargo run."
        ),
    })
}

/// Anchors the bundled metallib path at the executable's directory.
///
/// # Errors
///
/// Fails only when the executable path has no parent directory, which in
/// practice means a malformed `current_exe` result rather than a packaging
/// problem.
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

/// Rejects relative metallib paths before they can reach MLX.
///
/// MLX-C pins the metallib path process-globally as a C string; a relative
/// path would silently resolve against whatever the working directory
/// happens to be at first use, making kernel loading depend on where the
/// process was started. Failing here keeps that failure loud and local.
fn require_absolute_metallib_path(metallib_path: PathBuf) -> Result<PathBuf, MlxRuntimeError> {
    if !metallib_path.is_absolute() {
        return Err(MlxRuntimeError::InvalidMetallibPath {
            description: format!("path must be absolute: {metallib_path:?}"),
        });
    }
    Ok(metallib_path)
}

/// Symlink-rejecting existence check for resolution candidates.
///
/// The packaged-default probe must not follow a symlink: a link at the
/// bundled path could point anywhere in the filesystem, defeating the
/// "the app owns this exact file" guarantee the validation step relies on.
/// `symlink_metadata` inspects the link itself instead of its target, and a
/// missing candidate is an ordinary outcome here (fall through to the next
/// candidate), not an error.
fn is_regular_non_symlink_file(candidate_path: &Path) -> bool {
    candidate_path
        .symlink_metadata()
        .map(|path_metadata| path_metadata.is_file())
        .unwrap_or(false)
}

//! The MLX setting a Linux CUDA build applies once, before the bindings
//! context exists (issue #2699).
//!
//! # Why it is needed
//!
//! **Header folders.** MLX compiles some CUDA kernels at run time (its
//! quantized matrix multiplies among them) and looks for their CUTLASS,
//! CuTe and CCCL headers beside the binary MLX lives in. Astronomical links
//! MLX statically, so that binary is the worker, and nothing is there. The
//! native build publishes the headers into its store entry and hands their
//! folder, with the CUDA toolkit's own headers, to this crate as
//! compile-time values, which this module checks and passes to MLX.
//!
//! TF32, which MLX's CUDA backend would use for float32 matrix multiplies, is
//! forbidden on every backend by `float32_matrix_precision`.
//!
//! On macOS and on a Linux CPU build the compile-time values are absent and
//! nothing is changed.

use std::ffi::CString;
use std::path::Path;
use std::sync::OnceLock;

use crate::MlxRuntimeError;

/// Set by the native build for the Linux CUDA recipe only
/// (`native_linking::configure_linux_cuda_rust_linking`).
const CUDA_JIT_HEADERS_DIRECTORY: Option<&str> =
    option_env!("ASTRONOMICAL_MLX_CUDA_JIT_HEADERS_DIRECTORY");
const CUDA_TOOLKIT_INCLUDE_DIRECTORY: Option<&str> =
    option_env!("ASTRONOMICAL_MLX_CUDA_TOOLKIT_INCLUDE_DIRECTORY");

/// The two header folders MLX's run-time CUDA compiler is given, checked and
/// ready to cross into C.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CudaJitHeaderDirectories {
    pub(crate) bundled_headers_directory: CString,
    pub(crate) cuda_toolkit_include_directory: CString,
}

/// Names MLX's run-time CUDA compiler header folders once per process; later
/// calls repeat the first result. Does nothing on a build without the CUDA
/// backend.
///
/// # Errors
///
/// Fails when a header folder is missing or cannot cross into C, or MLX
/// rejects a setting.
pub(crate) fn configure_cuda_runtime_settings() -> Result<(), MlxRuntimeError> {
    static CONFIGURED: OnceLock<Result<(), String>> = OnceLock::new();
    match CONFIGURED.get_or_init(apply_cuda_runtime_settings) {
        Ok(()) => Ok(()),
        Err(description) => Err(MlxRuntimeError::CudaRuntimeSettings {
            description: description.clone(),
        }),
    }
}

fn apply_cuda_runtime_settings() -> Result<(), String> {
    let Some(jit_header_directories) =
        cuda_jit_header_directories(CUDA_JIT_HEADERS_DIRECTORY, CUDA_TOOLKIT_INCLUDE_DIRECTORY)?
    else {
        return Ok(());
    };
    mlx_c_fork::set_cuda_jit_header_directories(
        &jit_header_directories.bundled_headers_directory,
        &jit_header_directories.cuda_toolkit_include_directory,
    )
    .map_err(describe_mlx_error)
}

fn describe_mlx_error(mlx_error: mlx_bridge::MlxCError) -> String {
    mlx_error.to_string()
}

/// The header folders the native build recorded, checked: `None` when the
/// build has no CUDA backend (neither folder recorded).
///
/// Both folders are checked here because MLX's run-time compiler skips a
/// missing folder without a word, so a store entry removed after linking
/// would otherwise surface much later, as a kernel that cannot find
/// `cute/...` on the first quantized matrix multiply.
///
/// # Errors
///
/// Fails when only one folder is recorded, when a folder is not an existing
/// directory, or when a path holds a NUL byte.
pub(crate) fn cuda_jit_header_directories(
    recorded_jit_headers_directory: Option<&str>,
    recorded_cuda_toolkit_include_directory: Option<&str>,
) -> Result<Option<CudaJitHeaderDirectories>, String> {
    let (bundled_headers_directory, cuda_toolkit_include_directory) = match (
        recorded_jit_headers_directory,
        recorded_cuda_toolkit_include_directory,
    ) {
        (None, None) => return Ok(None),
        (Some(bundled_headers_directory), Some(cuda_toolkit_include_directory)) => {
            (bundled_headers_directory, cuda_toolkit_include_directory)
        }
        _ => {
            return Err(
                "the native build recorded only one of the two CUDA header folders".to_owned(),
            );
        }
    };
    Ok(Some(CudaJitHeaderDirectories {
        bundled_headers_directory: existing_directory_text(
            bundled_headers_directory,
            "MLX's run-time compiler headers",
        )?,
        cuda_toolkit_include_directory: existing_directory_text(
            cuda_toolkit_include_directory,
            "the CUDA toolkit headers",
        )?,
    }))
}

fn existing_directory_text(directory: &str, description: &str) -> Result<CString, String> {
    if !Path::new(directory).is_dir() {
        return Err(format!(
            "{description} folder {directory} is missing; rebuild so the native build \
             publishes it again"
        ));
    }
    match CString::new(directory) {
        Ok(directory_text) => Ok(directory_text),
        Err(_) => Err(format!(
            "{description} folder {directory:?} contains a NUL byte"
        )),
    }
}

#[cfg(test)]
#[path = "cuda_runtime_settings_tests.rs"]
mod tests;

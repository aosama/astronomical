//! The Astronomical fork surface of MLX-C: the patched native image declares
//! exactly three C symbols beyond the stock MLX-C release, and this crate owns
//! them. The public `mlx-bridge` crate binds the complete stock surface; this
//! extension layers the fork delta on top so the stock crate stays
//! publishable and vanilla.
//!
//! The coverage contract merges [`FORK_BRIDGED_FUNCTIONS`] with the stock
//! bridge inventory before comparing against the provisioned patched
//! headers, keeping the completeness proof exact across the fork boundary.

use std::ffi::{CStr, c_char};

use mlx_bridge::{MlxArray, MlxCError, error::check_status};

/// Every fork-added C function bridged here. The coverage contract appends
/// these to the stock bridge inventory so the patched header surface stays
/// fully accounted.
pub const FORK_BRIDGED_FUNCTIONS: [&str; 3] = [
    "mlx_array_detach",
    "mlx_cuda_set_jit_header_directories",
    "mlx_set_tf32_enabled",
];

// Added by the local patch `mlx-c-0.8.0-expose-array-detach.patch`: strips an
// array's lazy graph while keeping the realized buffer.
unsafe extern "C" {
    fn mlx_array_detach(arr: mlx_bridge::raw::mlx_array) -> i32;
}

/// Strips the lazy graph (primitive, inputs, siblings) while keeping the
/// realized buffer.
///
/// MLX retains every input of an array's graph for the array's lifetime, so
/// a retained `astro_expert_cache` table built from row-writes pins each source buffer — streamed
/// pages and retired tables — in active memory forever. Callers must
/// evaluate first: detaching before evaluation discards the unrealized graph
/// and leaves the buffer unwritten.
///
/// # Errors
/// Returns the captured MLX-C description when the detachment fails.
pub fn detach_array(array: &MlxArray) -> Result<(), MlxCError> {
    // SAFETY: `array` owns a live handle and MLX detachment does not retain
    // any Rust borrow beyond this call.
    let status = unsafe { mlx_array_detach(array.raw()) };
    check_status(status, "detach an MLX array")
}

// Added by the local patch `mlx-c-0.8.0-cuda-runtime-settings.patch`, over the
// MLX patch `mlx-0.32.5-cuda-jit-header-directories.patch`: the two run-time
// settings a statically linked MLX on CUDA needs (issue #2699).
unsafe extern "C" {
    fn mlx_cuda_set_jit_header_directories(
        bundled_headers_directory: *const c_char,
        cuda_toolkit_include_directory: *const c_char,
    ) -> i32;
    fn mlx_set_tf32_enabled(enabled: bool) -> i32;
}

/// Names the folders MLX's run-time CUDA compiler reads headers from: the
/// CUTLASS, CuTe and CCCL headers MLX installs, and the CUDA runtime headers.
///
/// A statically linked MLX looks for them beside the executable instead, and
/// fails to compile kernels such as `qmm_naive` without them. Call before the
/// first evaluation; later calls have no effect. A no-op without the CUDA
/// backend.
///
/// # Errors
/// Returns the captured MLX-C description when MLX rejects the call.
pub fn set_cuda_jit_header_directories(
    bundled_headers_directory: &CStr,
    cuda_toolkit_include_directory: &CStr,
) -> Result<(), MlxCError> {
    // SAFETY: both pointers come from live `CStr`s that MLX copies before
    // returning; it keeps no borrow.
    let status = unsafe {
        mlx_cuda_set_jit_header_directories(
            bundled_headers_directory.as_ptr(),
            cuda_toolkit_include_directory.as_ptr(),
        )
    };
    check_status(status, "name MLX's CUDA run-time compiler header folders")
}

/// Allows or forbids TF32 for float32 matrix multiplies, which MLX allows
/// by default.
///
/// The setting is MLX's process-wide `MLX_ENABLE_TF32`, read by every
/// backend: CUDA runs cuBLAS float32 products on TF32 tensor cores with it,
/// and Metal's NAX kernels read it too. TF32 keeps a 10-bit mantissa:
/// measured on Laya's projections on CUDA it made the largest difference
/// from a float32 product 200 to 1,000 times larger (issue #2699).
///
/// # Errors
/// Returns the captured MLX-C description when MLX rejects the call.
pub fn set_tf32_enabled(enabled: bool) -> Result<(), MlxCError> {
    // SAFETY: a plain value crosses the boundary.
    let status = unsafe { mlx_set_tf32_enabled(enabled) };
    check_status(status, "set MLX's TF32 setting")
}

//! The Astronomical fork surface of MLX-C: the patched native image declares
//! exactly one C symbol beyond the stock MLX-C release, and this crate owns
//! it. The public `mlx-c-rust` crate binds the complete stock surface; this
//! extension layers the fork delta on top so the stock crate stays
//! publishable and vanilla.
//!
//! The coverage contract merges [`FORK_BRIDGED_FUNCTIONS`] with the stock
//! bridge inventory before comparing against the provisioned patched
//! headers, keeping the completeness proof exact across the fork boundary.

use mlx_c_rust::{MlxArray, MlxCError, error::check_status};

/// Every fork-added C function bridged here. The coverage contract appends
/// these to the stock bridge inventory so the patched header surface stays
/// fully accounted.
pub const FORK_BRIDGED_FUNCTIONS: [&str; 1] = ["mlx_array_detach"];

// Added by the local patch `mlx-c-0.7.0-expose-array-detach.patch`: strips an
// array's lazy graph while keeping the realized buffer.
unsafe extern "C" {
    fn mlx_array_detach(arr: mlx_c_rust::raw::mlx_array) -> i32;
}

/// Strips the lazy graph (primitive, inputs, siblings) while keeping the
/// realized buffer.
///
/// MLX retains every input of an array's graph for the array's lifetime, so
/// a retained cache built from row-writes pins each source buffer — streamed
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

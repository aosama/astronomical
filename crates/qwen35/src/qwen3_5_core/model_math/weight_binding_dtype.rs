//! The dtype an engine binds Qwen3.5's tensors in.
//!
//! MLX's CPU backend runs Qwen3.5 in BF16 inaccurately enough to change its
//! answers, and slower than in float32 (#2441), so an engine that holds its
//! tensors as bound can bind half-precision tensors in float32 there. The
//! choice belongs to the engine because it decides how memory is planned.

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;

use super::error::Qwen3_5ExecutionError;

/// Which dtype an engine binds its tensors in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightBindingDtype {
    /// Exactly as stored. The streaming engine binds this way: its memory
    /// plans are sized from the bytes on disk.
    AsStored,
    /// The dtype the compute device runs the model accurately in: float32
    /// for half-precision tensors where MLX computes on its CPU, as stored on
    /// the Metal build.
    ForComputeDevice,
}

/// Pure: whether a tensor of `stored_dtype` binds widened to float32.
///
/// MLX's CPU backend runs Qwen3.5 in BF16 inaccurately enough to change its
/// answers, and slower than in float32 (#2441). Widening every half-precision
/// tensor once at bind makes the whole graph compute in float32: activations
/// promote from the norms and affine scales they meet, and the decoder-state
/// dtype replay reads the bound dtypes, so the `astro_decoder_state_cache`
/// and memory admission size float32 state. Packed quantized weights and
/// float32 tensors keep their storage.
#[must_use]
pub fn binds_widened_to_float32(
    stored_dtype: mlx_bridge::MlxDtype,
    weight_binding_dtype: WeightBindingDtype,
    mlx_computes_on_cpu: bool,
) -> bool {
    let is_half_precision = matches!(
        stored_dtype,
        mlx_bridge::MlxDtype::BFloat16 | mlx_bridge::MlxDtype::Float16
    );
    is_half_precision
        && weight_binding_dtype == WeightBindingDtype::ForComputeDevice
        && mlx_computes_on_cpu
}

/// The dtype a tensor stored as `stored_dtype` binds in under
/// `weight_binding_dtype`, for values an engine builds itself beside its bound
/// tensors (for example constant scales).
///
/// # Errors
///
/// Returns the MLX description when the backend query fails.
pub fn bound_dtype_of(
    stored_dtype: mlx_bridge::MlxDtype,
    weight_binding_dtype: WeightBindingDtype,
) -> Result<mlx_bridge::MlxDtype, Qwen3_5ExecutionError> {
    let mlx_computes_on_cpu = runtime_integration::mlx_computes_on_cpu()?;
    if binds_widened_to_float32(stored_dtype, weight_binding_dtype, mlx_computes_on_cpu) {
        Ok(mlx_bridge::MlxDtype::Float32)
    } else {
        Ok(stored_dtype)
    }
}

pub(super) fn bound_in_binding_dtype(
    runtime: &MlxRuntime,
    tensor: MlxArray,
    weight_binding_dtype: WeightBindingDtype,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    if weight_binding_dtype == WeightBindingDtype::AsStored {
        return Ok(tensor);
    }
    let mlx_computes_on_cpu = runtime_integration::mlx_computes_on_cpu()?;
    if !binds_widened_to_float32(tensor.dtype(), weight_binding_dtype, mlx_computes_on_cpu) {
        return Ok(tensor);
    }
    Ok(runtime.astype(&tensor, mlx_bridge::MlxDtype::Float32)?)
}

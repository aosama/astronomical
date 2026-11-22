//! Affine quantized matmul on the device MLX computes on.
//!
//! # Why the CPU takes a float32 route
//!
//! MLX's CPU backend runs `quantized_matmul` on BF16 and float16 activations
//! far slower than on float32 ones: on an x86 CPU without native half
//! arithmetic, one 64-token, 1024-to-3584 four-bit projection measured
//! 3,812 ms in BF16 and 383 ms through float32, and one reply token's
//! vocabulary head 3,645 ms against 412 ms. A 0.8B model's prompt chunk spent
//! six and a half minutes there. Both kernels accumulate in float32, so
//! computing in float32 and rounding the result back to the activations' own
//! dtype gives the same or slightly better values at a tenth of the time.
//! Metal has no such gap, so macOS calls MLX directly, exactly as before.

use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype};

use super::MlxRuntime;

impl MlxRuntime {
    /// `activations × dequantize(weights)` through MLX's affine quantized
    /// matmul, on the route that is fast for the device this build of MLX
    /// computes on. The result has the activations' dtype on every route.
    ///
    /// # Errors
    ///
    /// Returns the MLX description when the arguments are rejected or the
    /// operation fails.
    // The arguments are MLX's own quantized matmul parameters, passed through.
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_matmul_affine_on_platform(
        &self,
        activations: &MlxArray,
        quantized_weights: &MlxArray,
        scales: &MlxArray,
        biases: &MlxArray,
        transpose_weights: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<MlxArray, MlxCError> {
        if mlx_computes_half_precision_quantized_matmul_slowly()? {
            return quantized_matmul_affine_through_float32(
                self,
                activations,
                quantized_weights,
                scales,
                biases,
                transpose_weights,
                group_size,
                bits,
            );
        }
        self.quantized_matmul_affine(
            activations,
            quantized_weights,
            scales,
            biases,
            transpose_weights,
            group_size,
            bits,
        )
    }
}

/// The Metal build computes on the GPU, where half precision is fast.
#[cfg(target_os = "macos")]
#[allow(clippy::unnecessary_wraps)]
const fn mlx_computes_half_precision_quantized_matmul_slowly() -> Result<bool, MlxCError> {
    Ok(false)
}

/// Elsewhere MLX computes on its CPU backend unless it has a GPU backend.
#[cfg(not(target_os = "macos"))]
fn mlx_computes_half_precision_quantized_matmul_slowly() -> Result<bool, MlxCError> {
    static COMPUTES_ON_CPU: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if let Some(computes_on_cpu) = COMPUTES_ON_CPU.get() {
        return Ok(*computes_on_cpu);
    }
    let has_gpu_backend = mlx_bridge::metal_is_available()? || mlx_bridge::cuda_is_available()?;
    Ok(*COMPUTES_ON_CPU.get_or_init(|| !has_gpu_backend))
}

/// Computes the quantized matmul with float32 activations, scales and biases,
/// then rounds the result to the activations' own dtype. Float32 activations
/// take MLX's ordinary route, since there is nothing to widen.
///
/// # Errors
///
/// Returns the MLX description when a cast or the matmul fails.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_affine_through_float32(
    context: &MlxBindingsContext,
    activations: &MlxArray,
    quantized_weights: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    transpose_weights: bool,
    group_size: i32,
    bits: i32,
) -> Result<MlxArray, MlxCError> {
    let activation_dtype = activations.dtype();
    if activation_dtype == MlxDtype::Float32 {
        return context.quantized_matmul_affine(
            activations,
            quantized_weights,
            scales,
            biases,
            transpose_weights,
            group_size,
            bits,
        );
    }
    let float32_projection = context.quantized_matmul_affine(
        &context.astype(activations, MlxDtype::Float32)?,
        quantized_weights,
        &context.astype(scales, MlxDtype::Float32)?,
        &context.astype(biases, MlxDtype::Float32)?,
        transpose_weights,
        group_size,
        bits,
    )?;
    context.astype(&float32_projection, activation_dtype)
}

#[cfg(test)]
#[path = "quantized_projection_tests.rs"]
mod tests;

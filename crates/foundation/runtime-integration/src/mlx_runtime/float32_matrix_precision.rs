//! Float32 matrix multiplies stay float32 on every backend (issue #2699).
//!
//! MLX allows TF32 for float32 matrix multiplies by default
//! (`MLX_ENABLE_TF32`). TF32 keeps a 10-bit mantissa where float32 keeps 23.
//! Two backends act on it: CUDA runs cuBLAS float32 products on TF32 tensor
//! cores, and Metal on Apple GPUs with neural accelerators (M5 and later)
//! picks its NAX kernels for float32 matmuls, quantized matmuls and
//! attention.
//!
//! Astronomical never lowers a model's precision, and the float32 training
//! and reference journeys assume float32 arithmetic. Measured on an M5 Pro
//! with Laya's whole-model gradient check (`test-laya-whole-model-gradient`):
//! with TF32 allowed the loss's central differences were drowned by
//! rounding, a median gap of 0.60 between predicted and measured change; with
//! TF32 forbidden the median gap was 0.0044 and every array passed. So TF32
//! is forbidden once per process, before the bindings context exists. Models
//! in bfloat16 or quantized weights are unaffected: MLX's NAX choice only
//! consults the setting for float32 inputs.

use std::sync::OnceLock;

use crate::MlxRuntimeError;

/// Forbids TF32 for float32 matrix multiplies, once per process; later
/// calls repeat the first result.
///
/// # Errors
///
/// Returns MLX's description when it rejects the setting.
pub(crate) fn keep_float32_matrix_multiplies_float32() -> Result<(), MlxRuntimeError> {
    static CONFIGURED: OnceLock<Result<(), String>> = OnceLock::new();
    match CONFIGURED.get_or_init(forbid_tf32) {
        Ok(()) => Ok(()),
        Err(description) => Err(MlxRuntimeError::RuntimeOperation {
            operation: "forbid TF32 for float32 matrix multiplies",
            description: description.clone(),
        }),
    }
}

fn forbid_tf32() -> Result<(), String> {
    match mlx_c_fork::set_tf32_enabled(false) {
        Ok(()) => Ok(()),
        Err(mlx_error) => Err(mlx_error.to_string()),
    }
}

#[cfg(test)]
#[path = "float32_matrix_precision_tests.rs"]
mod tests;

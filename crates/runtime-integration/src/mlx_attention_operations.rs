//! Validated wrappers over MLX's fused scaled dot-product attention.
//!
//! C ABI: `mlx-c/mlx/c/fast.h::mlx_fast_scaled_dot_product_attention`.
//! C++ bridge: `mlx-c/mlx/c/fast.cpp`, forwarding to
//! `mlx::core::fast::scaled_dot_product_attention`. MLX computes attention as
//! `softmax(scale * Q @ K^T) @ V` and performs softmax in Float32 even for BF16
//! inputs. Model code should shape/pad/segment tensors, not reproduce this math.

use crate::{MlxArray, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Applies MLX-C fused unmasked attention over `[batch, heads, length, width]`.
    ///
    /// Q heads may be a multiple of K/V heads for grouped-query attention. Qwen3.5-MoE
    /// vision uses equal counts, while the text model exercises grouped heads.
    pub fn scaled_dot_product_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_attention_arguments(queries, keys, values, scale)?;
        self.scaled_dot_product_attention_with_mode(
            queries,
            keys,
            values,
            scale,
            c"".as_ptr(),
            "apply unmasked scaled dot-product attention",
        )
    }

    /// Applies causal scaled dot-product attention without an explicit mask array.
    pub fn causal_scaled_dot_product_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_attention_arguments(queries, keys, values, scale)?;
        self.scaled_dot_product_attention_with_mode(
            queries,
            keys,
            values,
            scale,
            c"causal".as_ptr(),
            "apply causal scaled dot-product attention",
        )
    }

    fn scaled_dot_product_attention_with_mode(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
        mask_mode: *const std::ffi::c_char,
        operation: &'static str,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array(operation, |output, stream| {
            // SAFETY: Input arrays and stream are live, optional mask/sinks are
            // represented by MLX's empty handle convention, the mode is a static
            // C string, and output is uniquely writable.
            unsafe {
                raw::mlx_fast_scaled_dot_product_attention(
                    output,
                    queries.raw(),
                    keys.raw(),
                    values.raw(),
                    scale,
                    mask_mode,
                    MlxArray::empty_raw(),
                    MlxArray::empty_raw(),
                    stream,
                )
            }
        })
    }
}

fn validate_attention_arguments(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "apply unmasked scaled dot-product attention";
    if !scale.is_finite() {
        return Err(runtime_operation_error(
            OPERATION,
            "attention scale must be finite",
        ));
    }
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    let value_shape = values.shape();
    if query_shape.len() != 4 || key_shape.len() != 4 || value_shape.len() != 4 {
        return Err(runtime_operation_error(
            OPERATION,
            "attention query, key, and value tensors must have rank four",
        ));
    }
    if key_shape != value_shape {
        return Err(runtime_operation_error(
            OPERATION,
            "attention keys and values must have identical shape for the unmasked wrapper",
        ));
    }
    if query_shape[0] != key_shape[0] {
        return Err(runtime_operation_error(
            OPERATION,
            "attention batch dimensions must match",
        ));
    }
    if query_shape[3] != key_shape[3] {
        return Err(runtime_operation_error(
            OPERATION,
            "attention query and key head widths must match",
        ));
    }
    if query_shape[1] % key_shape[1] != 0 {
        return Err(runtime_operation_error(
            OPERATION,
            "attention query heads must be a multiple of key/value heads",
        ));
    }
    Ok(())
}

fn runtime_operation_error(operation: &'static str, description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.to_owned(),
    }
}

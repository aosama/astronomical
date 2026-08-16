//! Validated wrappers over MLX's fused scaled dot-product attention.
//!
//! C ABI: `mlx-c/mlx/c/fast.h::mlx_fast_scaled_dot_product_attention`.
//! C++ bridge: `mlx-c/mlx/c/fast.cpp`, forwarding to
//! `mlx::core::fast::scaled_dot_product_attention`. MLX computes attention as
//! `softmax(scale * Q @ K^T + mask) @ V` and performs softmax in Float32 even for
//! BF16 inputs. Model code should shape/pad/segment tensors, not reproduce this
//! math. Callers can select unmasked, causal, or explicit array-mask execution.

use crate::{MlxArray, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Applies MLX-C fused unmasked attention over `[batch, heads, length, width]`.
    ///
    /// Q heads may be a multiple of K/V heads for grouped-query attention.
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
            MlxArray::empty_raw(),
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
            MlxArray::empty_raw(),
            "apply causal scaled dot-product attention",
        )
    }

    /// Applies fused attention with a broadcast-compatible boolean or additive mask.
    pub fn masked_scaled_dot_product_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
        mask: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_masked_attention_arguments(queries, keys, values, scale, mask)?;
        self.scaled_dot_product_attention_with_mode(
            queries,
            keys,
            values,
            scale,
            c"array".as_ptr(),
            mask.raw(),
            "apply masked scaled dot-product attention",
        )
    }

    fn scaled_dot_product_attention_with_mode(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
        mask_mode: *const std::ffi::c_char,
        mask_array: raw::mlx_array,
        operation: &'static str,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array(operation, |output, stream| {
            // SAFETY: Inputs and stream are live, mode is a static C string,
            // mask/sinks are valid MLX handles, and output is uniquely writable.
            unsafe {
                raw::mlx_fast_scaled_dot_product_attention(
                    output,
                    queries.raw(),
                    keys.raw(),
                    values.raw(),
                    scale,
                    mask_mode,
                    mask_array,
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
    const OPERATION: &str = "apply scaled dot-product attention";
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
            "attention keys and values must have identical shape",
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

/// Validates the explicit mask against the query and key sequence dimensions.
fn validate_masked_attention_arguments(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    mask: &MlxArray,
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "apply masked scaled dot-product attention";
    validate_attention_arguments(queries, keys, values, scale)?;
    let mask_shape = mask.shape();
    if mask_shape.len() < 2 || mask_shape.len() > 4 {
        return Err(runtime_operation_error(
            OPERATION,
            "attention mask must have rank 2, 3, or 4",
        ));
    }
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    let mask_rank = mask_shape.len();
    let mask_query_length = mask_shape[mask_rank - 2];
    let mask_key_length = mask_shape[mask_rank - 1];
    if mask_query_length == 0 || mask_key_length == 0 {
        return Err(runtime_operation_error(
            OPERATION,
            "attention mask sequence dimensions must be positive",
        ));
    }
    if mask_query_length != 1 && mask_query_length != query_shape[2] {
        return Err(runtime_operation_error(
            OPERATION,
            "attention mask query dimension must broadcast to the query length",
        ));
    }
    if mask_key_length != 1 && mask_key_length != key_shape[2] {
        return Err(runtime_operation_error(
            OPERATION,
            "attention mask key dimension must broadcast to the key/value length",
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

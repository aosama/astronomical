//! Construction of deterministic affine-quantized MLX weights.

use crate::{
    MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, mlx_array_vector::MlxArrayVector,
    mlx_runtime::check_status, raw,
};

impl MlxRuntime {
    /// Quantizes floating-point weights into MLX affine packed rows.
    pub fn quantize_affine(
        &self,
        weights: &MlxArray,
        group_size: i32,
        bits: i32,
    ) -> Result<(MlxArray, MlxArray, MlxArray), MlxRuntimeError> {
        const OPERATION: &str = "quantize MLX weights with affine parameters";
        validate_quantization_request(weights, group_size, bits, OPERATION)?;
        let optional_group_size = raw::mlx_optional_int {
            value: group_size,
            has_value: true,
        };
        let optional_bits = raw::mlx_optional_int {
            value: bits,
            has_value: true,
        };
        let mut output_vector = MlxArrayVector::empty(OPERATION)?;
        // SAFETY: Weights and stream are live, scalar options are copied, the
        // mode string is static, and the empty output vector is uniquely writable.
        let status = unsafe {
            raw::mlx_quantize(
                output_vector.raw_mut(),
                weights.raw(),
                optional_group_size,
                optional_bits,
                c"affine".as_ptr(),
                MlxArray::empty_raw(),
                self.gpu_stream().raw(),
            )
        };
        check_status(status, OPERATION)?;
        if output_vector.len() != 3 {
            return Err(operation_error(
                OPERATION,
                "affine quantization must return packed weights, scales, and biases",
            ));
        }
        Ok((
            output_vector.array_at(0, "read affine packed weights")?,
            output_vector.array_at(1, "read affine quantization scales")?,
            output_vector.array_at(2, "read affine quantization biases")?,
        ))
    }
}

fn validate_quantization_request(
    weights: &MlxArray,
    group_size: i32,
    bits: i32,
    operation: &'static str,
) -> Result<(), MlxRuntimeError> {
    if !matches!(
        weights.dtype(),
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::BFloat16
    ) {
        return Err(operation_error(
            operation,
            "weights must be float16, bfloat16, or float32",
        ));
    }
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(operation_error(
            operation,
            "affine group size must be 32, 64, or 128",
        ));
    }
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return Err(operation_error(
            operation,
            "affine bit width must be 2, 3, 4, 5, 6, or 8",
        ));
    }
    let input_dimension = weights
        .shape()
        .last()
        .copied()
        .filter(|dimension| *dimension > 0)
        .ok_or_else(|| operation_error(operation, "weights must have a positive tail dimension"))?;
    if input_dimension % group_size != 0 {
        return Err(operation_error(
            operation,
            "weight input dimension must be divisible by the affine group size",
        ));
    }
    Ok(())
}

fn operation_error(operation: &'static str, description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.to_owned(),
    }
}

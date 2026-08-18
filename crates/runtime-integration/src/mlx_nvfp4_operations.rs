//! NVFP4 construction and execution through the official MLX C API.

use crate::{
    MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, mlx_array_vector::MlxArrayVector,
    mlx_runtime::check_status, raw,
};

const NVFP4_GROUP_SIZE: i32 = 16;
const NVFP4_BITS: i32 = 4;

impl MlxRuntime {
    /// Quantizes floating-point weights into MLX NVFP4 packed weights and E4M3 scales.
    pub fn quantize_nvfp4(
        &self,
        weights: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
        const OPERATION: &str = "quantize MLX weights as NVFP4";
        validate_nvfp4_source_weights(weights, OPERATION)?;
        let mut output_vector = MlxArrayVector::empty(OPERATION)?;
        // SAFETY: The weight and stream handles are live, scalar options are copied,
        // the mode is static, and the output vector is uniquely writable.
        let status = unsafe {
            raw::mlx_quantize(
                output_vector.raw_mut(),
                weights.raw(),
                required_int(NVFP4_GROUP_SIZE),
                required_int(NVFP4_BITS),
                c"nvfp4".as_ptr(),
                MlxArray::empty_raw(),
                self.gpu_stream().raw(),
            )
        };
        check_status(status, OPERATION)?;
        if output_vector.len() != 2 {
            return Err(operation_error(
                OPERATION,
                "NVFP4 quantization must return packed weights and scales",
            ));
        }
        Ok((
            output_vector.array_at(0, "read NVFP4 packed weights")?,
            output_vector.array_at(1, "read NVFP4 scales")?,
        ))
    }

    /// Dequantizes MLX NVFP4 rows without introducing an affine bias.
    pub fn dequantize_nvfp4(
        &self,
        quantized_weights: &MlxArray,
        scales: &MlxArray,
        output_dtype: MlxDtype,
    ) -> Result<MlxArray, MlxRuntimeError> {
        const OPERATION: &str = "dequantize NVFP4 weights";
        validate_nvfp4_storage(quantized_weights, scales, OPERATION)?;
        if !is_supported_nvfp4_float_dtype(output_dtype) {
            return Err(operation_error(
                OPERATION,
                "NVFP4 output dtype must be float16, bfloat16, or float32",
            ));
        }
        let requested_output_dtype = raw::mlx_optional_dtype {
            value: output_dtype.to_raw(),
            has_value: true,
        };
        self.output_array(OPERATION, |output, stream| {
            // SAFETY: Array and stream handles are live, absent bias/global scale
            // use official empty handles, options are copied, and output is unique.
            unsafe {
                raw::mlx_dequantize(
                    output,
                    quantized_weights.raw(),
                    scales.raw(),
                    MlxArray::empty_raw(),
                    required_int(NVFP4_GROUP_SIZE),
                    required_int(NVFP4_BITS),
                    c"nvfp4".as_ptr(),
                    MlxArray::empty_raw(),
                    requested_output_dtype,
                    stream,
                )
            }
        })
    }

    /// Builds an NVFP4 matrix multiplication using MLX's native packed representation.
    pub fn quantized_matmul_nvfp4(
        &self,
        activations: &MlxArray,
        quantized_weights: &MlxArray,
        scales: &MlxArray,
        transpose_weights: bool,
    ) -> Result<MlxArray, MlxRuntimeError> {
        const OPERATION: &str = "build NVFP4 quantized matmul";
        validate_nvfp4_quantized_matmul(
            activations,
            quantized_weights,
            scales,
            transpose_weights,
            OPERATION,
        )?;
        self.output_array(OPERATION, |output, stream| {
            // SAFETY: Array and stream handles are live, NVFP4 has no affine bias,
            // scalar options are copied, the mode is static, and output is unique.
            unsafe {
                raw::mlx_quantized_matmul(
                    output,
                    activations.raw(),
                    quantized_weights.raw(),
                    scales.raw(),
                    MlxArray::empty_raw(),
                    transpose_weights,
                    required_int(NVFP4_GROUP_SIZE),
                    required_int(NVFP4_BITS),
                    c"nvfp4".as_ptr(),
                    stream,
                )
            }
        })
    }
}

fn validate_nvfp4_source_weights(
    weights: &MlxArray,
    operation: &'static str,
) -> Result<(), MlxRuntimeError> {
    if !is_supported_nvfp4_float_dtype(weights.dtype()) {
        return Err(operation_error(
            operation,
            "NVFP4 source weights must be float16, bfloat16, or float32",
        ));
    }
    let input_dimension = positive_last_dimension(&weights.shape(), operation)?;
    if input_dimension % NVFP4_GROUP_SIZE != 0 {
        return Err(operation_error(
            operation,
            "NVFP4 source width must be divisible by 16",
        ));
    }
    Ok(())
}

fn validate_nvfp4_storage(
    quantized_weights: &MlxArray,
    scales: &MlxArray,
    operation: &'static str,
) -> Result<i32, MlxRuntimeError> {
    if quantized_weights.dtype() != MlxDtype::UInt32 {
        return Err(operation_error(
            operation,
            "NVFP4 packed weights must have uint32 dtype",
        ));
    }
    if scales.dtype() != MlxDtype::UInt8 {
        return Err(operation_error(
            operation,
            "NVFP4 scales must have uint8 E4M3 storage",
        ));
    }
    let quantized_shape = quantized_weights.shape();
    let packed_tail_dimension = positive_last_dimension(&quantized_shape, operation)?;
    let expanded_tail_dimension = packed_tail_dimension.checked_mul(8).ok_or_else(|| {
        operation_error(
            operation,
            "NVFP4 packed weight width exceeds the supported range",
        )
    })?;
    if expanded_tail_dimension % NVFP4_GROUP_SIZE != 0 {
        return Err(operation_error(
            operation,
            "NVFP4 packed width must expand to complete 16-value groups",
        ));
    }
    let mut expected_scale_shape = quantized_shape;
    let scale_tail_position = expected_scale_shape.len() - 1;
    expected_scale_shape[scale_tail_position] = expanded_tail_dimension / NVFP4_GROUP_SIZE;
    if scales.shape() != expected_scale_shape {
        return Err(operation_error(
            operation,
            "NVFP4 scale shape does not match the packed weight shape",
        ));
    }
    Ok(expanded_tail_dimension)
}

fn validate_nvfp4_quantized_matmul(
    activations: &MlxArray,
    quantized_weights: &MlxArray,
    scales: &MlxArray,
    transpose_weights: bool,
    operation: &'static str,
) -> Result<(), MlxRuntimeError> {
    if !is_supported_nvfp4_float_dtype(activations.dtype()) {
        return Err(operation_error(
            operation,
            "NVFP4 activations must be float16, bfloat16, or float32",
        ));
    }
    let activation_shape = activations.shape();
    let weight_shape = quantized_weights.shape();
    if activation_shape.len() < 2 || weight_shape.len() < 2 {
        return Err(operation_error(
            operation,
            "NVFP4 activations and weights must have rank at least two",
        ));
    }
    let expanded_weight_tail = validate_nvfp4_storage(quantized_weights, scales, operation)?;
    let activation_inner_dimension = positive_last_dimension(&activation_shape, operation)?;
    let required_activation_dimension = if transpose_weights {
        expanded_weight_tail
    } else {
        weight_shape[weight_shape.len() - 2]
    };
    if activation_inner_dimension != required_activation_dimension {
        return Err(operation_error(
            operation,
            "activation inner dimension does not match the NVFP4 weight input",
        ));
    }
    Ok(())
}

const fn required_int(value: i32) -> raw::mlx_optional_int {
    raw::mlx_optional_int {
        value,
        has_value: true,
    }
}

const fn is_supported_nvfp4_float_dtype(dtype: MlxDtype) -> bool {
    matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::BFloat16
    )
}

fn positive_last_dimension(shape: &[i32], operation: &'static str) -> Result<i32, MlxRuntimeError> {
    shape
        .last()
        .copied()
        .filter(|dimension| *dimension > 0)
        .ok_or_else(|| operation_error(operation, "array must have a positive tail dimension"))
}

fn operation_error(operation: &'static str, description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.to_owned(),
    }
}

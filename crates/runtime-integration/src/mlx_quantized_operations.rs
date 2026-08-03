use crate::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Dequantizes selected affine rows, as required by quantized embedding lookup.
    pub fn dequantize_affine(
        &self,
        quantized_weights: &MlxArray,
        scales: &MlxArray,
        biases: &MlxArray,
        group_size: i32,
        bits: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_affine_dequantize_arguments(quantized_weights, scales, biases, group_size, bits)?;
        let optional_group_size = raw::mlx_optional_int {
            value: group_size,
            has_value: true,
        };
        let optional_bits = raw::mlx_optional_int {
            value: bits,
            has_value: true,
        };
        let absent_output_dtype = raw::mlx_optional_dtype {
            value: raw::mlx_dtype__MLX_FLOAT32,
            has_value: false,
        };
        self.output_array("dequantize selected affine rows", |output, stream| {
            // SAFETY: All array handles and the stream are live, scalar options
            // are copied by value, absent global scale/output dtype use official
            // empty conventions, and output is uniquely writable.
            unsafe {
                raw::mlx_dequantize(
                    output,
                    quantized_weights.raw(),
                    scales.raw(),
                    biases.raw(),
                    optional_group_size,
                    optional_bits,
                    c"affine".as_ptr(),
                    MlxArray::empty_raw(),
                    absent_output_dtype,
                    stream,
                )
            }
        })
    }

    /// Builds an affine quantized matrix multiplication using parameters supported by MLX.
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_matmul_affine(
        &self,
        activations: &MlxArray,
        quantized_weights: &MlxArray,
        scales: &MlxArray,
        biases: &MlxArray,
        transpose_weights: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_affine_quantized_matmul_arguments(
            "build affine quantized matmul",
            activations,
            quantized_weights,
            scales,
            biases,
            transpose_weights,
            group_size,
            bits,
        )?;
        let optional_group_size = raw::mlx_optional_int {
            value: group_size,
            has_value: true,
        };
        let optional_bits = raw::mlx_optional_int {
            value: bits,
            has_value: true,
        };
        self.output_array("build affine quantized matmul", |output, stream| {
            // SAFETY: All array handles and the stream are live, scalar options
            // are copied by value, the mode points to a static NUL-terminated
            // string, and output is uniquely writable for MLX to populate.
            unsafe {
                raw::mlx_quantized_matmul(
                    output,
                    activations.raw(),
                    quantized_weights.raw(),
                    scales.raw(),
                    biases.raw(),
                    transpose_weights,
                    optional_group_size,
                    optional_bits,
                    c"affine".as_ptr(),
                    stream,
                )
            }
        })
    }

    /// Builds selected affine quantized matrix multiplications for MoE experts.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_quantized_matmul_affine(
        &self,
        activations: &MlxArray,
        quantized_weights: &MlxArray,
        scales: &MlxArray,
        biases: &MlxArray,
        lhs_indices: Option<&MlxArray>,
        rhs_indices: Option<&MlxArray>,
        transpose_weights: bool,
        group_size: i32,
        bits: i32,
        sorted_indices: bool,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_affine_quantized_matmul_arguments(
            "build affine gather_qmm",
            activations,
            quantized_weights,
            scales,
            biases,
            transpose_weights,
            group_size,
            bits,
        )?;
        validate_gather_quantized_matmul_indices(lhs_indices, rhs_indices)?;
        let optional_group_size = raw::mlx_optional_int {
            value: group_size,
            has_value: true,
        };
        let optional_bits = raw::mlx_optional_int {
            value: bits,
            has_value: true,
        };
        let raw_lhs_indices = lhs_indices.map_or_else(MlxArray::empty_raw, MlxArray::raw);
        let raw_rhs_indices = rhs_indices.map_or_else(MlxArray::empty_raw, MlxArray::raw);
        self.output_array("build affine gather_qmm", |output, stream| {
            // SAFETY: All array handles and the stream are live, absent optional
            // index arrays are represented by MLX's empty handle convention,
            // scalar options are copied by value, the mode is a static C string,
            // and output is uniquely writable for MLX to populate.
            unsafe {
                raw::mlx_gather_qmm(
                    output,
                    activations.raw(),
                    quantized_weights.raw(),
                    scales.raw(),
                    biases.raw(),
                    raw_lhs_indices,
                    raw_rhs_indices,
                    transpose_weights,
                    optional_group_size,
                    optional_bits,
                    c"affine".as_ptr(),
                    sorted_indices,
                    stream,
                )
            }
        })
    }
}

fn validate_affine_dequantize_arguments(
    quantized_weights: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    group_size: i32,
    bits: i32,
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "dequantize selected affine rows";
    if !is_mlx_supported_affine_group_size(group_size) {
        return Err(runtime_operation_error(
            OPERATION,
            "affine group size must be 32, 64, or 128",
        ));
    }
    if !is_mlx_supported_affine_bit_width(bits) {
        return Err(runtime_operation_error(
            OPERATION,
            "affine bit width must be 2, 3, 4, 5, 6, or 8",
        ));
    }
    if quantized_weights.dtype() != MlxDtype::UInt32 {
        return Err(runtime_operation_error(
            OPERATION,
            "quantized weights must have uint32 dtype",
        ));
    }
    if !is_supported_quantized_float_dtype(scales.dtype())
        || !is_supported_quantized_float_dtype(biases.dtype())
    {
        return Err(runtime_operation_error(
            OPERATION,
            "affine scales and biases must use a supported floating-point dtype",
        ));
    }
    let quantized_shape = quantized_weights.shape();
    let scale_shape = scales.shape();
    if quantized_shape.is_empty() || scale_shape != biases.shape() {
        return Err(runtime_operation_error(
            OPERATION,
            "quantized weights must have rank and scale/bias shapes must match",
        ));
    }
    let packed_tail_dimension = positive_last_dimension(&quantized_shape, OPERATION)?;
    let expanded_tail_dimension = packed_tail_dimension
        .checked_mul(32)
        .and_then(|packed_bits| (packed_bits % bits == 0).then_some(packed_bits / bits))
        .ok_or_else(|| {
            runtime_operation_error(
                OPERATION,
                "packed weight shape overflows or is incompatible with the requested bit width",
            )
        })?;
    let mut expected_scale_shape = quantized_shape;
    let scale_tail_position = expected_scale_shape.len() - 1;
    expected_scale_shape[scale_tail_position] =
        exact_group_count(expanded_tail_dimension, group_size, OPERATION)?;
    if scale_shape != expected_scale_shape {
        return Err(runtime_operation_error(
            OPERATION,
            "affine scale and bias shapes do not match the packed weight shape",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_affine_quantized_matmul_arguments(
    operation: &'static str,
    activations: &MlxArray,
    quantized_weights: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    transpose_weights: bool,
    group_size: i32,
    bits: i32,
) -> Result<(), MlxRuntimeError> {
    if !is_mlx_supported_affine_group_size(group_size) {
        return Err(runtime_operation_error(
            operation,
            "affine group size must be 32, 64, or 128",
        ));
    }
    if !is_mlx_supported_affine_bit_width(bits) {
        return Err(runtime_operation_error(
            operation,
            "affine bit width must be 2, 3, 4, 5, 6, or 8",
        ));
    }
    if !is_supported_quantized_float_dtype(activations.dtype()) {
        return Err(runtime_operation_error(
            operation,
            "activation dtype must be float16, bfloat16, or float32",
        ));
    }
    if quantized_weights.dtype() != MlxDtype::UInt32 {
        return Err(runtime_operation_error(
            operation,
            "quantized weights must have uint32 dtype",
        ));
    }
    if !is_supported_quantized_float_dtype(scales.dtype()) {
        return Err(runtime_operation_error(
            operation,
            "affine scales must be float16, bfloat16, or float32",
        ));
    }
    if !is_supported_quantized_float_dtype(biases.dtype()) {
        return Err(runtime_operation_error(
            operation,
            "affine biases must be float16, bfloat16, or float32",
        ));
    }

    let activation_shape = activations.shape();
    let weight_shape = quantized_weights.shape();
    let scale_shape = scales.shape();
    let bias_shape = biases.shape();
    if activation_shape.len() < 2 || weight_shape.len() < 2 {
        return Err(runtime_operation_error(
            operation,
            "activation and weight arrays must both have rank at least two",
        ));
    }
    if scale_shape != bias_shape {
        return Err(runtime_operation_error(
            operation,
            "affine scales and biases must have the same shape",
        ));
    }

    let activation_inner_dimension = positive_last_dimension(&activation_shape, operation)?;
    let packed_weight_tail_dimension = positive_last_dimension(&weight_shape, operation)?;
    let weight_row_dimension = positive_second_last_dimension(&weight_shape, operation)?;
    let expanded_tail_dimension = packed_weight_tail_dimension
        .checked_mul(32)
        .and_then(|packed_bits| (packed_bits % bits == 0).then_some(packed_bits / bits))
        .ok_or_else(|| {
            runtime_operation_error(
                operation,
                "packed weight shape overflows or is incompatible with the requested bit width",
            )
        })?;

    let scale_group_count = if transpose_weights {
        if activation_inner_dimension != expanded_tail_dimension {
            return Err(runtime_operation_error(
                operation,
                "activation inner dimension does not match the expanded quantized weight input",
            ));
        }
        exact_group_count(expanded_tail_dimension, group_size, operation)?
    } else {
        if activation_inner_dimension != weight_row_dimension {
            return Err(runtime_operation_error(
                operation,
                "activation inner dimension does not match the untransposed quantized weight rows",
            ));
        }
        exact_group_count(expanded_tail_dimension, group_size, operation)?
    };
    let mut expected_scale_shape = weight_shape;
    let expected_scale_tail_position = expected_scale_shape.len() - 1;
    expected_scale_shape[expected_scale_tail_position] = scale_group_count;
    if scale_shape != expected_scale_shape {
        return Err(runtime_operation_error(
            operation,
            "affine scale and bias shapes do not match the packed weight shape",
        ));
    }
    Ok(())
}

const fn is_mlx_supported_affine_group_size(group_size: i32) -> bool {
    matches!(group_size, 32 | 64 | 128)
}

const fn is_mlx_supported_affine_bit_width(bits: i32) -> bool {
    matches!(bits, 2 | 3 | 4 | 5 | 6 | 8)
}

fn validate_gather_quantized_matmul_indices(
    lhs_indices: Option<&MlxArray>,
    rhs_indices: Option<&MlxArray>,
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "build affine gather_qmm";
    if lhs_indices.is_none() && rhs_indices.is_none() {
        return Err(runtime_operation_error(
            OPERATION,
            "gather_qmm requires at least one index array",
        ));
    }
    if let Some(lhs_index_array) = lhs_indices
        && !is_integral_index_dtype(lhs_index_array.dtype())
    {
        return Err(runtime_operation_error(
            OPERATION,
            "lhs indices must have an integral dtype",
        ));
    }
    if let Some(rhs_index_array) = rhs_indices
        && !is_integral_index_dtype(rhs_index_array.dtype())
    {
        return Err(runtime_operation_error(
            OPERATION,
            "rhs indices must have an integral dtype",
        ));
    }
    Ok(())
}

fn positive_last_dimension(shape: &[i32], operation: &'static str) -> Result<i32, MlxRuntimeError> {
    shape
        .last()
        .copied()
        .filter(|dimension| *dimension > 0)
        .ok_or_else(|| {
            runtime_operation_error(operation, "array shape must have a positive tail dimension")
        })
}

fn positive_second_last_dimension(
    shape: &[i32],
    operation: &'static str,
) -> Result<i32, MlxRuntimeError> {
    shape
        .get(shape.len().saturating_sub(2))
        .copied()
        .filter(|dimension| *dimension > 0)
        .ok_or_else(|| {
            runtime_operation_error(operation, "array shape must have a positive row dimension")
        })
}

fn exact_group_count(
    expanded_dimension: i32,
    group_size: i32,
    operation: &'static str,
) -> Result<i32, MlxRuntimeError> {
    if expanded_dimension % group_size != 0 {
        return Err(runtime_operation_error(
            operation,
            "expanded quantized dimension must be divisible by the affine group size",
        ));
    }
    Ok(expanded_dimension / group_size)
}

fn is_supported_quantized_float_dtype(dtype: MlxDtype) -> bool {
    matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::BFloat16
    )
}

fn is_integral_index_dtype(dtype: MlxDtype) -> bool {
    matches!(
        dtype,
        MlxDtype::UInt8
            | MlxDtype::UInt16
            | MlxDtype::UInt32
            | MlxDtype::UInt64
            | MlxDtype::Int8
            | MlxDtype::Int16
            | MlxDtype::Int32
            | MlxDtype::Int64
    )
}

fn runtime_operation_error(operation: &'static str, description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.to_owned(),
    }
}

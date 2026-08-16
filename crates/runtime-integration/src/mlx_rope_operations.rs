use crate::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Applies the certified nontraditional Llama rotary embedding.
    pub fn rope(
        &self,
        input: &MlxArray,
        dimensions: i32,
        base: f32,
        offset_tokens: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let optional_base = raw::mlx_optional_float {
            value: base,
            has_value: true,
        };
        self.output_array("apply MLX rotary embedding", |output, stream| {
            // SAFETY: Input and stream are live, the optional scalar is passed
            // by value, and an empty freqs handle selects generated frequencies.
            unsafe {
                raw::mlx_fast_rope(
                    output,
                    input.raw(),
                    dimensions,
                    false,
                    optional_base,
                    1.0,
                    offset_tokens,
                    MlxArray::empty_raw(),
                    stream,
                )
            }
        })
    }

    /// Applies native MLX RoPE to one text sequence at explicit token positions.
    pub fn rope_with_token_position_offsets(
        &self,
        input: &MlxArray,
        token_position_offsets: &MlxArray,
        dimensions: i32,
        base: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let input_shape = input.shape();
        if input_shape.len() != 4
            || input_shape[0] != 1
            || token_position_offsets.shape() != [input_shape[2]]
            || token_position_offsets.dtype() != MlxDtype::Int32
        {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "apply MLX rotary embedding at token positions",
                description: "input must be one rank-four sequence with one int32 offset per token"
                    .to_owned(),
            });
        }
        let token_batched_input = self.transpose_axes(input, &[2, 1, 0, 3])?;
        let optional_base = raw::mlx_optional_float {
            value: base,
            has_value: true,
        };
        let token_batched_output =
            self.output_array("apply dynamic MLX rotary embedding", |output, stream| {
                // SAFETY: Inputs and stream are live, each batch row has one
                // int32 offset, and an empty freqs handle selects generated frequencies.
                unsafe {
                    raw::mlx_fast_rope_dynamic(
                        output,
                        token_batched_input.raw(),
                        dimensions,
                        false,
                        optional_base,
                        1.0,
                        token_position_offsets.raw(),
                        MlxArray::empty_raw(),
                        stream,
                    )
                }
            })?;
        self.transpose_axes(&token_batched_output, &[2, 1, 0, 3])
    }

    /// Applies rotary embedding with caller-supplied frequency denominators.
    ///
    /// MLX takes the reciprocal of this array, so callers pass denominators such
    /// as `theta^(2i/d)`, never inverse frequencies.
    pub fn rope_with_custom_frequencies(
        &self,
        input: &MlxArray,
        dimensions: i32,
        frequency_denominators: &MlxArray,
        scale: f32,
        offset_tokens: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_rope_dimensions(input, dimensions, frequency_denominators, scale)?;
        let no_base = raw::mlx_optional_float {
            value: 0.0,
            has_value: false,
        };
        self.output_array(
            "apply MLX rotary embedding with custom frequency denominators",
            |output, stream| {
                // SAFETY: Inputs and stream are live; `has_value=false` selects
                // the supplied frequency array instead of generated frequencies.
                unsafe {
                    raw::mlx_fast_rope(
                        output,
                        input.raw(),
                        dimensions,
                        false,
                        no_base,
                        scale,
                        offset_tokens,
                        frequency_denominators.raw(),
                        stream,
                    )
                }
            },
        )
    }

    /// Applies custom-denominator rotary embedding at explicit token positions.
    pub fn rope_with_custom_frequencies_at_positions(
        &self,
        input: &MlxArray,
        token_position_offsets: &MlxArray,
        dimensions: i32,
        frequency_denominators: &MlxArray,
        scale: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let input_shape = input.shape();
        if input_shape.len() != 4 || input_shape[0] != 1 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "apply custom MLX rotary embedding at token positions",
                description: "input must be one rank-four sequence".to_owned(),
            });
        }
        validate_rope_dimensions(input, dimensions, frequency_denominators, scale)?;
        if token_position_offsets.shape() != [input_shape[2]]
            || token_position_offsets.dtype() != MlxDtype::Int32
        {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "apply custom MLX rotary embedding at token positions",
                description: "token positions must be int32 with one offset per token".to_owned(),
            });
        }
        // MLX dynamic RoPE accepts one offset per batch row, so transpose token
        // rows into the batch dimension and restore the original axes afterward.
        let token_batched_input = self.transpose_axes(input, &[2, 1, 0, 3])?;
        let no_base = raw::mlx_optional_float {
            value: 0.0,
            has_value: false,
        };
        let token_batched_output = self.output_array(
            "apply dynamic MLX rotary embedding with custom frequency denominators",
            |output, stream| {
                // SAFETY: All input handles and stream are live and validated.
                unsafe {
                    raw::mlx_fast_rope_dynamic(
                        output,
                        token_batched_input.raw(),
                        dimensions,
                        false,
                        no_base,
                        scale,
                        token_position_offsets.raw(),
                        frequency_denominators.raw(),
                        stream,
                    )
                }
            },
        )?;
        self.transpose_axes(&token_batched_output, &[2, 1, 0, 3])
    }
}

fn validate_rope_dimensions(
    input: &MlxArray,
    dimensions: i32,
    frequency_denominators: &MlxArray,
    scale: f32,
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "validate MLX rotary embedding dimensions";
    if dimensions <= 0 || dimensions % 2 != 0 {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "rotary dimensions must be positive and even".to_owned(),
        });
    }
    let frequency_shape = frequency_denominators.shape();
    if frequency_shape.as_slice() != [dimensions / 2] {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "frequency denominators must contain one value per rotary pair".to_owned(),
        });
    }
    if frequency_denominators.dtype() != MlxDtype::Float32 {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "frequency denominators must be Float32".to_owned(),
        });
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "rotary position scale must be positive and finite".to_owned(),
        });
    }
    let input_shape = input.shape();
    if input_shape.len() != 4 || dimensions > input_shape[3] {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "rotary input must be rank four and contain the rotary width".to_owned(),
        });
    }
    Ok(())
}

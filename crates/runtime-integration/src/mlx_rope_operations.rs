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
}

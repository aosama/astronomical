use crate::decoder_cache::{
    ConvolutionState, DecoderCacheLayerLayout, DecoderCacheState, FullAttentionKeyValueState,
    GatedDeltaRecurrentState,
};
use astronomical_runtime_integration::MlxRuntimeError;

pub(super) fn decoder_cache_layout_projection_error(
    decoder_cache_layout_error: crate::decoder_cache::DecoderCacheLayoutError,
) -> MlxRuntimeError {
    request_decoder_state_error_from_string(format!(
        "validated decoder-cache tensor payload geometry is invalid: {decoder_cache_layout_error}"
    ))
}

pub(super) fn request_decoder_layer_state_from_layout(
    decoder_cache_layer_layout: &DecoderCacheLayerLayout,
    full_attention_kv_state_growth_tokens_override: Option<i32>,
) -> Result<DecoderCacheState, MlxRuntimeError> {
    match decoder_cache_layer_layout {
        DecoderCacheLayerLayout::AppendOnlyAttention {
            capacity_growth_tokens,
            ..
        } => {
            let full_attention_kv_state_growth_tokens =
                full_attention_kv_state_growth_tokens_override.map_or_else(
                    || {
                        i32::try_from(*capacity_growth_tokens).map_err(|_| {
                            request_decoder_state_error(
                                "append-only attention growth exceeds the i32 range",
                            )
                        })
                    },
                    Ok,
                )?;
            if full_attention_kv_state_growth_tokens <= 0 {
                return Err(request_decoder_state_error(
                    "append-only attention growth must be positive",
                ));
            }
            Ok(DecoderCacheState::AppendOnlyAttention {
                attention: FullAttentionKeyValueState::empty_with_validated_growth_tokens(
                    full_attention_kv_state_growth_tokens,
                ),
            })
        }
        DecoderCacheLayerLayout::Composite { components } => {
            let [
                DecoderCacheLayerLayout::RecurrentTensor {
                    tensor: convolution_tensor,
                },
                DecoderCacheLayerLayout::RecurrentTensor {
                    tensor: recurrent_tensor,
                },
            ] = components.as_slice()
            else {
                return Err(request_decoder_state_error(
                    "Qwen linear attention requires convolution and recurrent tensor components",
                ));
            };
            if convolution_tensor.qualified_role_name()
                != crate::qwen3_5::decoder::cache_layout::QWEN_CONVOLUTION_TENSOR_ROLE
                || recurrent_tensor.qualified_role_name()
                    != crate::qwen3_5::decoder::cache_layout::QWEN_RECURRENCE_TENSOR_ROLE
            {
                return Err(request_decoder_state_error(
                    "Qwen linear attention tensor roles do not match the model contract",
                ));
            }
            let [1, convolution_history_tokens, convolution_dimension] =
                convolution_tensor.dimensions()
            else {
                return Err(request_decoder_state_error(
                    "Qwen convolution state must have rank three with batch size one",
                ));
            };
            let [
                1,
                recurrent_value_head_count,
                recurrent_value_head_dimension,
                recurrent_key_head_dimension,
            ] = recurrent_tensor.dimensions()
            else {
                return Err(request_decoder_state_error(
                    "Qwen recurrent state must have rank four with batch size one",
                ));
            };
            let convolution_kernel_dimension =
                convolution_history_tokens.checked_add(1).ok_or_else(|| {
                    request_decoder_state_error("Qwen convolution history length overflowed")
                })?;
            Ok(DecoderCacheState::Composite {
                convolution: ConvolutionState::empty_with_shape(
                    i32::try_from(convolution_kernel_dimension).map_err(|_| {
                        request_decoder_state_error("Qwen convolution kernel dimension exceeds i32")
                    })?,
                    i32::try_from(*convolution_dimension).map_err(|_| {
                        request_decoder_state_error("Qwen convolution dimension exceeds i32")
                    })?,
                ),
                recurrent: GatedDeltaRecurrentState::empty_with_shape(
                    i32::try_from(*recurrent_value_head_count).map_err(|_| {
                        request_decoder_state_error("Qwen recurrent value head count exceeds i32")
                    })?,
                    i32::try_from(*recurrent_value_head_dimension).map_err(|_| {
                        request_decoder_state_error(
                            "Qwen recurrent value head dimension exceeds i32",
                        )
                    })?,
                    i32::try_from(*recurrent_key_head_dimension).map_err(|_| {
                        request_decoder_state_error("Qwen recurrent key head dimension exceeds i32")
                    })?,
                ),
            })
        }
        DecoderCacheLayerLayout::RecurrentTensor { .. } => Err(request_decoder_state_error(
            "Qwen decoder layers cannot contain a standalone recurrent tensor",
        )),
    }
}

pub(super) fn request_decoder_state_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "create Qwen3.5 request decoder state",
        description: description.to_owned(),
    }
}

pub(super) fn request_decoder_state_error_from_string(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "create Qwen3.5 request decoder state",
        description,
    }
}

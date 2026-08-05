use astronomical_runtime_integration::MlxArray;

use super::error::invalid_request_decoder_state;
use super::{Qwen3_5ExecutionError, RequestDecoderStateStack};
use crate::DecoderCacheState;

pub(super) fn forward_state_arrays<'state>(
    output: &'state MlxArray,
    request_decoder_state: &'state RequestDecoderStateStack,
) -> Result<Vec<&'state MlxArray>, Qwen3_5ExecutionError> {
    collect_evaluation_arrays(Some(output), request_decoder_state, false)
}

pub(super) fn decoder_state_arrays(
    request_decoder_state: &RequestDecoderStateStack,
) -> Result<Vec<&MlxArray>, Qwen3_5ExecutionError> {
    collect_evaluation_arrays(None, request_decoder_state, true)
}

fn collect_evaluation_arrays<'state>(
    output: Option<&'state MlxArray>,
    request_decoder_state: &'state RequestDecoderStateStack,
    include_forward_reachable_state_arrays: bool,
) -> Result<Vec<&'state MlxArray>, Qwen3_5ExecutionError> {
    // Forward outputs already reach attention storage and the recurrent kernel's sibling state.
    // Convolution state branches from its input, so it always remains an explicit evaluation root.
    let decoder_layer_count = request_decoder_state.layer_count();
    let mut evaluation_arrays =
        Vec::with_capacity(decoder_layer_count * 2 + usize::from(output.is_some()));
    evaluation_arrays.extend(output);
    for layer_index in 0..decoder_layer_count {
        match request_decoder_state.layer(layer_index) {
            Some(DecoderCacheState::Composite {
                convolution,
                recurrent,
            }) if convolution.state().is_some() && recurrent.state().is_some() => {
                let convolution = convolution.state().ok_or_else(|| {
                    invalid_request_decoder_state(
                        layer_index,
                        "model forward did not populate the convolution state array",
                    )
                })?;
                let recurrent = recurrent.state().ok_or_else(|| {
                    invalid_request_decoder_state(
                        layer_index,
                        "model forward did not populate the recurrent state array",
                    )
                })?;
                evaluation_arrays.push(convolution);
                if include_forward_reachable_state_arrays {
                    evaluation_arrays.push(recurrent);
                }
            }
            Some(DecoderCacheState::AppendOnlyAttention { attention })
                if attention.keys_state().is_some() && attention.values_state().is_some() =>
            {
                let attention_keys = attention.keys_state().ok_or_else(|| {
                    invalid_request_decoder_state(
                        layer_index,
                        "model forward did not populate the attention key state array",
                    )
                })?;
                let attention_values = attention.values_state().ok_or_else(|| {
                    invalid_request_decoder_state(
                        layer_index,
                        "model forward did not populate the attention value state array",
                    )
                })?;
                if include_forward_reachable_state_arrays {
                    evaluation_arrays.push(attention_keys);
                    evaluation_arrays.push(attention_values);
                }
            }
            _ => {
                return Err(invalid_request_decoder_state(
                    layer_index,
                    "model forward did not populate both required state arrays",
                ));
            }
        }
    }
    Ok(evaluation_arrays)
}

pub(super) fn validate_generated_token_forward(
    generated_token: &MlxArray,
    starting_position_tokens: u32,
    layer_model_state_count: usize,
    expected_decoder_layer_count: usize,
    maximum_position_count: u32,
) -> Result<(), Qwen3_5ExecutionError> {
    if generated_token.shape() != [1, 1] {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "generated token array must have shape [1, 1]",
        });
    }
    if layer_model_state_count != expected_decoder_layer_count {
        return Err(Qwen3_5ExecutionError::DecoderLayerCountMismatch {
            actual_decoder_layer_count: layer_model_state_count,
            expected_decoder_layer_count,
        });
    }
    if starting_position_tokens >= maximum_position_count {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "forward chunk exceeds the certified maximum position count",
        });
    }
    Ok(())
}

pub(super) fn validate_forward_input(
    token_ids: &[u32],
    starting_position_tokens: u32,
    layer_model_state_count: usize,
    expected_decoder_layer_count: usize,
    vocabulary_size: u32,
    maximum_position_count: u32,
) -> Result<i32, Qwen3_5ExecutionError> {
    if token_ids.is_empty() {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "forward chunk must contain at least one token",
        });
    }
    if layer_model_state_count != expected_decoder_layer_count {
        return Err(Qwen3_5ExecutionError::DecoderLayerCountMismatch {
            actual_decoder_layer_count: layer_model_state_count,
            expected_decoder_layer_count,
        });
    }
    if token_ids
        .iter()
        .any(|token_id| *token_id >= vocabulary_size)
    {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "forward chunk contains an out-of-vocabulary token ID",
        });
    }
    let token_count =
        u32::try_from(token_ids.len()).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
            description: "forward chunk token count exceeds the u32 range",
        })?;
    if starting_position_tokens
        .checked_add(token_count)
        .is_none_or(|ending_position| ending_position > maximum_position_count)
    {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "forward chunk exceeds certified model positions",
        });
    }
    i32::try_from(token_count).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
        description: "forward chunk token count exceeds the MLX integer range",
    })
}

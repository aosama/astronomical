use crate::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError, DecoderCacheTensorDtype,
    DecoderCacheTensorLayout,
};

use super::Qwen3_5Config;

pub(crate) const QWEN_CONVOLUTION_TENSOR_ROLE: &str = "linear.convolution";
pub(crate) const QWEN_RECURRENCE_TENSOR_ROLE: &str = "linear.gated_delta_recurrent";
pub(crate) const QWEN_ATTENTION_KEYS_TENSOR_ROLE: &str = "attention.keys";
pub(crate) const QWEN_ATTENTION_VALUES_TENSOR_ROLE: &str = "attention.values";

/// Exact live execution dtypes for one Qwen decoder layer's persistent state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5DecoderLayerCacheDtypes {
    LinearAttention {
        convolution: DecoderCacheTensorDtype,
    },
    FullAttention {
        keys: DecoderCacheTensorDtype,
        values: DecoderCacheTensorDtype,
    },
}

/// Combines model geometry with dtypes derived from the bound execution graph.
///
/// Geometry comes from validated configuration, while scalar types come from
/// actual weight propagation. Keeping those inputs separate prevents a nominal
/// activation dtype from silently narrowing persistent state.
pub fn qwen3_5_decoder_cache_layout(
    qwen3_5_config: &Qwen3_5Config,
    full_attention_key_value_growth_tokens: usize,
    decoder_layer_cache_dtypes: &[Qwen3_5DecoderLayerCacheDtypes],
) -> Result<DecoderCacheLayout, DecoderCacheLayoutError> {
    let decoder_layer_count = qwen3_5_config.layer_count() as usize;
    if decoder_layer_cache_dtypes.len() != decoder_layer_count {
        return Err(DecoderCacheLayoutError::ExecutionDtypeLayerCountMismatch {
            expected_layer_count: decoder_layer_count,
            actual_layer_count: decoder_layer_cache_dtypes.len(),
        });
    }
    let linear_convolution_state_dimension =
        usize::try_from(qwen3_5_config.linear_convolution_state_dimension()).map_err(|_| {
            DecoderCacheLayoutError::ModelConfigurationDimensionOutsideUsizeRange {
                dimension_name: "linear convolution state",
            }
        })?;
    let full_attention_key_value_dimensions = vec![
        1,
        qwen3_5_config.key_value_head_count() as usize,
        0,
        qwen3_5_config.head_dimension() as usize,
    ];
    let linear_convolution_dimensions = vec![
        1,
        (qwen3_5_config.linear_convolution_kernel_dimension() as usize).saturating_sub(1),
        linear_convolution_state_dimension,
    ];
    let linear_recurrent_dimensions = vec![
        1,
        qwen3_5_config.linear_value_head_count() as usize,
        qwen3_5_config.linear_value_head_dimension() as usize,
        qwen3_5_config.linear_key_head_dimension() as usize,
    ];
    let decoder_layer_layouts = decoder_layer_cache_dtypes
        .iter()
        .enumerate()
        .map(|(decoder_layer_index, decoder_layer_cache_dtypes)| {
            match (
                qwen3_5_config.decoder_layer_is_full_attention(decoder_layer_index),
                decoder_layer_cache_dtypes,
            ) {
                (true, Qwen3_5DecoderLayerCacheDtypes::FullAttention { keys, values }) => {
                    Ok(DecoderCacheLayerLayout::append_only_attention(
                        DecoderCacheTensorLayout::sequence(
                            QWEN_ATTENTION_KEYS_TENSOR_ROLE,
                            *keys,
                            full_attention_key_value_dimensions.clone(),
                            2,
                        ),
                        DecoderCacheTensorLayout::sequence(
                            QWEN_ATTENTION_VALUES_TENSOR_ROLE,
                            *values,
                            full_attention_key_value_dimensions.clone(),
                            2,
                        ),
                        full_attention_key_value_growth_tokens,
                    ))
                }
                (false, Qwen3_5DecoderLayerCacheDtypes::LinearAttention { convolution }) => {
                    Ok(DecoderCacheLayerLayout::composite(vec![
                        DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
                            QWEN_CONVOLUTION_TENSOR_ROLE,
                            *convolution,
                            linear_convolution_dimensions.clone(),
                        )),
                        DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
                            // The gated-delta recurrent accumulator is Float32 in live execution;
                            // preserve it independently from the BF16 activation state above.
                            QWEN_RECURRENCE_TENSOR_ROLE,
                            DecoderCacheTensorDtype::Float32,
                            linear_recurrent_dimensions.clone(),
                        )),
                    ]))
                }
                _ => Err(DecoderCacheLayoutError::ExecutionDtypeLayerFamilyMismatch {
                    layer_index: decoder_layer_index,
                }),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    DecoderCacheLayout::new(decoder_layer_layouts)
}

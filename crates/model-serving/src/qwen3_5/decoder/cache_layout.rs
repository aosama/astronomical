use crate::{
    DEFAULT_APPEND_ONLY_ATTENTION_CAPACITY_GROWTH_TOKENS, DecoderCacheLayerLayout,
    DecoderCacheLayout, DecoderCacheLayoutError, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};

use super::Qwen3_5Config;

pub(crate) const QWEN_CONVOLUTION_TENSOR_ROLE: &str = "linear.convolution";
pub(crate) const QWEN_RECURRENCE_TENSOR_ROLE: &str = "linear.gated_delta_recurrent";
pub(crate) const QWEN_ATTENTION_KEYS_TENSOR_ROLE: &str = "attention.keys";
pub(crate) const QWEN_ATTENTION_VALUES_TENSOR_ROLE: &str = "attention.values";

/// Derives Qwen's live and persistent decoder-state contract from validated model metadata.
pub fn qwen3_5_decoder_cache_layout(
    qwen3_5_config: &Qwen3_5Config,
) -> Result<DecoderCacheLayout, DecoderCacheLayoutError> {
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
    // The live Qwen kernels retain full-attention key/value and convolution state as BF16.
    // Persist the execution dtype exactly: a narrower or wider cache dtype would make restore
    // either numerically different or incompatible with the model's normal forward path.
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
    let decoder_layer_layouts = (0..qwen3_5_config.layer_count() as usize)
        .map(|decoder_layer_index| {
            if qwen3_5_config.decoder_layer_is_full_attention(decoder_layer_index) {
                DecoderCacheLayerLayout::append_only_attention(
                    DecoderCacheTensorLayout::sequence(
                        QWEN_ATTENTION_KEYS_TENSOR_ROLE,
                        DecoderCacheTensorDtype::BFloat16,
                        full_attention_key_value_dimensions.clone(),
                        2,
                    ),
                    DecoderCacheTensorLayout::sequence(
                        QWEN_ATTENTION_VALUES_TENSOR_ROLE,
                        DecoderCacheTensorDtype::BFloat16,
                        full_attention_key_value_dimensions.clone(),
                        2,
                    ),
                    DEFAULT_APPEND_ONLY_ATTENTION_CAPACITY_GROWTH_TOKENS,
                )
            } else {
                DecoderCacheLayerLayout::composite(vec![
                    DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
                        QWEN_CONVOLUTION_TENSOR_ROLE,
                        DecoderCacheTensorDtype::BFloat16,
                        linear_convolution_dimensions.clone(),
                    )),
                    DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
                        // The gated-delta recurrent accumulator is Float32 in live execution;
                        // preserve it independently from the BF16 activation state above.
                        QWEN_RECURRENCE_TENSOR_ROLE,
                        DecoderCacheTensorDtype::Float32,
                        linear_recurrent_dimensions.clone(),
                    )),
                ])
            }
        })
        .collect();
    DecoderCacheLayout::new(decoder_layer_layouts)
}

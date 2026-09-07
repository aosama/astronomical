//! Append-only attention cache layout derived from family config.
//!
//! Sequence axis 2 matches the live K2 key/value tensors
//! `[batch, kv_heads, tokens, head_dim]`.

use crate::decoder_cache::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError, DecoderCacheTensorDtype,
    DecoderCacheTensorLayout,
};
use crate::k2_horizon_mova::configuration::K2HorizonMoVAConfig;

const APPEND_ONLY_CAPACITY_GROWTH_TOKENS: usize = 256;

/// Builds the SSD prompt-cache tensor contract for one K2 Horizon MoVA member.
pub fn k2_horizon_mova_decoder_cache_layout(
    config: &K2HorizonMoVAConfig,
) -> Result<DecoderCacheLayout, DecoderCacheLayoutError> {
    let tensor_dtype = DecoderCacheTensorDtype::BFloat16;
    let key_value_head_count = config.num_key_value_heads();
    let head_dimension = config.head_dim();
    let mut layer_layouts = Vec::with_capacity(config.num_hidden_layers());
    for _decoder_layer_index in 0..config.num_hidden_layers() {
        layer_layouts.push(DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "attention.keys",
                tensor_dtype,
                vec![1, key_value_head_count, 0, head_dimension],
                2,
            ),
            DecoderCacheTensorLayout::sequence(
                "attention.values",
                tensor_dtype,
                vec![1, key_value_head_count, 0, head_dimension],
                2,
            ),
            APPEND_ONLY_CAPACITY_GROWTH_TOKENS,
        ));
    }
    DecoderCacheLayout::new(layer_layouts)
}

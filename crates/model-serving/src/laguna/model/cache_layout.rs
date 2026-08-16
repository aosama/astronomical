//! Decoder-cache layout derived only from canonical layer descriptors.

use crate::decoder_cache::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError, DecoderCacheTensorDtype,
    DecoderCacheTensorLayout,
};
use crate::laguna::normalization::{
    LagunaAttentionDescriptor, LagunaCacheDescriptor, LagunaExecutionDtype, LagunaTargetContract,
};

/// Builds an architecture-neutral cache layout from a Laguna target contract.
pub fn laguna_decoder_cache_layout(
    contract: &LagunaTargetContract,
) -> Result<DecoderCacheLayout, DecoderCacheLayoutError> {
    let tensor_dtype = cache_dtype(contract.model().execution_dtype());
    let mut layer_layouts = Vec::with_capacity(contract.layers().len());
    for layer_descriptor in contract.layers() {
        layer_layouts.push(layer_cache_layout(
            layer_descriptor.attention(),
            tensor_dtype,
        ));
    }
    DecoderCacheLayout::new(layer_layouts)
}

fn layer_cache_layout(
    attention: &LagunaAttentionDescriptor,
    tensor_dtype: DecoderCacheTensorDtype,
) -> DecoderCacheLayerLayout {
    let key_value_head_count = attention.key_value_head_count() as usize;
    let head_dimension = attention.head_dimension() as usize;
    match *attention.cache() {
        LagunaCacheDescriptor::AppendOnly => DecoderCacheLayerLayout::append_only_attention(
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
            256,
        ),
        LagunaCacheDescriptor::Rotating { window_size } => {
            DecoderCacheLayerLayout::rotating_window_attention(
                DecoderCacheTensorLayout::fixed(
                    "attention.keys",
                    tensor_dtype,
                    vec![
                        1,
                        key_value_head_count,
                        window_size as usize,
                        head_dimension,
                    ],
                ),
                DecoderCacheTensorLayout::fixed(
                    "attention.values",
                    tensor_dtype,
                    vec![
                        1,
                        key_value_head_count,
                        window_size as usize,
                        head_dimension,
                    ],
                ),
                window_size as usize,
            )
        }
    }
}

fn cache_dtype(execution_dtype: LagunaExecutionDtype) -> DecoderCacheTensorDtype {
    match execution_dtype {
        LagunaExecutionDtype::Float16 => DecoderCacheTensorDtype::Float16,
        LagunaExecutionDtype::Bfloat16 => DecoderCacheTensorDtype::BFloat16,
        LagunaExecutionDtype::Float32 => DecoderCacheTensorDtype::Float32,
    }
}

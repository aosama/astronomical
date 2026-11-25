//! An empty request decoder state built from a Qwen 3.5 configuration alone.
//!
//! Synthetic tests use it, because they have no bound weights to inspect.
//! Production model loading derives each layer's exact dtype from the live
//! affine graph before it creates state, so it never calls this.

use runtime_integration::MlxRuntimeError;
use text_generation::RequestDecoderStateStack;

use super::Qwen3_5Config;
use super::astro_decoder_state_cache_layout::{
    Qwen3_5AstroDecoderStateCacheLayerDtypes, qwen3_5_astro_decoder_state_cache_layout,
};
use crate::astro_decoder_state_cache::AstroDecoderStateCacheTensorDtype;

/// Creates empty decoder-layer state entries for `qwen3_5_config`, with
/// BFloat16 activations and an explicit full-attention KV capacity-growth
/// granularity.
pub fn request_decoder_state_from_qwen3_5_config(
    qwen3_5_config: &Qwen3_5Config,
    full_attention_kv_state_growth_tokens: i32,
) -> Result<RequestDecoderStateStack, MlxRuntimeError> {
    let full_attention_key_value_growth_tokens =
        usize::try_from(full_attention_kv_state_growth_tokens)
            .ok()
            .filter(|growth_tokens| *growth_tokens > 0)
            .ok_or_else(|| {
                request_decoder_state_error(
                    "full-attention KV-state growth tokens must be positive".to_owned(),
                )
            })?;
    let configured_activation_dtype = AstroDecoderStateCacheTensorDtype::BFloat16;
    let mut astro_decoder_state_cache_layer_dtypes = Vec::new();
    for decoder_layer_index in 0..qwen3_5_config.layer_count() as usize {
        astro_decoder_state_cache_layer_dtypes.push(
            if qwen3_5_config.decoder_layer_is_full_attention(decoder_layer_index) {
                Qwen3_5AstroDecoderStateCacheLayerDtypes::FullAttention {
                    keys: configured_activation_dtype,
                    values: configured_activation_dtype,
                }
            } else {
                Qwen3_5AstroDecoderStateCacheLayerDtypes::LinearAttention {
                    convolution: configured_activation_dtype,
                }
            },
        );
    }
    let astro_decoder_state_cache_layout = qwen3_5_astro_decoder_state_cache_layout(
        qwen3_5_config,
        full_attention_key_value_growth_tokens,
        &astro_decoder_state_cache_layer_dtypes,
    )
    .map_err(|astro_decoder_state_cache_layout_error| {
        request_decoder_state_error(format!(
            "validated Qwen astro-decoder-state-cache layout is invalid: {astro_decoder_state_cache_layout_error}"
        ))
    })?;
    RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        &astro_decoder_state_cache_layout,
        full_attention_kv_state_growth_tokens,
    )
}

fn request_decoder_state_error(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "create Qwen3.5 request decoder state",
        description,
    }
}

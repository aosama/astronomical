//! The smallest storage contract the store publishes real files for: one
//! float16 attention layer, no recurrent state, blocks as long as its
//! capacity growth. Shared by the tests that need a contract but no model
//! geometry.

use crate::astro_decoder_state_cache::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCacheTensorDtype, AstroDecoderStateCacheTensorLayout,
};
use crate::astro_ssd_prompt_cache::model_contract::AstroSsdPromptCacheModelContract;

pub fn synthetic_attention_model_contract(
    model_id: &str,
    model_revision: &str,
    attention_capacity_growth_tokens: usize,
) -> AstroSsdPromptCacheModelContract {
    let astro_decoder_state_cache_layout = AstroDecoderStateCacheLayout::new(vec![
        AstroDecoderStateCacheLayerLayout::append_only_attention(
            AstroDecoderStateCacheTensorLayout::sequence(
                "attention.keys",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            AstroDecoderStateCacheTensorLayout::sequence(
                "attention.values",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            attention_capacity_growth_tokens,
        ),
    ])
    .expect("the synthetic sequence layout should be valid");
    AstroSsdPromptCacheModelContract::resolve(
        model_id.to_owned(),
        model_revision.to_owned(),
        astro_decoder_state_cache_layout,
        128,
        1_000_000,
        1_000_000,
        None,
        4,
    )
    .expect("the synthetic storage contract should resolve")
}

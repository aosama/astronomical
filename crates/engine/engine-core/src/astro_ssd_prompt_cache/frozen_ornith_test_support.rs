//! The frozen Ornith 1.0 storage geometry, written out for the engine core's
//! `astro_ssd_prompt_cache` tests.
//!
//! These are the values the Qwen 3.5 family derives from its frozen Ornith
//! 1.0 configuration: 40 decoder layers, every fourth one full attention, two
//! key-value heads of dimension 256, and linear attention layers with a
//! kernel-4 convolution over 8,192 channels and 32 recurrent heads of
//! 128 by 128. The engine core cannot read the family's configuration, so the
//! storage contracts are checked against this fixed geometry instead; the
//! family's own tests cover how its configuration becomes a layout.

use crate::astro_decoder_state_cache::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCacheTensorDtype, AstroDecoderStateCacheTensorLayout,
};
use crate::astro_ssd_prompt_cache::PersistentVisualEmbeddingModelContract;

pub const ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
pub const ORNITH_1_0_35B_OPTIQ_4BIT_REVISION: &str = "ce62c23d34b91d84f838e0b292d517dbe4b9b60f";
pub const FROZEN_ORNITH_1_0_MAXIMUM_POSITION_COUNT: usize = 262_144;
const FROZEN_ORNITH_1_0_LAYER_COUNT: usize = 40;
const FULL_ATTENTION_KEY_VALUE_DIMENSIONS: [usize; 4] = [1, 2, 0, 256];
const LINEAR_CONVOLUTION_DIMENSIONS: [usize; 3] = [1, 3, 8_192];
const LINEAR_RECURRENT_DIMENSIONS: [usize; 4] = [1, 32, 128, 128];

/// Whether decoder layer `layer_index` of Ornith 1.0 is full attention.
fn is_full_attention_layer(layer_index: usize) -> bool {
    layer_index % 4 == 3
}

/// The Ornith 1.0 decoder state layout with every tensor the family stores
/// in BFloat16 held in BFloat16, as the frozen fixtures have no loaded
/// weights to take their dtypes from.
pub fn frozen_ornith_1_0_bfloat16_layout(
    full_attention_key_value_growth_tokens: usize,
) -> AstroDecoderStateCacheLayout {
    let mut decoder_layer_layouts = Vec::with_capacity(FROZEN_ORNITH_1_0_LAYER_COUNT);
    for layer_index in 0..FROZEN_ORNITH_1_0_LAYER_COUNT {
        decoder_layer_layouts.push(if is_full_attention_layer(layer_index) {
            full_attention_layer_layout(full_attention_key_value_growth_tokens)
        } else {
            linear_attention_layer_layout()
        });
    }
    AstroDecoderStateCacheLayout::new(decoder_layer_layouts)
        .expect("the frozen Ornith 1.0 layout should be valid")
}

/// The Ornith 1.0 visual embedding contract: 2,048-wide projected rows and
/// at most 16,384 of them per image.
pub fn frozen_ornith_1_0_visual_embedding_model_contract() -> PersistentVisualEmbeddingModelContract
{
    PersistentVisualEmbeddingModelContract::new(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.to_owned(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.to_owned(),
        2_048,
        16_384,
    )
}

fn full_attention_layer_layout(
    full_attention_key_value_growth_tokens: usize,
) -> AstroDecoderStateCacheLayerLayout {
    AstroDecoderStateCacheLayerLayout::append_only_attention(
        AstroDecoderStateCacheTensorLayout::sequence(
            "attention.keys",
            AstroDecoderStateCacheTensorDtype::BFloat16,
            FULL_ATTENTION_KEY_VALUE_DIMENSIONS.to_vec(),
            2,
        ),
        AstroDecoderStateCacheTensorLayout::sequence(
            "attention.values",
            AstroDecoderStateCacheTensorDtype::BFloat16,
            FULL_ATTENTION_KEY_VALUE_DIMENSIONS.to_vec(),
            2,
        ),
        full_attention_key_value_growth_tokens,
    )
}

fn linear_attention_layer_layout() -> AstroDecoderStateCacheLayerLayout {
    AstroDecoderStateCacheLayerLayout::composite(vec![
        AstroDecoderStateCacheLayerLayout::recurrent_tensor(
            AstroDecoderStateCacheTensorLayout::fixed(
                "linear.convolution",
                AstroDecoderStateCacheTensorDtype::BFloat16,
                LINEAR_CONVOLUTION_DIMENSIONS.to_vec(),
            ),
        ),
        AstroDecoderStateCacheLayerLayout::recurrent_tensor(
            AstroDecoderStateCacheTensorLayout::fixed(
                "linear.gated_delta_recurrent",
                AstroDecoderStateCacheTensorDtype::Float32,
                LINEAR_RECURRENT_DIMENSIONS.to_vec(),
            ),
        ),
    ])
}

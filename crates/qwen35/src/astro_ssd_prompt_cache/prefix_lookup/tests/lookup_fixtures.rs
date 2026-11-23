//! The frozen Ornith 1.0 fixture web for the persisted-storage contracts, built
//! from the crate's public API. The configuration documents come from
//! `test_harness::fixtures::frozen_ornith_1_0_fixture`, shared with the Qwen 3.5 unit
//! tests and the acceptance GPU contracts; only the typed wrappers live here.

use crate::AstroSsdPromptCacheModelContract;
use crate::{
    AstroDecoderStateCacheTensorDtype, ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
    ORNITH_1_0_35B_OPTIQ_4BIT_REVISION, Qwen3_5AstroDecoderStateCacheLayerDtypes, Qwen3_5Config,
};
pub use test_harness::fixtures::frozen_ornith_1_0_fixture::frozen_ornith_1_0_optiq_config_bytes;

pub fn frozen_ornith_1_0_config() -> Qwen3_5Config {
    Qwen3_5Config::from_json_bytes(&frozen_ornith_1_0_optiq_config_bytes())
        .expect("the frozen Ornith 1.0 config should parse")
}

pub fn astro_ssd_prompt_cache_model_contract() -> AstroSsdPromptCacheModelContract {
    let frozen_ornith_1_0_config = frozen_ornith_1_0_config();
    // This fixture has no bound MLX arrays. Supply its frozen BF16
    // state contract explicitly; real engine acceptance uses load-derived
    // dtypes and read the resulting block geometry back from the engine.
    let astro_decoder_state_cache_layer_dtypes =
        bfloat16_astro_decoder_state_cache_layer_dtypes(&frozen_ornith_1_0_config);
    AstroSsdPromptCacheModelContract::resolve(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.to_owned(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.to_owned(),
        crate::qwen3_5_astro_decoder_state_cache_layout(
            &frozen_ornith_1_0_config,
            256,
            &astro_decoder_state_cache_layer_dtypes,
        )
        .expect(
            "the frozen Ornith 1.0 configuration should build an astro-decoder-state-cache layout",
        ),
        frozen_ornith_1_0_config.maximum_position_count() as usize,
        20_000_000_000,
        50_000_000_000,
        None,
        4,
    )
    .expect("the frozen Ornith 1.0 configuration should resolve a storage contract")
}

pub fn bfloat16_astro_decoder_state_cache_layer_dtypes(
    qwen3_5_config: &Qwen3_5Config,
) -> Vec<Qwen3_5AstroDecoderStateCacheLayerDtypes> {
    astro_decoder_state_cache_layer_dtypes(
        qwen3_5_config,
        AstroDecoderStateCacheTensorDtype::BFloat16,
    )
}

fn astro_decoder_state_cache_layer_dtypes(
    qwen3_5_config: &Qwen3_5Config,
    activation_state_dtype: AstroDecoderStateCacheTensorDtype,
) -> Vec<Qwen3_5AstroDecoderStateCacheLayerDtypes> {
    (0..qwen3_5_config.layer_count() as usize)
        .map(|decoder_layer_index| {
            if qwen3_5_config.decoder_layer_is_full_attention(decoder_layer_index) {
                Qwen3_5AstroDecoderStateCacheLayerDtypes::FullAttention {
                    keys: activation_state_dtype,
                    values: activation_state_dtype,
                }
            } else {
                Qwen3_5AstroDecoderStateCacheLayerDtypes::LinearAttention {
                    convolution: activation_state_dtype,
                }
            }
        })
        .collect()
}

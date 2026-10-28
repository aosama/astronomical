//! Test fixtures shared by the restore-support tests: a real storage contract
//! resolved from the frozen Ornith 1.0 configuration, so key derivation is
//! exercised against production geometry rather than a synthetic stub.

use crate::PersistentPromptCacheModelContract;
use crate::decoder_cache::DecoderCacheTensorDtype;
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::decoder::cache_layout::{
    Qwen3_5DecoderLayerCacheDtypes, qwen3_5_decoder_cache_layout,
};
use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config;

const TEST_MLX_MEMORY_CEILING_BYTES: u64 = 20_000_000_000;
const TEST_SSD_QUOTA_BYTES: u64 = 50_000_000_000;

/// Resolves a storage contract from the frozen configuration, mirroring how
/// production derives one for a loaded model.
pub(crate) fn test_model_contract() -> PersistentPromptCacheModelContract {
    let ornith_config = frozen_ornith_1_0_config();
    let decoder_layer_cache_dtypes = bfloat16_decoder_layer_cache_dtypes(&ornith_config);
    PersistentPromptCacheModelContract::resolve(
        "restore-support-test-model".to_owned(),
        "rev-unit-test".to_owned(),
        qwen3_5_decoder_cache_layout(&ornith_config, 256, &decoder_layer_cache_dtypes)
            .expect("the frozen configuration should build a decoder-cache layout"),
        ornith_config.maximum_position_count() as usize,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_SSD_QUOTA_BYTES,
        None,
        4,
    )
    .expect("the test model should resolve a storage contract")
}

fn bfloat16_decoder_layer_cache_dtypes(
    ornith_config: &Qwen3_5Config,
) -> Vec<Qwen3_5DecoderLayerCacheDtypes> {
    (0..ornith_config.layer_count() as usize)
        .map(|decoder_layer_index| {
            if ornith_config.decoder_layer_is_full_attention(decoder_layer_index) {
                Qwen3_5DecoderLayerCacheDtypes::FullAttention {
                    keys: DecoderCacheTensorDtype::BFloat16,
                    values: DecoderCacheTensorDtype::BFloat16,
                }
            } else {
                Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
                    convolution: DecoderCacheTensorDtype::BFloat16,
                }
            }
        })
        .collect()
}

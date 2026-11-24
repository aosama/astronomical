//! Test fixtures shared by the restore-support tests: a real storage contract
//! resolved from the frozen Ornith 1.0 configuration, so key derivation is
//! exercised against production geometry rather than a synthetic stub.

use crate::astro_ssd_prompt_cache::AstroSsdPromptCacheModelContract;
use crate::astro_ssd_prompt_cache::frozen_ornith_test_support::{
    FROZEN_ORNITH_1_0_MAXIMUM_POSITION_COUNT, frozen_ornith_1_0_bfloat16_layout,
};

const TEST_MLX_MEMORY_CEILING_BYTES: u64 = 20_000_000_000;
const TEST_SSD_QUOTA_BYTES: u64 = 50_000_000_000;

pub fn test_model_contract() -> AstroSsdPromptCacheModelContract {
    AstroSsdPromptCacheModelContract::resolve(
        "restore-support-test-model".to_owned(),
        "rev-unit-test".to_owned(),
        frozen_ornith_1_0_bfloat16_layout(256),
        FROZEN_ORNITH_1_0_MAXIMUM_POSITION_COUNT,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_SSD_QUOTA_BYTES,
        None,
        4,
    )
    .expect("the test model should resolve a storage contract")
}

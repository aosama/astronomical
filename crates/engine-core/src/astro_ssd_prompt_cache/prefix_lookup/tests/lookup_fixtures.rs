//! The frozen Ornith 1.0 storage contract the prefix-lookup contracts run
//! against, built from the engine core's frozen geometry.

use crate::astro_ssd_prompt_cache::AstroSsdPromptCacheModelContract;
use crate::astro_ssd_prompt_cache::frozen_ornith_test_support::{
    FROZEN_ORNITH_1_0_MAXIMUM_POSITION_COUNT, ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
    ORNITH_1_0_35B_OPTIQ_4BIT_REVISION, frozen_ornith_1_0_bfloat16_layout,
};

pub fn astro_ssd_prompt_cache_model_contract() -> AstroSsdPromptCacheModelContract {
    AstroSsdPromptCacheModelContract::resolve(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.to_owned(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.to_owned(),
        frozen_ornith_1_0_bfloat16_layout(256),
        FROZEN_ORNITH_1_0_MAXIMUM_POSITION_COUNT,
        20_000_000_000,
        50_000_000_000,
        None,
        4,
    )
    .expect("the frozen Ornith 1.0 geometry should resolve a storage contract")
}

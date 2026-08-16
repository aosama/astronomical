use astronomical_model_serving::{LagunaModel, PerformanceAttribution};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[test]
fn should_select_the_highest_logit_token_on_the_gpu() {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("Laguna sampling test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let prompt_logits = runtime
        .array_from_f32(&[1.0, 5.0, 2.0, 0.5, 9.0, 3.0], &[1, 2, 3])
        .expect("prompt logits should be placed on the runtime");
    let mut performance_attribution = PerformanceAttribution::disabled();
    let selected_token_id =
        LagunaModel::greedy_token_id(&runtime, &prompt_logits, &mut performance_attribution)
            .expect("greedy sampling should copy one token ID");
    assert_eq!(
        selected_token_id, 1,
        "the last-token row [0.5, 9.0, 3.0] should select vocabulary index 1"
    );
}

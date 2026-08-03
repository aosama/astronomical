use astronomical_model_serving::qwen3_5_moe_apply_top_p_mask;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_keep_the_candidate_that_crosses_the_top_p_threshold() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the sampler test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let mut candidate_probabilities = vec![0.0_f32; 20];
    candidate_probabilities[0] = 0.5;
    candidate_probabilities[1] = 0.4;
    candidate_probabilities[2] = 0.1;
    let candidate_logits = candidate_probabilities
        .iter()
        .map(|probability| probability.ln())
        .collect::<Vec<_>>();
    let probability_array = runtime
        .array_from_f32(&candidate_probabilities, &[1, 1, 20])
        .expect("the candidate probabilities should be valid");
    let logit_array = runtime
        .array_from_f32(&candidate_logits, &[1, 1, 20])
        .expect("the candidate logits should be valid");

    let masked_logits =
        qwen3_5_moe_apply_top_p_mask(&runtime, &probability_array, &logit_array, 600)
            .expect("top-p filtering should build a valid MLX graph");
    let masked_logit_values = masked_logits
        .to_vec_f32()
        .expect("the masked logits should evaluate as float32");

    assert_eq!(masked_logit_values[0], candidate_logits[0]);
    assert_eq!(masked_logit_values[1], candidate_logits[1]);
    assert!(
        masked_logit_values[2..]
            .iter()
            .all(|logit| logit.is_infinite() && logit.is_sign_negative())
    );
}

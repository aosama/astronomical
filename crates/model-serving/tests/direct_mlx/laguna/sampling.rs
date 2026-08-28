use std::time::Duration;

use astronomical_model_serving::{LagunaModel, LagunaSamplerConfig, PerformanceAttribution};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const LAGUNA_SAMPLING_TEST_TIMEOUT: Duration = Duration::from_secs(30);

#[tokio::test]
async fn should_select_the_highest_logit_token_on_the_gpu() {
    tokio::time::timeout(LAGUNA_SAMPLING_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = sampling_runtime();
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
    })
    .await
    .expect("greedy Laguna sampling should finish within 30 seconds");
}

#[tokio::test]
async fn should_match_greedy_when_top_k_keeps_only_the_highest_logit() {
    tokio::time::timeout(LAGUNA_SAMPLING_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = sampling_runtime();
        let prompt_logits = runtime
            .array_from_f32(&[0.5, 9.0, 3.0], &[1, 1, 3])
            .expect("prompt logits should be placed on the runtime");
        let mut performance_attribution = PerformanceAttribution::disabled();
        let greedy_token_id =
            LagunaModel::greedy_token_id(&runtime, &prompt_logits, &mut performance_attribution)
                .expect("greedy sampling should copy one token ID");
        let mut random_state = runtime
            .random_key(42)
            .expect("a sampling key should be created");
        let sampled_token_id = LagunaModel::sampled_token_id(
            &runtime,
            &prompt_logits,
            1_000,
            1_000,
            Some(1),
            &mut random_state,
            &mut performance_attribution,
        )
        .expect("top-k 1 sampling should copy one token ID");
        assert_eq!(greedy_token_id, 1);
        assert_eq!(sampled_token_id, greedy_token_id);
    })
    .await
    .expect("top-k 1 Laguna sampling should finish within 30 seconds");
}

#[tokio::test]
async fn should_draw_both_tied_high_logits_instead_of_always_argmax() {
    tokio::time::timeout(LAGUNA_SAMPLING_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = sampling_runtime();
        let prompt_logits = runtime
            .array_from_f32(&[5.0, 5.0, 0.0], &[1, 1, 3])
            .expect("tied high logits should be placed on the runtime");
        let mut performance_attribution = PerformanceAttribution::disabled();
        let greedy_token_id =
            LagunaModel::greedy_token_id(&runtime, &prompt_logits, &mut performance_attribution)
                .expect("greedy sampling should copy one token ID");
        assert_eq!(greedy_token_id, 0);

        let mut random_state = runtime
            .random_key(7)
            .expect("a sampling key should be created");
        let mut selected_first = false;
        let mut selected_second = false;
        for _ in 0..64 {
            let sampled_token_id = LagunaModel::sampled_token_id(
                &runtime,
                &prompt_logits,
                1_000,
                1_000,
                Some(2),
                &mut random_state,
                &mut performance_attribution,
            )
            .expect("tied-logit sampling should copy one token ID");
            match sampled_token_id {
                0 => selected_first = true,
                1 => selected_second = true,
                other => panic!("top-k 2 should not select vocabulary index {other}"),
            }
        }
        assert!(
            selected_first && selected_second,
            "temperature 1.0 with two equal top logits should not collapse to greedy argmax"
        );
    })
    .await
    .expect("tied-logit Laguna sampling should finish within 30 seconds");
}

#[tokio::test]
async fn should_use_the_laguna_default_top_k_when_the_artifact_omits_it() {
    tokio::time::timeout(LAGUNA_SAMPLING_TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = sampling_runtime();
        let prompt_logits = runtime
            .array_from_f32(&[0.5, 9.0, 3.0, 1.0, 0.2], &[1, 1, 5])
            .expect("prompt logits should be placed on the runtime");
        let mut omitted_top_k_random_state = runtime
            .random_key(42)
            .expect("a sampling key should be created");
        let mut explicit_top_k_random_state = runtime
            .random_key(42)
            .expect("a matching sampling key should be created");
        let mut performance_attribution = PerformanceAttribution::disabled();
        let omitted_top_k_token_id = LagunaModel::sampled_token_id(
            &runtime,
            &prompt_logits,
            1_000,
            1_000,
            None,
            &mut omitted_top_k_random_state,
            &mut performance_attribution,
        )
        .expect("omitted top-k should sample");
        let explicit_default_top_k_token_id = LagunaModel::sampled_token_id(
            &runtime,
            &prompt_logits,
            1_000,
            1_000,
            Some(LagunaSamplerConfig::DEFAULT_SAMPLING_TOP_K),
            &mut explicit_top_k_random_state,
            &mut performance_attribution,
        )
        .expect("explicit default top-k should sample");
        assert_eq!(
            omitted_top_k_token_id, explicit_default_top_k_token_id,
            "omitted Laguna top-k should execute the same truncation as top_k 20"
        );
    })
    .await
    .expect("default Laguna top-k sampling should finish within 30 seconds");
}

fn sampling_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("Laguna sampling test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

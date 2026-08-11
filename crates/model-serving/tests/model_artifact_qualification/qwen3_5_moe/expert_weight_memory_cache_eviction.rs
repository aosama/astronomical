use std::time::Duration;

use astronomical_model_serving::{
    ExpertWeightMemoryCacheStatistics, Qwen3_5ArtifactValidator, Qwen3_5Model,
};
use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS;

const EXPERT_WEIGHT_MEMORY_CACHE_EVICTION_TEST_TIMEOUT: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "loads real expert pages and proves byte-bounded native global LRU eviction"]
async fn should_evict_the_oldest_unselected_expert_without_reloading_selected_hits() {
    tokio::time::timeout(
        EXPERT_WEIGHT_MEMORY_CACHE_EVICTION_TEST_TIMEOUT,
        run_oldest_unselected_expert_eviction_proof(),
    )
    .await
    .expect("the native expert-cache eviction proof should finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads real expert pages and proves a zero retention budget uses an ephemeral native snapshot"]
async fn should_use_an_ephemeral_snapshot_when_the_route_exceeds_the_retention_ceiling() {
    tokio::time::timeout(
        EXPERT_WEIGHT_MEMORY_CACHE_EVICTION_TEST_TIMEOUT,
        run_ephemeral_route_proof(),
    )
    .await
    .expect("the ephemeral native expert proof should finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads the model-artifact checkpoint and proves an expert miss derives a finite automatic live budget"]
async fn should_derive_the_automatic_live_budget_on_an_expert_cache_miss() {
    tokio::time::timeout(
        EXPERT_WEIGHT_MEMORY_CACHE_EVICTION_TEST_TIMEOUT,
        run_automatic_live_budget_derivation_proof(),
    )
    .await
    .expect("the automatic native expert-cache budget proof should finish within 120 seconds");
}

async fn run_oldest_unselected_expert_eviction_proof() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (runtime, _config, expert_pager) =
        super::construct_model_artifact_expert_pager("[native-expert-cache-eviction]").await;
    let initial_expert_ids = (0..8).collect::<Vec<_>>();
    let turnover_expert_ids = (0..7).chain([8]).collect::<Vec<_>>();
    let initial_route = selected_expert_indices(&runtime, &initial_expert_ids);
    let (_, initial_report) = expert_pager
        .prepare_native_expert_snapshot(&runtime, 0, &initial_route, true)
        .expect("the initial route should populate native retention");
    assert_eq!(initial_report.cache_miss_count(), 8);
    let initial_statistics = expert_pager.native_expert_cache_statistics();
    assert_eq!(initial_statistics.resident_expert_count(), 8);
    assert!(expert_pager.freeze_native_expert_retention_growth());

    let turnover_route = selected_expert_indices(&runtime, &turnover_expert_ids);
    let (_, turnover_report) = expert_pager
        .prepare_native_expert_snapshot(&runtime, 0, &turnover_route, true)
        .expect("one new expert should replace the oldest unselected native slot");
    assert_eq!(turnover_report.cache_hit_count(), 7);
    assert_eq!(turnover_report.cache_miss_count(), 1);
    assert_eq!(turnover_report.disk_page_load_count(), 1);
    let turnover_statistics = expert_pager.native_expert_cache_statistics();
    assert_eq!(turnover_statistics.resident_expert_count(), 8);
    assert_eq!(turnover_statistics.eviction_count(), 1);

    let (_, warm_report) = expert_pager
        .prepare_native_expert_snapshot(&runtime, 0, &turnover_route, true)
        .expect("the replacement route should remain warm");
    assert_eq!(warm_report.cache_hit_count(), 8);
    assert_eq!(warm_report.disk_page_load_count(), 0);
    assert!(expert_pager.resume_native_expert_retention_growth());
}

async fn run_ephemeral_route_proof() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (runtime, _config, expert_pager) =
        super::construct_model_artifact_expert_pager("[native-expert-cache-ephemeral]").await;
    expert_pager
        .update_native_expert_retention_ceiling(0)
        .expect("the native retention ceiling should accept zero bytes");
    assert!(expert_pager.freeze_native_expert_retention_growth());
    let selected_expert_ids = (0..8).collect::<Vec<_>>();
    let selected_route = selected_expert_indices(&runtime, &selected_expert_ids);
    let (_ephemeral_snapshot, request_report) = expert_pager
        .prepare_native_expert_snapshot(&runtime, 0, &selected_route, true)
        .expect("the oversized route should execute through an ephemeral native snapshot");
    assert_eq!(request_report.cache_miss_count(), 8);
    assert_eq!(request_report.disk_page_load_count(), 8);
    assert_eq!(
        expert_pager
            .native_expert_cache_statistics()
            .resident_expert_count(),
        0
    );
    assert!(expert_pager.resume_native_expert_retention_growth());
}

fn selected_expert_indices(runtime: &MlxRuntime, expert_ids: &[i32]) -> MlxArray {
    runtime
        .array_from_i32(expert_ids, &[1, expert_ids.len() as i32])
        .expect("the selected expert route should be valid")
}

async fn run_automatic_live_budget_derivation_proof() {
    let cache_statistics =
        load_model_decode_one_token_and_read_cache_statistics("[native-expert-cache-budget]").await;
    assert_ne!(
        cache_statistics.maximum_resident_payload_byte_count,
        u64::MAX
    );
    assert!(cache_statistics.entry_count > 0);
    assert!(
        cache_statistics.resident_payload_byte_count
            <= cache_statistics.maximum_resident_payload_byte_count
    );
}

async fn load_model_decode_one_token_and_read_cache_statistics(
    progress_prefix: &str,
) -> ExpertWeightMemoryCacheStatistics {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    eprintln!("{progress_prefix} status=start phase=artifact_validation");
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the model artifact should validate before native cache-budget testing");
    let config = validated_artifact.config().clone();
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the MLX runtime should initialize for native cache-budget testing");
    let qwen3_5_model = Qwen3_5Model::load(
        runtime,
        validated_artifact,
        &model_directory,
        false,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the model artifact should load with native demand-only expert caching");
    let mut request_decoder_state = crate::common::standard_request_decoder_state(&config);
    qwen3_5_model
        .prefill_chunck(
            &SAY_HI_PROMPT_TOKEN_IDS[..SAY_HI_PROMPT_TOKEN_IDS.len() - 1],
            0,
            &mut request_decoder_state,
        )
        .expect("the prompt prefix should materialize before one paged decode step");
    let logits = qwen3_5_model
        .forward_chunk(
            &SAY_HI_PROMPT_TOKEN_IDS[SAY_HI_PROMPT_TOKEN_IDS.len() - 1..],
            (SAY_HI_PROMPT_TOKEN_IDS.len() - 1) as u32,
            &mut request_decoder_state,
        )
        .expect("one paged decode step should complete under the automatic native budget");
    qwen3_5_model
        .greedy_token_id(&logits)
        .expect("the paged decode logits should produce one greedy token");
    qwen3_5_model.expert_weight_memory_cache_statistics()
}

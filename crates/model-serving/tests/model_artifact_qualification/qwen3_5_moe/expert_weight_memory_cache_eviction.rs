use std::time::Duration;

use astronomical_model_serving::{
    ExpertWeightMemoryCache, ExpertWeightMemoryCacheStatistics, Qwen3_5ArtifactValidator,
    Qwen3_5Model, RequestDecoderStateStack,
};
use astronomical_runtime_integration::MlxRuntime;

use super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS;

const EXPERT_WEIGHT_MEMORY_CACHE_EVICTION_TEST_TIMEOUT: Duration = Duration::from_secs(120);

fn create_expert_weight_memory_cache_with_maximum(
    expert_layer_count: usize,
    maximum_resident_payload_bytes: u64,
) -> ExpertWeightMemoryCache {
    let mut expert_weight_memory_cache =
        ExpertWeightMemoryCache::new(expert_layer_count, vec![0; expert_layer_count]);
    expert_weight_memory_cache
        .update_maximum_resident_payload_byte_count(maximum_resident_payload_bytes);
    expert_weight_memory_cache
}

#[tokio::test]
#[ignore = "loads real expert pages and proves byte-bounded layer-local eviction"]
async fn should_evict_the_oldest_unselected_expert_without_reloading_selected_hits() {
    tokio::time::timeout(
        EXPERT_WEIGHT_MEMORY_CACHE_EVICTION_TEST_TIMEOUT,
        run_oldest_unselected_expert_eviction_proof(),
    )
    .await
    .expect("the expert weight memory-cache eviction proof should finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads real expert pages and proves a too-small retention budget uses transient paging"]
async fn should_use_a_transient_page_when_one_layer_selection_exceeds_its_cache_share() {
    tokio::time::timeout(
        EXPERT_WEIGHT_MEMORY_CACHE_EVICTION_TEST_TIMEOUT,
        run_too_small_layer_cache_share_proof(),
    )
    .await
    .expect("the transient expert page proof should finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads real expert pages and proves a shrinking live budget evicts the oldest layer entries"]
async fn should_evict_oldest_layer_entries_when_the_live_budget_shrinks() {
    tokio::time::timeout(
        EXPERT_WEIGHT_MEMORY_CACHE_EVICTION_TEST_TIMEOUT,
        run_shrinking_live_budget_proof(),
    )
    .await
    .expect(
        "the shrinking expert weight memory-cache budget proof should finish within 120 seconds",
    );
}

#[tokio::test]
#[ignore = "loads the model-artifact checkpoint and proves an expert miss derives a finite automatic live budget"]
async fn should_derive_the_automatic_live_budget_on_an_expert_cache_miss() {
    tokio::time::timeout(
        EXPERT_WEIGHT_MEMORY_CACHE_EVICTION_TEST_TIMEOUT,
        run_automatic_live_budget_derivation_proof(),
    )
    .await
    .expect(
        "the automatic expert weight memory-cache budget proof should finish within 120 seconds",
    );
}

async fn run_oldest_unselected_expert_eviction_proof() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    eprintln!("[expert-cache-eviction] status=start phase=construct_model_artifact_expert_pager");
    let (runtime, _config, expert_pager) =
        super::construct_model_artifact_expert_pager("[expert-cache-eviction]").await;
    let layer_index = 0usize;
    let initial_expert_ids = (0usize..8).collect::<Vec<_>>();
    let selected_expert_ids_after_turnover = (0usize..7).chain([8]).collect::<Vec<_>>();

    eprintln!("[expert-cache-eviction] status=progress phase=measure_layer_capacity");
    let (_direct_page, initial_page_manifest, _) = expert_pager
        .load_selected_experts(&runtime, layer_index, &initial_expert_ids)
        .expect("the direct page should establish the exact eight-expert payload size");
    let maximum_resident_payload_bytes = initial_page_manifest
        .payload_byte_count
        .checked_mul(expert_pager.layer_count() as u64)
        .expect("the per-layer expert payload budget should fit in bytes");
    let mut expert_weight_memory_cache = create_expert_weight_memory_cache_with_maximum(
        expert_pager.layer_count(),
        maximum_resident_payload_bytes,
    );

    eprintln!("[expert-cache-eviction] status=progress phase=populate_initial_layer_entries");
    let (_, _, initial_load_report) = expert_pager
        .load_selected_experts_through_memory_cache(
            &runtime,
            layer_index,
            &initial_expert_ids,
            &mut expert_weight_memory_cache,
        )
        .expect("the initial selected experts should populate the bounded memory cache");
    assert_eq!(initial_load_report.cache_miss_count, 8);

    eprintln!("[expert-cache-eviction] status=progress phase=replace_one_layer_entry");
    let (_, _, turnover_report) = expert_pager
        .load_selected_experts_through_memory_cache(
            &runtime,
            layer_index,
            &selected_expert_ids_after_turnover,
            &mut expert_weight_memory_cache,
        )
        .expect("one new selected expert should replace the oldest unselected layer entry");
    assert_eq!(turnover_report.cache_hit_count, 7);
    assert_eq!(turnover_report.cache_miss_count, 1);
    assert_eq!(turnover_report.disk_page_load_count, 1);

    let (_, _, repeated_turnover_report) = expert_pager
        .load_selected_experts_through_memory_cache(
            &runtime,
            layer_index,
            &selected_expert_ids_after_turnover,
            &mut expert_weight_memory_cache,
        )
        .expect("the retained turnover selection should be fully warm");
    assert_eq!(repeated_turnover_report.cache_hit_count, 8);
    assert_eq!(repeated_turnover_report.disk_page_load_count, 0);

    let final_cache_statistics = expert_weight_memory_cache.statistics();
    assert_eq!(final_cache_statistics.entry_count, 8);
    assert_eq!(final_cache_statistics.eviction_count, 1);
    assert!(
        final_cache_statistics.resident_payload_byte_count
            <= initial_page_manifest.payload_byte_count,
        "the layer-local payload must remain within its equal share of the global byte ceiling"
    );
    eprintln!(
        "[expert-cache-eviction] status=success entries={} evictions={} resident_payload_bytes={} maximum_resident_payload_bytes={}",
        final_cache_statistics.entry_count,
        final_cache_statistics.eviction_count,
        final_cache_statistics.resident_payload_byte_count,
        maximum_resident_payload_bytes
    );
}

async fn run_too_small_layer_cache_share_proof() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    eprintln!("[expert-cache-transient] status=start phase=construct_model_artifact_expert_pager");
    let (runtime, _config, expert_pager) =
        super::construct_model_artifact_expert_pager("[expert-cache-transient]").await;
    let layer_index = 0usize;
    let selected_expert_ids = (0usize..8).collect::<Vec<_>>();
    let (_direct_page, selected_page_manifest, _) = expert_pager
        .load_selected_experts(&runtime, layer_index, &selected_expert_ids)
        .expect("the direct page should establish the exact eight-expert payload size");
    let too_small_per_layer_payload_bytes = selected_page_manifest.payload_byte_count - 1;
    let maximum_resident_payload_bytes = too_small_per_layer_payload_bytes
        .checked_mul(expert_pager.layer_count() as u64)
        .expect("the intentionally small global cache budget should fit in bytes");
    let mut expert_weight_memory_cache = create_expert_weight_memory_cache_with_maximum(
        expert_pager.layer_count(),
        maximum_resident_payload_bytes,
    );

    eprintln!("[expert-cache-transient] status=progress phase=load_transient_selected_page");
    let (_, _, transient_page_report) = expert_pager
        .load_selected_experts_through_memory_cache(
            &runtime,
            layer_index,
            &selected_expert_ids,
            &mut expert_weight_memory_cache,
        )
        .expect("a too-small retention budget should still serve one transient selected page");

    assert_eq!(transient_page_report.cache_hit_count, 0);
    assert_eq!(transient_page_report.cache_miss_count, 8);
    assert_eq!(transient_page_report.disk_page_load_count, 8);
    let final_cache_statistics = expert_weight_memory_cache.statistics();
    assert_eq!(final_cache_statistics.entry_count, 0);
    assert_eq!(final_cache_statistics.resident_payload_byte_count, 0);
    eprintln!(
        "[expert-cache-transient] status=success selected_payload_bytes={} per_layer_cache_share_bytes={} disk_page_loads={}",
        selected_page_manifest.payload_byte_count,
        too_small_per_layer_payload_bytes,
        transient_page_report.disk_page_load_count
    );
}

async fn run_shrinking_live_budget_proof() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    eprintln!("[expert-cache-shrink] status=start phase=construct_model_artifact_expert_pager");
    let (runtime, _config, expert_pager) =
        super::construct_model_artifact_expert_pager("[expert-cache-shrink]").await;
    let layer_index = 0usize;
    let older_expert_ids = (0usize..8).collect::<Vec<_>>();
    let newer_expert_ids = (8usize..16).collect::<Vec<_>>();
    let (_direct_page, eight_expert_page_manifest, _) = expert_pager
        .load_selected_experts(&runtime, layer_index, &older_expert_ids)
        .expect("the direct page should establish the exact eight-expert payload size");
    let one_set_global_maximum_bytes = eight_expert_page_manifest
        .payload_byte_count
        .checked_mul(expert_pager.layer_count() as u64)
        .expect("the one-set global maximum should fit in bytes");
    let two_set_global_maximum_bytes = one_set_global_maximum_bytes
        .checked_mul(2)
        .expect("the two-set global maximum should fit in bytes");
    let mut expert_weight_memory_cache = create_expert_weight_memory_cache_with_maximum(
        expert_pager.layer_count(),
        two_set_global_maximum_bytes,
    );

    for selected_expert_ids in [&older_expert_ids, &newer_expert_ids] {
        expert_pager
            .load_selected_experts_through_memory_cache(
                &runtime,
                layer_index,
                selected_expert_ids,
                &mut expert_weight_memory_cache,
            )
            .expect("both expert sets should fit before the live budget shrinks");
    }
    assert_eq!(expert_weight_memory_cache.statistics().entry_count, 16);

    eprintln!("[expert-cache-shrink] status=progress phase=apply_smaller_live_budget");
    expert_weight_memory_cache
        .update_maximum_resident_payload_byte_count(one_set_global_maximum_bytes);
    let shrunken_cache_statistics = expert_weight_memory_cache.statistics();
    assert_eq!(shrunken_cache_statistics.entry_count, 8);
    assert_eq!(shrunken_cache_statistics.eviction_count, 8);
    assert_eq!(
        shrunken_cache_statistics.maximum_resident_payload_byte_count,
        one_set_global_maximum_bytes
    );

    let (_, _, retained_newer_set_report) = expert_pager
        .load_selected_experts_through_memory_cache(
            &runtime,
            layer_index,
            &newer_expert_ids,
            &mut expert_weight_memory_cache,
        )
        .expect("the most recently used expert set should survive the budget shrink");
    assert_eq!(retained_newer_set_report.cache_hit_count, 8);
    assert_eq!(retained_newer_set_report.disk_page_load_count, 0);
    eprintln!(
        "[expert-cache-shrink] status=success entries={} evictions={} resident_payload_bytes={} maximum_resident_payload_bytes={}",
        shrunken_cache_statistics.entry_count,
        shrunken_cache_statistics.eviction_count,
        shrunken_cache_statistics.resident_payload_byte_count,
        shrunken_cache_statistics.maximum_resident_payload_byte_count
    );
}

async fn run_automatic_live_budget_derivation_proof() {
    let cache_statistics =
        load_model_decode_one_token_and_read_cache_statistics("[expert-cache-automatic-budget]")
            .await;
    assert_ne!(
        cache_statistics.maximum_resident_payload_byte_count,
        u64::MAX,
        "the first expert cache miss must replace the initial unbounded ceiling"
    );
    assert!(cache_statistics.entry_count > 0);
    assert!(
        cache_statistics.resident_payload_byte_count
            <= cache_statistics.maximum_resident_payload_byte_count
    );
    eprintln!(
        "[expert-cache-automatic-budget] status=success entries={} resident_payload_bytes={} maximum_resident_payload_bytes={}",
        cache_statistics.entry_count,
        cache_statistics.resident_payload_byte_count,
        cache_statistics.maximum_resident_payload_byte_count
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
        .expect(
            "the model-artifact checkpoint should validate before automatic cache-budget testing",
        );
    let config = validated_artifact.config().clone();
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the MLX runtime should initialize for automatic cache-budget testing");
    let qwen3_5_model = Qwen3_5Model::load(runtime, validated_artifact, &model_directory, false)
        .expect("the model-artifact checkpoint should load with automatic expert residency");
    let mut request_decoder_state = RequestDecoderStateStack::empty_from_config(&config);
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
        .expect("one paged decode step should complete under the automatic cache budget");
    qwen3_5_model
        .greedy_token_id(&logits)
        .expect("the paged decode logits should produce one greedy token");

    qwen3_5_model.expert_weight_memory_cache_statistics()
}

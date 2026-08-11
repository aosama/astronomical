//! Integration tests for expert paging with the real Ornith model artifact.
//!
//! These tests are marked `#[ignore]` because they require the pinned model
//! directory to be configured in `~/.astronomical/config.json`.

use std::future::Future;
use std::time::{Duration, Instant};

use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Model};
use astronomical_runtime_integration::MlxRuntime;
use tokio::time::{MissedTickBehavior, interval, sleep};

const EXPERT_PAGING_TEST_TIMEOUT: Duration = Duration::from_secs(120);

use super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS;

/// Constructs a Qwen3_5ExpertPager from the real Ornith model artifact and verifies
/// that layer plans are built correctly for all 40 MoE layers.
#[tokio::test]
#[ignore = "loads the complete pinned Ornith artifact for expert paging integration"]
async fn should_construct_expert_pager_and_build_layer_plans() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (_runtime, config, expert_pager) =
        super::construct_model_artifact_expert_pager("[expert-pager]").await;

    // The pager should have one layer plan per MoE layer (40 for Ornith).
    let layer_count = expert_pager.layer_count();
    assert_eq!(
        layer_count,
        config.layer_count() as usize,
        "Qwen3_5ExpertPager should have one layer plan per MoE layer"
    );
}

/// Loads selected expert weights for layer 0 and verifies manifest correctness.
/// Only top-K experts are loaded (K = experts_per_token from config).
#[tokio::test]
#[ignore = "loads the complete pinned Ornith artifact for expert paging integration"]
async fn should_load_selected_expert_weights_with_correct_manifest() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (runtime, config, expert_pager) =
        super::construct_model_artifact_expert_pager("[expert-pager]").await;

    // Select 8 experts for layer 0 (matching experts_per_token from config)
    let experts_per_token = config.experts_per_token() as usize;
    let selected_expert_ids = (0..experts_per_token as i32).collect::<Vec<_>>();
    let selected_expert_indices = runtime
        .array_from_i32(&selected_expert_ids, &[1, experts_per_token as i32])
        .expect("the selected expert route should be valid");
    let (_native_snapshot, request_report) = expert_pager
        .prepare_native_expert_snapshot(&runtime, 0, &selected_expert_indices, true)
        .expect("should load selected expert weights for layer 0");
    assert_eq!(request_report.cache_miss_count(), experts_per_token as u64);
    assert!(request_report.successful_source_read_byte_count() > 0);
    assert_eq!(request_report.payload_copy_byte_count(), 0);
}

/// Verifies that loading different expert selections produces correct page slots.
#[tokio::test]
#[ignore = "loads the complete pinned Ornith artifact for expert paging integration"]
async fn should_map_non_contiguous_expert_selections_to_correct_page_slots() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (runtime, _config, expert_pager) =
        super::construct_model_artifact_expert_pager("[expert-pager]").await;

    // Select non-contiguous experts: [0, 42, 100, 200, 255]
    let selected_expert_ids = [0, 42, 100, 200, 255];
    let selected_expert_indices = runtime
        .array_from_i32(&selected_expert_ids, &[1, selected_expert_ids.len() as i32])
        .expect("the non-contiguous route should be valid");
    let (_native_snapshot, request_report) = expert_pager
        .prepare_native_expert_snapshot(&runtime, 0, &selected_expert_indices, true)
        .expect("should load non-contiguous expert weights for layer 0");
    assert_eq!(
        request_report.cache_miss_count(),
        selected_expert_ids.len() as u64
    );
    assert_eq!(request_report.payload_copy_byte_count(), 0);
}

/// Lowest-level performance-design proof for expert paging cache behavior.
///
/// This intentionally avoids the supervisor, worker process, macOS app, REST API,
/// and shell scripts. It constructs only the real Qwen3_5ExpertPager, then asks for the
/// same single expert twice. The first request must load from SSD; the second
/// request must hit the in-memory expert cache and perform zero disk page loads.
#[tokio::test]
#[ignore = "loads one real Ornith expert page and proves the memory cache removes the second disk load"]
async fn should_load_same_expert_once_then_hit_memory_cache_without_second_disk_load() {
    require_expert_paging_test_completion(run_same_expert_cache_proof()).await;
}

/// Low-level top-K proof for one decode layer: a cold selected-expert request
/// loads each selected expert once, while the identical warm request performs
/// zero disk page loads.
#[tokio::test]
#[ignore = "loads eight real Ornith expert pages and proves warm top-k reuse avoids disk"]
async fn should_load_top_8_experts_once_then_hit_memory_cache_without_second_disk_loads() {
    require_expert_paging_test_completion(run_top_8_expert_cache_proof()).await;
}

#[tokio::test]
#[ignore = "loads two routed Ornith experts from different layers and proves one-expert-only retention"]
async fn should_cache_only_routed_one_expert_pages_across_layers() {
    require_expert_paging_test_completion(run_cross_layer_one_expert_cache_proof()).await;
}

/// Direct model-level proof that paged decode is wired to the expert memory cache.
///
/// This still avoids REST, the supervisor, the worker process, the macOS app,
/// and shell scripts. It runs the same prompt twice through the loaded model.
/// The first pass retains only routed one-expert pages; the second identical pass
/// should reuse that model-owned memory and add no disk loads.
#[tokio::test]
#[ignore = "loads the full Ornith model and proves direct paged decode reuses expert cache"]
async fn should_reuse_expert_memory_cache_across_direct_paged_decodes() {
    require_expert_paging_test_completion(run_direct_paged_decode_cache_reuse_proof()).await;
}

async fn run_direct_paged_decode_cache_reuse_proof() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let started_at = Instant::now();
    eprintln!("[direct-cache-proof] status=start phase=artifact_validation");
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the pinned Ornith artifact should validate before model cache proof");
    let config = validated_artifact.config().clone();
    eprintln!(
        "[direct-cache-proof] status=progress phase=artifact_validated shards={} payload_bytes={}",
        validated_artifact.shard_count(),
        validated_artifact.total_payload_bytes()
    );

    eprintln!("[direct-cache-proof] status=progress phase=runtime_init");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize for model cache proof");

    eprintln!("[direct-cache-proof] status=progress phase=model_load");
    let model_load_started_at = Instant::now();
    let qwen3_5_model = Qwen3_5Model::load(
        runtime,
        validated_artifact,
        &model_directory,
        false,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the Ornith model should load with demand-only one-expert caching");
    eprintln!(
        "[direct-cache-proof] status=progress phase=model_loaded elapsed_seconds={:.2}",
        model_load_started_at.elapsed().as_secs_f64()
    );

    let initial_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    assert_eq!(
        initial_cache_statistics.disk_page_load_count, 0,
        "freshly loaded model should start with no expert cache disk loads"
    );

    eprintln!("[direct-cache-proof] status=progress phase=first_identical_decode");
    run_one_say_hi_paged_decode(&qwen3_5_model, &config, "first")
        .expect("first direct paged decode should complete");
    let after_first_decode_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    eprintln!(
        "[direct-cache-proof] status=progress phase=first_decode_done cache_entries={} disk_page_loads={} one_expert_page_hits={} cache_misses={}",
        after_first_decode_statistics.entry_count,
        after_first_decode_statistics.disk_page_load_count,
        after_first_decode_statistics.cache_hit_count,
        after_first_decode_statistics.cache_miss_count
    );
    assert!(
        after_first_decode_statistics.entry_count > initial_cache_statistics.entry_count,
        "first paged pass should retain routed one-expert pages"
    );

    eprintln!("[direct-cache-proof] status=progress phase=second_identical_decode");
    run_one_say_hi_paged_decode(&qwen3_5_model, &config, "second")
        .expect("second direct paged decode should complete");
    let after_second_decode_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    eprintln!(
        "[direct-cache-proof] status=progress phase=second_decode_done cache_entries={} disk_page_loads={} one_expert_page_hits={} cache_misses={}",
        after_second_decode_statistics.entry_count,
        after_second_decode_statistics.disk_page_load_count,
        after_second_decode_statistics.cache_hit_count,
        after_second_decode_statistics.cache_miss_count
    );
    assert_eq!(
        after_second_decode_statistics.disk_page_load_count,
        after_first_decode_statistics.disk_page_load_count,
        "second identical paged decode should not add disk page loads"
    );
    let first_pass_reuse_hit_count = after_first_decode_statistics.cache_hit_count;
    let second_pass_reuse_hit_count = after_second_decode_statistics.cache_hit_count;
    assert!(
        second_pass_reuse_hit_count > first_pass_reuse_hit_count,
        "second identical paged pass should add one-expert-page reuse hits"
    );
    eprintln!(
        "[direct-cache-proof] status=success elapsed_seconds={:.2}",
        started_at.elapsed().as_secs_f64()
    );
}

fn run_one_say_hi_paged_decode(
    qwen3_5_model: &Qwen3_5Model,
    config: &astronomical_model_serving::Qwen3_5Config,
    decode_label: &str,
) -> Result<(), astronomical_model_serving::Qwen3_5ExecutionError> {
    let mut request_decoder_state = crate::common::standard_request_decoder_state(config);
    let prefill_started_at = Instant::now();
    qwen3_5_model.prefill_chunck(
        &SAY_HI_PROMPT_TOKEN_IDS[..SAY_HI_PROMPT_TOKEN_IDS.len() - 1],
        0,
        &mut request_decoder_state,
    )?;
    eprintln!(
        "[direct-cache-proof] status=progress decode={decode_label} phase=prefill_done elapsed_ms={:.2}",
        prefill_started_at.elapsed().as_secs_f64() * 1000.0
    );

    let decode_started_at = Instant::now();
    let final_position_logits = qwen3_5_model.forward_chunk(
        &SAY_HI_PROMPT_TOKEN_IDS[SAY_HI_PROMPT_TOKEN_IDS.len() - 1..],
        (SAY_HI_PROMPT_TOKEN_IDS.len() - 1) as u32,
        &mut request_decoder_state,
    )?;
    let first_token_id = qwen3_5_model.greedy_token_id(&final_position_logits)?;
    eprintln!(
        "[direct-cache-proof] status=progress decode={decode_label} phase=paged_decode_done elapsed_ms={:.2} greedy_token_id={first_token_id}",
        decode_started_at.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

async fn run_same_expert_cache_proof() {
    run_selected_experts_cache_proof("single", vec![42]).await;
}

async fn run_top_8_expert_cache_proof() {
    run_selected_experts_cache_proof("top_8", (0usize..8).collect()).await;
}

async fn run_cross_layer_one_expert_cache_proof() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (runtime, _config, expert_pager) =
        super::construct_model_artifact_expert_pager("[cross-layer-one-expert-cache]").await;
    let routed_layer_and_expert_ids = [(0usize, 42usize), (1usize, 7usize)];

    for (layer_index, expert_id) in routed_layer_and_expert_ids {
        let selected_expert_indices = runtime
            .array_from_i32(&[expert_id as i32], &[1])
            .expect("the layer-qualified route should be valid");
        let (_, cold_request_report) = expert_pager
            .prepare_native_expert_snapshot(&runtime, layer_index, &selected_expert_indices, true)
            .expect("a routed expert should populate exactly one layer-qualified cache entry");
        assert_eq!(cold_request_report.cache_miss_count(), 1);
        assert_eq!(cold_request_report.disk_page_load_count(), 1);
    }

    assert_eq!(
        expert_pager
            .native_expert_cache_statistics()
            .resident_expert_count(),
        routed_layer_and_expert_ids.len() as u64,
        "only the two explicitly routed layer-qualified experts should be retained"
    );

    for (layer_index, expert_id) in routed_layer_and_expert_ids {
        let selected_expert_indices = runtime
            .array_from_i32(&[expert_id as i32], &[1])
            .expect("the repeated layer-qualified route should be valid");
        let (_, warm_request_report) = expert_pager
            .prepare_native_expert_snapshot(&runtime, layer_index, &selected_expert_indices, true)
            .expect("each routed layer-qualified expert should remain independently reusable");
        assert_eq!(warm_request_report.cache_hit_count(), 1);
        assert_eq!(warm_request_report.disk_page_load_count(), 0);
    }
}

async fn run_selected_experts_cache_proof(
    cache_case_label: &'static str,
    selected_expert_ids: Vec<usize>,
) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let started_at = Instant::now();
    eprintln!(
        "[expert-cache-proof] status=start case={cache_case_label} phase=artifact_validation"
    );
    let (runtime, _config, expert_pager) =
        super::construct_model_artifact_expert_pager("[expert-cache-proof]").await;

    let layer_index = 0usize;
    let expected_disk_page_load_count = selected_expert_ids.len();
    let selected_expert_indices = runtime
        .array_from_i32(
            &selected_expert_ids
                .iter()
                .map(|expert_id| *expert_id as i32)
                .collect::<Vec<_>>(),
            &[1, selected_expert_ids.len() as i32],
        )
        .expect("the selected route should be valid");

    eprintln!(
        "[expert-cache-proof] status=progress case={cache_case_label} phase=cold_selected_experts_load layer_index={layer_index} selected_expert_ids={selected_expert_ids:?}"
    );
    let cold_load_start = Instant::now();
    let (_cold_snapshot, cold_report) = expert_pager
        .prepare_native_expert_snapshot(&runtime, layer_index, &selected_expert_indices, true)
        .expect("cold expert request should load one expert from disk and cache it");
    let cold_elapsed = cold_load_start.elapsed();
    eprintln!(
        "[expert-cache-proof] status=progress case={cache_case_label} phase=cold_done elapsed_ms={:.2} disk_page_loads={} disk_batch_loads={} cache_hits={} cache_misses={} payload_bytes={}",
        cold_elapsed.as_secs_f64() * 1000.0,
        cold_report.disk_page_load_count(),
        cold_report.disk_batch_load_count(),
        cold_report.cache_hit_count(),
        cold_report.cache_miss_count(),
        cold_report.successful_source_read_byte_count()
    );
    assert_eq!(
        cold_report.disk_page_load_count(),
        expected_disk_page_load_count as u64,
        "the first request should perform one disk page load per selected uncached expert"
    );
    assert_eq!(
        cold_report.cache_hit_count(),
        0,
        "the first request should not report a cache hit"
    );
    assert_eq!(
        cold_report.cache_miss_count(),
        expected_disk_page_load_count as u64,
        "the first request should report one cache miss per selected expert"
    );
    assert!(
        cold_report.disk_batch_load_count() > 0,
        "the cold request should issue at least one native range-read batch"
    );

    eprintln!(
        "[expert-cache-proof] status=progress case={cache_case_label} phase=warm_selected_experts_load layer_index={layer_index} selected_expert_ids={selected_expert_ids:?}"
    );
    let warm_load_start = Instant::now();
    let (_warm_snapshot, warm_report) = expert_pager
        .prepare_native_expert_snapshot(&runtime, layer_index, &selected_expert_indices, true)
        .expect("warm expert request should use the in-memory cache");
    let warm_elapsed = warm_load_start.elapsed();
    eprintln!(
        "[expert-cache-proof] status=progress case={cache_case_label} phase=warm_done elapsed_ms={:.2} disk_page_loads={} disk_batch_loads={} cache_hits={} cache_misses={} payload_bytes={}",
        warm_elapsed.as_secs_f64() * 1000.0,
        warm_report.disk_page_load_count(),
        warm_report.disk_batch_load_count(),
        warm_report.cache_hit_count(),
        warm_report.cache_miss_count(),
        warm_report.successful_source_read_byte_count()
    );
    assert_eq!(
        warm_report.disk_page_load_count(),
        0,
        "the second request for the same selected experts must not perform another disk page load"
    );
    assert_eq!(
        warm_report.disk_batch_load_count(),
        0,
        "the warm request should not perform a disk batch load"
    );
    assert_eq!(
        warm_report.cache_hit_count(),
        expected_disk_page_load_count as u64,
        "the second request should report one cache hit per selected expert"
    );
    assert_eq!(
        warm_report.cache_miss_count(),
        0,
        "the second request should report zero cache misses"
    );

    let cache_statistics = expert_pager.native_expert_cache_statistics();
    eprintln!(
        "[expert-cache-proof] status=success case={cache_case_label} elapsed_seconds={:.2} cache_entries={} resident_payload_bytes={} total_disk_page_loads={} total_cache_hits={} total_cache_misses={}",
        started_at.elapsed().as_secs_f64(),
        cache_statistics.resident_expert_count(),
        cache_statistics.resident_payload_byte_count(),
        cache_statistics.disk_page_load_count(),
        cache_statistics.cache_hit_count(),
        cache_statistics.cache_miss_count()
    );
    assert_eq!(
        cache_statistics.resident_expert_count(),
        expected_disk_page_load_count as u64,
        "the cache should hold exactly one independent entry per selected expert"
    );
    assert_eq!(
        cache_statistics.disk_page_load_count(),
        expected_disk_page_load_count as u64,
        "only the cold request should have touched disk"
    );
    assert_eq!(
        cache_statistics.resident_payload_byte_count(),
        cold_report.successful_source_read_byte_count(),
        "resident payload bytes should match the selected native source ranges"
    );
}

async fn require_expert_paging_test_completion(test_future: impl Future<Output = ()>) {
    let started_at = Instant::now();
    let timeout_deadline = sleep(EXPERT_PAGING_TEST_TIMEOUT);
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tokio::pin!(test_future);
    tokio::pin!(timeout_deadline);
    progress_interval.tick().await;
    eprintln!(
        "[expert-cache-proof] status=timeout_guard_started timeout_seconds={}",
        EXPERT_PAGING_TEST_TIMEOUT.as_secs()
    );

    loop {
        tokio::select! {
            () = &mut test_future => {
                eprintln!(
                    "[expert-cache-proof] status=completed elapsed_seconds={:.1}",
                    started_at.elapsed().as_secs_f64()
                );
                return;
            }
            () = &mut timeout_deadline => {
                panic!("the expert paging cache proof exceeded {} seconds", EXPERT_PAGING_TEST_TIMEOUT.as_secs());
            }
            _ = progress_interval.tick() => {
                let elapsed = started_at.elapsed();
                let remaining = EXPERT_PAGING_TEST_TIMEOUT.saturating_sub(elapsed);
                eprintln!(
                    "[expert-cache-proof] status=running elapsed_seconds={:.0} ETA<={:.0}",
                    elapsed.as_secs_f64(),
                    remaining.as_secs_f64()
                );
            }
        }
    }
}

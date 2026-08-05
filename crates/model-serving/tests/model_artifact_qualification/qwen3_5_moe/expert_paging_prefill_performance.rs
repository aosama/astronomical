//! Direct automatic sparse-expert prefill throughput probes.

use std::time::Instant;

use astronomical_model_serving::Qwen3_5MoEPagedPrefillExecutionMode;

pub(crate) use super::exact_model_prompt::prepare_reproduced_long_prompt_token_ids_for_model;
use super::expert_paging_prefill::{
    maximum_absolute_difference, prefill_tokens_per_second, prepare_reproduced_prompt_token_ids,
    require_prefill_comparison_completion, run_prefill_snapshot,
};

const MAXIMUM_EXPECTED_TOKEN_LOCAL_TO_COMPACT_ABSOLUTE_LOGIT_DELTA: f32 = 1.0;
const PROMPT_PROCESSING_TPS_PROMPT_TOKEN_COUNT: usize = 2_048;

#[tokio::test]
#[ignore = "loads the full Ornith model twice to compare token-local and compact paged prefill speed"]
async fn should_report_compact_multi_token_prefill_speed_against_token_local_fallback() {
    require_prefill_comparison_completion(run_compact_prefill_performance_probe()).await;
}

#[tokio::test]
#[ignore = "loads the full Ornith model once to measure 2048-token automatic prefill speed"]
async fn should_report_automatic_prefill_tps_for_2048_prompt_tokens() {
    require_prefill_comparison_completion(run_automatic_prefill_2048_prompt_tps_probe()).await;
}

async fn run_compact_prefill_performance_probe() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_started_at = Instant::now();
    let prompt_token_ids = prepare_reproduced_prompt_token_ids();
    eprintln!(
        "[paged-prefill-performance] status=progress phase=prompt_prepared token_count={}",
        prompt_token_ids.len()
    );

    let token_local_prefill_snapshot = run_prefill_snapshot(
        &prompt_token_ids,
        "paged_token_local_performance",
        test_started_at,
        Some(Qwen3_5MoEPagedPrefillExecutionMode::TokenLocalDiagnostic),
    )
    .await;
    let compact_prefill_snapshot = run_prefill_snapshot(
        &prompt_token_ids,
        "paged_compact_performance",
        test_started_at,
        Some(Qwen3_5MoEPagedPrefillExecutionMode::CompactMultiTokenDiagnostic),
    )
    .await;
    let maximum_token_local_to_compact_absolute_logit_delta = maximum_absolute_difference(
        &token_local_prefill_snapshot.final_position_logits,
        &compact_prefill_snapshot.final_position_logits,
    );
    eprintln!(
        "[paged-prefill-performance] status=success token_count={} token_local_prefill_elapsed_seconds={:.3} token_local_prefill_tokens_per_second={:.2} compact_prefill_elapsed_seconds={:.3} compact_prefill_tokens_per_second={:.2} token_local_expert_weight_memory_cache_disk_page_loads={} compact_expert_weight_memory_cache_disk_page_loads={} token_local_expert_weight_memory_cache_disk_batch_loads={} compact_expert_weight_memory_cache_disk_batch_loads={} token_local_expert_weight_memory_cache_hits={} compact_expert_weight_memory_cache_hits={} token_local_expert_weight_memory_cache_misses={} compact_expert_weight_memory_cache_misses={} token_local_expert_weight_memory_cache_resident_payload_bytes={} compact_expert_weight_memory_cache_resident_payload_bytes={} max_abs_logit_delta={:.6}",
        prompt_token_ids.len(),
        token_local_prefill_snapshot.prefill_elapsed.as_secs_f64(),
        prefill_tokens_per_second(
            prompt_token_ids.len(),
            token_local_prefill_snapshot.prefill_elapsed,
        ),
        compact_prefill_snapshot.prefill_elapsed.as_secs_f64(),
        prefill_tokens_per_second(
            prompt_token_ids.len(),
            compact_prefill_snapshot.prefill_elapsed
        ),
        token_local_prefill_snapshot.expert_weight_memory_cache_disk_page_load_count,
        compact_prefill_snapshot.expert_weight_memory_cache_disk_page_load_count,
        token_local_prefill_snapshot.expert_weight_memory_cache_disk_batch_load_count,
        compact_prefill_snapshot.expert_weight_memory_cache_disk_batch_load_count,
        token_local_prefill_snapshot.expert_weight_memory_cache_hit_count,
        compact_prefill_snapshot.expert_weight_memory_cache_hit_count,
        token_local_prefill_snapshot.expert_weight_memory_cache_miss_count,
        compact_prefill_snapshot.expert_weight_memory_cache_miss_count,
        token_local_prefill_snapshot.expert_weight_memory_cache_resident_payload_byte_count,
        compact_prefill_snapshot.expert_weight_memory_cache_resident_payload_byte_count,
        maximum_token_local_to_compact_absolute_logit_delta,
    );
    assert_eq!(
        compact_prefill_snapshot.greedy_token_id, token_local_prefill_snapshot.greedy_token_id,
        "compact multi-token prefill changed the token-local greedy token"
    );
    assert!(
        maximum_token_local_to_compact_absolute_logit_delta
            <= MAXIMUM_EXPECTED_TOKEN_LOCAL_TO_COMPACT_ABSOLUTE_LOGIT_DELTA,
        "compact multi-token prefill changed token-local logits: max_abs_logit_delta={maximum_token_local_to_compact_absolute_logit_delta:.6}"
    );
}

async fn run_automatic_prefill_2048_prompt_tps_probe() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_started_at = Instant::now();
    let prompt_token_ids = prepare_reproduced_2048_prompt_token_ids();
    assert_eq!(
        prompt_token_ids.len(),
        PROMPT_PROCESSING_TPS_PROMPT_TOKEN_COUNT,
        "the prompt-processing TPS probe must exercise exactly 2048 prompt tokens"
    );
    eprintln!(
        "[paged-prefill-performance] status=progress phase=prompt_2048_prepared token_count={}",
        prompt_token_ids.len()
    );

    let automatic_prefill_snapshot = run_prefill_snapshot(
        &prompt_token_ids,
        "automatic_production_2048_prompt",
        test_started_at,
        None,
    )
    .await;
    let prompt_processing_prefill_tokens_per_second = prefill_tokens_per_second(
        prompt_token_ids.len(),
        automatic_prefill_snapshot.prefill_elapsed,
    );
    eprintln!(
        "[paged-prefill-performance] status=success token_count={} prompt_processing_prefill_elapsed_seconds={:.3} prompt_processing_prefill_tokens_per_second={:.2} expert_weight_memory_cache_disk_page_loads={} expert_weight_memory_cache_disk_batch_loads={} expert_weight_memory_cache_hits={} expert_weight_memory_cache_misses={} expert_weight_memory_cache_resident_payload_bytes={}",
        prompt_token_ids.len(),
        automatic_prefill_snapshot.prefill_elapsed.as_secs_f64(),
        prompt_processing_prefill_tokens_per_second,
        automatic_prefill_snapshot.expert_weight_memory_cache_disk_page_load_count,
        automatic_prefill_snapshot.expert_weight_memory_cache_disk_batch_load_count,
        automatic_prefill_snapshot.expert_weight_memory_cache_hit_count,
        automatic_prefill_snapshot.expert_weight_memory_cache_miss_count,
        automatic_prefill_snapshot.expert_weight_memory_cache_resident_payload_byte_count,
    );
    assert!(
        prompt_processing_prefill_tokens_per_second.is_finite()
            && prompt_processing_prefill_tokens_per_second > 0.0,
        "the prompt-processing TPS measurement should be finite and positive"
    );
}

fn prepare_reproduced_2048_prompt_token_ids() -> Vec<u32> {
    prepare_reproduced_long_prompt_token_ids(PROMPT_PROCESSING_TPS_PROMPT_TOKEN_COUNT, 512)
}

pub(crate) fn prepare_reproduced_long_prompt_token_ids(
    prompt_token_count: usize,
    maximum_output_token_count: u16,
) -> Vec<u32> {
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    prepare_reproduced_long_prompt_token_ids_for_model(
        &model_directory,
        "Ornith-1.0-35B-OptiQ-4bit",
        prompt_token_count,
        maximum_output_token_count,
    )
    .expect("the reproduced Ornith prompt should prepare at the exact requested length")
}

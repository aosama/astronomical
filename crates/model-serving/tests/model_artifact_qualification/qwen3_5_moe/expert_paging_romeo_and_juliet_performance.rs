use std::time::{Duration, Instant};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{ExpertWeightMemoryCacheStatistics, Qwen3_5Config, Qwen3_5Model};

use super::expert_paging_decode::{
    load_paged_qwen3_5_model_for_decode_probe, require_expert_paging_decode_completion,
};
use super::speculative_prefill_qualification_support::prepare_romeo_and_juliet_three_paragraph_summary_prompt;

const MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
const ROUTE_WARMUP_PROMPT_TOKEN_COUNT: usize = 2_048;
const MEASURED_PROMPT_TOKEN_COUNT: usize = 31_913;
const PREFILL_CHUNCK_TOKEN_COUNT: usize = 2_048;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 32;
const RECORDED_BRANCH_BASELINE_TOKENS_PER_SECOND: f64 = 548.0;

#[tokio::test]
#[ignore = "loads Ornith and measures a warmed 31913-token Romeo and Juliet production prefill"]
async fn should_measure_warm_31913_token_romeo_and_juliet_prompt() {
    require_expert_paging_decode_completion(
        run_romeo_and_juliet_production_prefill_journey(),
        "[paged-romeo-31913]",
        "the warmed 31913-token Romeo and Juliet production prefill journey",
    )
    .await;
}

async fn run_romeo_and_juliet_production_prefill_journey() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_started_at = Instant::now();
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    eprintln!(
        "[paged-romeo-31913] status=progress phase=prepare_prompts measured_prompt_tokens={MEASURED_PROMPT_TOKEN_COUNT} ETA_seconds=120"
    );
    let warmup_prompt = prepare_romeo_and_juliet_three_paragraph_summary_prompt(
        &model_directory,
        MODEL_ID,
        RequestId::new(96_100),
        ROUTE_WARMUP_PROMPT_TOKEN_COUNT,
        MAXIMUM_OUTPUT_TOKEN_COUNT,
    );
    let measured_prompt = prepare_romeo_and_juliet_three_paragraph_summary_prompt(
        &model_directory,
        MODEL_ID,
        RequestId::new(96_101),
        MEASURED_PROMPT_TOKEN_COUNT,
        MAXIMUM_OUTPUT_TOKEN_COUNT,
    );
    let (qwen3_5_model, config) =
        load_paged_qwen3_5_model_for_decode_probe("[paged-romeo-31913]").await;

    let warmup_measurement = run_prefill_pass(
        &qwen3_5_model,
        &config,
        &warmup_prompt.prompt_token_ids,
        "route_warmup",
    );
    let measured_prefill = run_prefill_pass(
        &qwen3_5_model,
        &config,
        &measured_prompt.prompt_token_ids,
        "measured_romeo_and_juliet",
    );
    let parity_replay = run_prefill_pass(
        &qwen3_5_model,
        &config,
        &warmup_prompt.prompt_token_ids,
        "warmup_parity_replay",
    );

    let measured_tokens_per_second = measured_prefill.tokens_per_second();
    let baseline_speedup_ratio =
        measured_tokens_per_second / RECORDED_BRANCH_BASELINE_TOKENS_PER_SECOND;
    eprintln!(
        "[paged-romeo-31913] status=success total_elapsed_seconds={:.3} prompt_tokens={} prefill_elapsed_seconds={:.3} prefill_tokens_per_second={measured_tokens_per_second:.2} recorded_branch_baseline_tokens_per_second={RECORDED_BRANCH_BASELINE_TOKENS_PER_SECOND:.2} baseline_speedup_ratio={baseline_speedup_ratio:.3} greedy_token_id={} disk_page_loads={} disk_batch_loads={} expert_entries={} resident_payload_bytes={}",
        test_started_at.elapsed().as_secs_f64(),
        measured_prefill.processed_token_count,
        measured_prefill.elapsed.as_secs_f64(),
        measured_prefill.greedy_token_id,
        measured_prefill.disk_page_load_count_delta,
        measured_prefill.disk_batch_load_count_delta,
        measured_prefill.final_cache_statistics.entry_count,
        measured_prefill
            .final_cache_statistics
            .resident_payload_byte_count,
    );

    assert_eq!(
        parity_replay.greedy_token_id, warmup_measurement.greedy_token_id,
        "the same greedy prompt must remain exact after the long paged prefill"
    );
    assert!(measured_tokens_per_second.is_finite() && measured_tokens_per_second > 0.0);
    assert!(
        baseline_speedup_ratio >= 2.0,
        "the warmed production page-table path must process the 31,913-token prompt at least twice as fast as the recorded 548 tokens/second branch baseline; measured {measured_tokens_per_second:.2} tokens/second ({baseline_speedup_ratio:.3}x)"
    );
    assert!(
        measured_prefill
            .final_cache_statistics
            .resident_payload_byte_count
            <= measured_prefill
                .final_cache_statistics
                .maximum_resident_payload_byte_count
    );
}

fn run_prefill_pass(
    qwen3_5_model: &Qwen3_5Model,
    config: &Qwen3_5Config,
    prompt_token_ids: &[u32],
    pass_label: &str,
) -> PrefillPassMeasurement {
    let initial_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    let mut request_decoder_state = crate::common::standard_request_decoder_state(config);
    let prefill_started_at = Instant::now();
    let prefix_token_count = prompt_token_ids.len() - 1;
    let mut processed_prefix_token_count = 0usize;
    while processed_prefix_token_count < prefix_token_count {
        let next_prefix_token_count = processed_prefix_token_count
            .saturating_add(PREFILL_CHUNCK_TOKEN_COUNT)
            .min(prefix_token_count);
        qwen3_5_model
            .prefill_chunck(
                &prompt_token_ids[processed_prefix_token_count..next_prefix_token_count],
                processed_prefix_token_count as u32,
                &mut request_decoder_state,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{pass_label} prefill failed at token {processed_prefix_token_count}: {error}"
                )
            });
        qwen3_5_model
            .runtime()
            .synchronize_gpu_stream_and_reclaim_allocator_cache_above_threshold(
                astronomical_runtime_integration::ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{pass_label} allocator cleanup failed after token {next_prefix_token_count}: {error}"
                )
            });
        processed_prefix_token_count = next_prefix_token_count;
        let elapsed = prefill_started_at.elapsed();
        let processed_tokens_per_second =
            processed_prefix_token_count as f64 / elapsed.as_secs_f64();
        let remaining_token_count = prompt_token_ids.len() - processed_prefix_token_count;
        let estimated_remaining_seconds =
            remaining_token_count as f64 / processed_tokens_per_second.max(f64::EPSILON);
        eprintln!(
            "[paged-romeo-31913] status=progress pass={pass_label} processed_tokens={processed_prefix_token_count}/{} prefill_tokens_per_second={processed_tokens_per_second:.2} ETA_seconds={estimated_remaining_seconds:.1}",
            prompt_token_ids.len(),
        );
    }
    let final_logits = qwen3_5_model
        .forward_chunk(
            &prompt_token_ids[prefix_token_count..],
            prefix_token_count as u32,
            &mut request_decoder_state,
        )
        .unwrap_or_else(|error| panic!("{pass_label} final prompt token failed: {error}"));
    let greedy_token_id = qwen3_5_model
        .greedy_token_id(&final_logits)
        .unwrap_or_else(|error| panic!("{pass_label} final logits should select a token: {error}"));
    let elapsed = prefill_started_at.elapsed();
    let final_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    PrefillPassMeasurement {
        processed_token_count: prompt_token_ids.len(),
        elapsed,
        greedy_token_id,
        disk_page_load_count_delta: final_cache_statistics
            .disk_page_load_count
            .saturating_sub(initial_cache_statistics.disk_page_load_count),
        disk_batch_load_count_delta: final_cache_statistics
            .disk_batch_load_count
            .saturating_sub(initial_cache_statistics.disk_batch_load_count),
        final_cache_statistics,
    }
}

struct PrefillPassMeasurement {
    processed_token_count: usize,
    elapsed: Duration,
    greedy_token_id: u32,
    disk_page_load_count_delta: u64,
    disk_batch_load_count_delta: u64,
    final_cache_statistics: ExpertWeightMemoryCacheStatistics,
}

impl PrefillPassMeasurement {
    fn tokens_per_second(&self) -> f64 {
        self.processed_token_count as f64 / self.elapsed.as_secs_f64()
    }
}

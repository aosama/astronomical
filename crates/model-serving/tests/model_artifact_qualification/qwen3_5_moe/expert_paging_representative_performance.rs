//! Representative direct-model performance probe with substantial context and output.

use std::time::Instant;

use astronomical_model_serving::{Qwen3_5Config, Qwen3_5Model, RequestDecoderStateStack};

use super::expert_paging_decode::{
    bytes_to_gib, load_paged_qwen3_5_model_for_decode_probe,
    require_expert_paging_decode_completion,
};
use super::expert_paging_prefill_performance::prepare_reproduced_long_prompt_token_ids;

pub(crate) const REPRESENTATIVE_INPUT_TOKEN_COUNT: usize = 1_024;
pub(crate) const REPRESENTATIVE_OUTPUT_TOKEN_COUNT: u32 = 512;
const OUTPUT_PROGRESS_INTERVAL_TOKEN_COUNT: u32 = 25;
const OUTPUT_TOKEN_ID_CHECKSUM_OFFSET: u64 = 1_469_598_103_934_665_603;
const OUTPUT_TOKEN_ID_CHECKSUM_MULTIPLIER: u64 = 1_099_511_628_211;

#[tokio::test]
#[ignore = "loads the full Ornith model with 1024 input tokens and generates 512 output tokens"]
async fn should_measure_1024_input_tokens_and_512_output_tokens() {
    require_expert_paging_decode_completion(
        run_representative_performance_probe(),
        "[paged-representative]",
        "the 1024-input-token and 512-output-token representative performance probe",
    )
    .await;
}

async fn run_representative_performance_probe() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_started_at = Instant::now();
    let prompt_token_ids = prepare_reproduced_long_prompt_token_ids(
        REPRESENTATIVE_INPUT_TOKEN_COUNT,
        REPRESENTATIVE_OUTPUT_TOKEN_COUNT as u16,
    );
    let (qwen3_5_model, config) =
        load_paged_qwen3_5_model_for_decode_probe("[paged-representative]").await;
    run_representative_performance_probe_for_loaded_model(
        &qwen3_5_model,
        &config,
        &prompt_token_ids,
        "[paged-representative]",
        test_started_at,
    );
}

pub(crate) fn run_representative_performance_probe_for_loaded_model(
    qwen3_5_model: &Qwen3_5Model,
    config: &Qwen3_5Config,
    prompt_token_ids: &[u32],
    progress_log_prefix: &str,
    test_started_at: Instant,
) {
    assert_eq!(prompt_token_ids.len(), REPRESENTATIVE_INPUT_TOKEN_COUNT);
    eprintln!(
        "{progress_log_prefix} status=progress phase=prompt_prepared input_tokens={} requested_output_tokens={}",
        prompt_token_ids.len(),
        REPRESENTATIVE_OUTPUT_TOKEN_COUNT
    );
    let mut request_decoder_state = RequestDecoderStateStack::empty_from_config(config);
    let prefill_chunck_token_count = prompt_token_ids.len() - 1;
    eprintln!(
        "{progress_log_prefix} status=progress phase=prefill prefill_chunck_tokens={prefill_chunck_token_count}"
    );
    let prefill_started_at = Instant::now();
    qwen3_5_model
        .prefill_chunck(
            &prompt_token_ids[..prefill_chunck_token_count],
            0,
            &mut request_decoder_state,
        )
        .expect("the representative prompt prefix should materialize decoder state");
    let prefill_elapsed = prefill_started_at.elapsed();
    eprintln!(
        "{progress_log_prefix} status=progress phase=prefill_done elapsed_seconds={:.3} prefill_tokens_per_second={:.2}",
        prefill_elapsed.as_secs_f64(),
        prefill_chunck_token_count as f64 / prefill_elapsed.as_secs_f64()
    );

    let initial_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    let mut current_token_id = prompt_token_ids[prefill_chunck_token_count];
    let mut generated_token_id_checksum = OUTPUT_TOKEN_ID_CHECKSUM_OFFSET;
    let generation_started_at = Instant::now();
    for output_token_index in 0..REPRESENTATIVE_OUTPUT_TOKEN_COUNT {
        let position_tokens = (prefill_chunck_token_count as u32)
            .checked_add(output_token_index)
            .expect("the representative decode position should fit u32");
        let logits = qwen3_5_model
            .forward_chunk(
                &[current_token_id],
                position_tokens,
                &mut request_decoder_state,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "representative paged decode output token {} failed: {error}",
                    output_token_index + 1
                )
            });
        current_token_id = qwen3_5_model
            .greedy_token_id(&logits)
            .unwrap_or_else(|error| {
                panic!(
                    "representative greedy token selection {} failed: {error}",
                    output_token_index + 1
                )
            });
        generated_token_id_checksum ^= u64::from(current_token_id);
        generated_token_id_checksum =
            generated_token_id_checksum.wrapping_mul(OUTPUT_TOKEN_ID_CHECKSUM_MULTIPLIER);

        let completed_output_token_count = output_token_index + 1;
        if completed_output_token_count == 1
            || completed_output_token_count % OUTPUT_PROGRESS_INTERVAL_TOKEN_COUNT == 0
            || completed_output_token_count == REPRESENTATIVE_OUTPUT_TOKEN_COUNT
        {
            let generation_elapsed = generation_started_at.elapsed();
            let average_output_tokens_per_second =
                f64::from(completed_output_token_count) / generation_elapsed.as_secs_f64();
            let remaining_output_token_count =
                REPRESENTATIVE_OUTPUT_TOKEN_COUNT - completed_output_token_count;
            let estimated_remaining_seconds =
                f64::from(remaining_output_token_count) / average_output_tokens_per_second;
            let cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
            let mlx_memory_snapshot = qwen3_5_model
                .runtime()
                .memory_snapshot()
                .expect("the representative probe should sample MLX memory at progress boundaries");
            eprintln!(
                "{progress_log_prefix} status=progress output_tokens={completed_output_token_count:03}/{REPRESENTATIVE_OUTPUT_TOKEN_COUNT} average_output_tokens_per_second={average_output_tokens_per_second:.2} ETA_seconds={estimated_remaining_seconds:.1} cache_entries={} complete_layers={} cache_evictions={} resident_payload_gib={:.2} maximum_resident_payload_gib={:.2} mlx_active_gib={:.2} mlx_allocator_gib={:.2} mlx_peak_gib={:.2}",
                cache_statistics.entry_count,
                cache_statistics.complete_layer_count,
                cache_statistics.eviction_count,
                bytes_to_gib(cache_statistics.resident_payload_byte_count),
                bytes_to_gib(cache_statistics.maximum_resident_payload_byte_count),
                bytes_to_gib(mlx_memory_snapshot.active_memory_bytes() as u64),
                bytes_to_gib(mlx_memory_snapshot.allocator_cache_memory_bytes() as u64),
                bytes_to_gib(mlx_memory_snapshot.peak_memory_bytes() as u64),
            );
        }
    }

    let generation_elapsed = generation_started_at.elapsed();
    let request_execution_elapsed = prefill_elapsed + generation_elapsed;
    let final_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    let final_mlx_memory_snapshot = qwen3_5_model
        .runtime()
        .memory_snapshot()
        .expect("the representative probe should sample final MLX memory");
    let average_output_tokens_per_second =
        f64::from(REPRESENTATIVE_OUTPUT_TOKEN_COUNT) / generation_elapsed.as_secs_f64();
    let end_to_end_output_tokens_per_second =
        f64::from(REPRESENTATIVE_OUTPUT_TOKEN_COUNT) / request_execution_elapsed.as_secs_f64();
    eprintln!(
        "{progress_log_prefix} status=success total_elapsed_seconds={:.3} input_tokens={} output_tokens={} output_token_id_checksum={} prefill_elapsed_seconds={:.3} generation_elapsed_seconds={:.3} average_output_tokens_per_second={average_output_tokens_per_second:.2} end_to_end_output_tokens_per_second={end_to_end_output_tokens_per_second:.2} cache_entries={} complete_layers={} cache_evictions={} resident_payload_gib={:.2} maximum_resident_payload_gib={:.2} disk_page_loads={} disk_batch_loads={} cache_hits={} cache_misses={} mlx_active_bytes={} mlx_allocator_bytes={} mlx_peak_bytes={} mlx_active_gib={:.2} mlx_allocator_gib={:.2} mlx_peak_gib={:.2}",
        test_started_at.elapsed().as_secs_f64(),
        REPRESENTATIVE_INPUT_TOKEN_COUNT,
        REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
        generated_token_id_checksum,
        prefill_elapsed.as_secs_f64(),
        generation_elapsed.as_secs_f64(),
        final_cache_statistics.entry_count,
        final_cache_statistics.complete_layer_count,
        final_cache_statistics.eviction_count,
        bytes_to_gib(final_cache_statistics.resident_payload_byte_count),
        bytes_to_gib(final_cache_statistics.maximum_resident_payload_byte_count),
        final_cache_statistics
            .disk_page_load_count
            .saturating_sub(initial_cache_statistics.disk_page_load_count),
        final_cache_statistics
            .disk_batch_load_count
            .saturating_sub(initial_cache_statistics.disk_batch_load_count),
        final_cache_statistics
            .cache_hit_count
            .saturating_sub(initial_cache_statistics.cache_hit_count),
        final_cache_statistics
            .cache_miss_count
            .saturating_sub(initial_cache_statistics.cache_miss_count),
        final_mlx_memory_snapshot.active_memory_bytes(),
        final_mlx_memory_snapshot.allocator_cache_memory_bytes(),
        final_mlx_memory_snapshot.peak_memory_bytes(),
        bytes_to_gib(final_mlx_memory_snapshot.active_memory_bytes() as u64),
        bytes_to_gib(final_mlx_memory_snapshot.allocator_cache_memory_bytes() as u64),
        bytes_to_gib(final_mlx_memory_snapshot.peak_memory_bytes() as u64),
    );
    assert!(average_output_tokens_per_second.is_finite());
    assert!(average_output_tokens_per_second > 0.0);
    assert_ne!(
        final_cache_statistics.maximum_resident_payload_byte_count,
        u64::MAX,
        "representative paging must derive a finite automatic retention ceiling"
    );
    assert!(
        final_cache_statistics.resident_payload_byte_count
            <= final_cache_statistics.maximum_resident_payload_byte_count
    );
}

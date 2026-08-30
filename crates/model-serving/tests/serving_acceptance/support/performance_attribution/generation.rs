use super::{IMAGE_PAD_TOKEN_ID, PROGRESS_INTERVAL};
use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, Qwen3_5Engine, Qwen3_5InferenceRequest,
};
use std::time::Instant;
use tokio::time::{MissedTickBehavior, interval};

pub(crate) async fn run_attributed_generation(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    prompt_token_ids: &[u32],
    phase_name: &str,
    output_token_count: u16,
    end_of_sequence_token_ids: &[u32],
) -> Vec<u32> {
    let request_started_at = Instant::now();
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new_sampling(
                request_id,
                prompt_token_ids.to_vec(),
                output_token_count,
                1_000,
                1_000,
                Some(1_234),
            )
            .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID)
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await
        .expect("the benchmark request should be accepted");
    let mut generated_token_ids = Vec::with_capacity(usize::from(output_token_count));
    let mut generation_started_at = None;
    let mut latest_prefill_processed_token_count = 0_u32;
    let mut latest_prefill_elapsed_millis = 0_u64;
    let prompt_token_count = u32::try_from(prompt_token_ids.len()).unwrap_or(u32::MAX);
    let mut progress_interval = interval(PROGRESS_INTERVAL);
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    progress_interval.tick().await;
    while generated_token_ids.len() < usize::from(output_token_count) {
        let generated_token_outcome = crate::common::generation_progress::await_generation_advance_with_live_progress(qwen3_5_engine.decode_next_token(request_id), &mut progress_interval, || {
            if generated_token_ids.is_empty() {
                let prefill_elapsed_seconds =
                    latest_prefill_elapsed_millis as f64 / 1000.0;
                let prefill_tokens_per_second = if latest_prefill_elapsed_millis == 0 {
                    0.0
                } else {
                    f64::from(latest_prefill_processed_token_count)
                        / prefill_elapsed_seconds.max(f64::EPSILON)
                };
                eprintln!(
                    "[performance-attribution] status=progress phase={phase_name} stage=prefill processed_prompt_tokens={latest_prefill_processed_token_count}/{prompt_token_count} prefill_tokens_per_second={prefill_tokens_per_second:.2} request_elapsed_seconds={:.1}",
                    request_started_at.elapsed().as_secs_f64()
                );
                return;
            }
            let steady_state_output_token_count = generated_token_ids.len().saturating_sub(1);
            let generation_elapsed_seconds = generation_started_at.map(|started_at: Instant| started_at.elapsed().as_secs_f64()).unwrap_or(0.0);
            let output_tokens_per_second = steady_state_output_token_count as f64 / generation_elapsed_seconds.max(f64::EPSILON);
            let remaining_output_token_count = usize::from(output_token_count).saturating_sub(generated_token_ids.len());
            let estimated_remaining_seconds = if steady_state_output_token_count == 0 { f64::INFINITY } else { remaining_output_token_count as f64 / output_tokens_per_second.max(f64::EPSILON) };
            eprintln!("[performance-attribution] status=progress phase={phase_name} stage=generation completed_output_tokens={}/{output_token_count} steady_state_output_tokens={steady_state_output_token_count} output_tokens_per_second={output_tokens_per_second:.2} ETA_seconds={estimated_remaining_seconds:.1}", generated_token_ids.len());
        }).await;
        match generated_token_outcome.unwrap_or_else(|generation_advance_error| {
            panic!(
                "phase {phase_name} failed after {} generated tokens: {generation_advance_error}",
                generated_token_ids.len()
            )
        }) {
            GeneratedToken::TokenId {
                token_id: generated_token_id,
                ..
            } => {
                generated_token_ids.push(generated_token_id);
                generation_started_at.get_or_insert_with(Instant::now);
                if end_of_sequence_token_ids.contains(&generated_token_id) {
                    break;
                }
            }
            GeneratedToken::PrefillProgress {
                processed_token_count,
                elapsed_millis,
                ..
            } => {
                latest_prefill_processed_token_count = processed_token_count;
                latest_prefill_elapsed_millis = elapsed_millis;
                let prefill_tokens_per_second = if elapsed_millis == 0 {
                    0.0
                } else {
                    f64::from(processed_token_count)
                        / (elapsed_millis as f64 / 1000.0).max(f64::EPSILON)
                };
                eprintln!(
                    "[performance-attribution] status=progress phase={phase_name} stage=prefill processed_prompt_tokens={processed_token_count}/{prompt_token_count} prefill_tokens_per_second={prefill_tokens_per_second:.2}"
                );
            }
            GeneratedToken::PromptProcessingPhaseStarted {
                prompt_processing_phase,
                total_token_count,
            } => {
                eprintln!(
                    "[performance-attribution] status=progress phase={phase_name} stage=prefill_start prompt_processing_phase={prompt_processing_phase:?} total_prompt_tokens={total_token_count}"
                );
            }
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }
    let steady_state_output_token_count = generated_token_ids.len().saturating_sub(1);
    let generation_elapsed_seconds = generation_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let output_tokens_per_second =
        steady_state_output_token_count as f64 / generation_elapsed_seconds.max(f64::EPSILON);
    eprintln!(
        "[performance-attribution] status=success phase={phase_name} output_tokens={} steady_state_output_tokens={steady_state_output_token_count} generation_elapsed_seconds={generation_elapsed_seconds:.3} output_tokens_per_second={output_tokens_per_second:.3} request_elapsed_seconds={:.3}",
        generated_token_ids.len(),
        request_started_at.elapsed().as_secs_f64()
    );
    generated_token_ids
}

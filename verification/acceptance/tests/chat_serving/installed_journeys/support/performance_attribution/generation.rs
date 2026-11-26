use super::{IMAGE_PAD_TOKEN_ID, PROGRESS_INTERVAL};
use ipc_protocol::RequestId;
use model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, TextInferenceRequest,
};
use std::time::Instant;
use tokio::time::{MissedTickBehavior, interval};

/// How a journey's request picks each token. Shared so a journey can ask another engine
/// the same question with the same draws.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum GenerationSampling {
    /// Temperature 0: the top token, so a changed token means changed logits.
    Greedy,
    /// Temperature 1.0 with a fixed seed, like a default chat request. A seeded draw
    /// flips on logit changes far smaller than greedy ever sees.
    Seeded,
    /// Temperature 0.7 with a fixed seed, the temperature people use for code.
    SeededForCode,
}

impl GenerationSampling {
    pub(crate) const fn temperature_thousandths(self) -> u16 {
        // In declaration order: greedy, seeded, seeded for code.
        const TEMPERATURE_THOUSANDTHS_BY_SAMPLING: [u16; 3] = [0, 1_000, 700];
        TEMPERATURE_THOUSANDTHS_BY_SAMPLING[self as usize]
    }

    pub(crate) const fn seed(self) -> Option<u64> {
        match self {
            Self::Greedy => None,
            Self::Seeded | Self::SeededForCode => Some(1_234),
        }
    }
}

pub(crate) async fn run_attributed_generation(
    qwen3_5_engine: &mut impl InferenceEngine<Request = TextInferenceRequest>,
    request_id: RequestId,
    prompt_token_ids: &[u32],
    phase_name: &str,
    output_token_count: u16,
    end_of_sequence_token_ids: &[u32],
) -> Vec<u32> {
    run_generation_with_sampling(
        qwen3_5_engine,
        request_id,
        prompt_token_ids,
        phase_name,
        output_token_count,
        end_of_sequence_token_ids,
        GenerationSampling::Seeded,
    )
    .await
}

pub(crate) async fn run_generation_with_sampling(
    qwen3_5_engine: &mut impl InferenceEngine<Request = TextInferenceRequest>,
    request_id: RequestId,
    prompt_token_ids: &[u32],
    phase_name: &str,
    output_token_count: u16,
    end_of_sequence_token_ids: &[u32],
    generation_sampling: GenerationSampling,
) -> Vec<u32> {
    run_generation_with_request_attribution(
        qwen3_5_engine,
        request_id,
        prompt_token_ids,
        phase_name,
        output_token_count,
        end_of_sequence_token_ids,
        generation_sampling,
        RequestAttribution::Enabled,
    )
    .await
}

/// Whether a request records a performance-attribution report. Attribution
/// synchronizes the graphics processor after each prefill stage, so a
/// request that must behave like production serving turns it off.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RequestAttribution {
    Enabled,
    Disabled,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_generation_with_request_attribution(
    qwen3_5_engine: &mut impl InferenceEngine<Request = TextInferenceRequest>,
    request_id: RequestId,
    prompt_token_ids: &[u32],
    phase_name: &str,
    output_token_count: u16,
    end_of_sequence_token_ids: &[u32],
    generation_sampling: GenerationSampling,
    request_attribution: RequestAttribution,
) -> Vec<u32> {
    let request_started_at = Instant::now();
    let inference_request = attributed_inference_request(
        request_id,
        prompt_token_ids,
        output_token_count,
        generation_sampling,
        request_attribution,
    );
    qwen3_5_engine
        .start_generation(inference_request)
        .await
        .expect("the benchmark request should be accepted");
    let mut attributed_run = AttributedRun {
        phase_name,
        output_token_count,
        prompt_token_count: u32::try_from(prompt_token_ids.len()).unwrap_or(u32::MAX),
        request_started_at,
        generated_token_ids: Vec::with_capacity(usize::from(output_token_count)),
        generation_started_at: None,
        latest_prefill_processed_token_count: 0,
        latest_prefill_elapsed_millis: 0,
    };
    let mut progress_interval = interval(PROGRESS_INTERVAL);
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    progress_interval.tick().await;
    while attributed_run.generated_token_ids.len() < usize::from(output_token_count) {
        let generated_token_outcome =
            crate::common::generation_progress::await_generation_advance_with_live_progress(
                qwen3_5_engine.decode_next_token(request_id),
                &mut progress_interval,
                || attributed_run.print_live_progress(),
            )
            .await;
        let generated_token = generated_token_outcome.unwrap_or_else(|generation_advance_error| {
            panic!(
                "phase {phase_name} failed after {} generated tokens: {generation_advance_error}",
                attributed_run.generated_token_ids.len()
            )
        });
        if attributed_run.apply_generated_token(generated_token, end_of_sequence_token_ids) {
            break;
        }
    }
    attributed_run.print_success();
    attributed_run.generated_token_ids
}

fn attributed_inference_request(
    request_id: RequestId,
    prompt_token_ids: &[u32],
    output_token_count: u16,
    generation_sampling: GenerationSampling,
    request_attribution: RequestAttribution,
) -> TextInferenceRequest {
    let inference_request = TextInferenceRequest::new_sampling(
        request_id,
        prompt_token_ids.to_vec(),
        output_token_count,
        generation_sampling.temperature_thousandths(),
        1_000,
        generation_sampling.seed(),
    )
    .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID);
    match request_attribution {
        RequestAttribution::Enabled => {
            inference_request.with_performance_attribution(PerformanceAttribution::enabled())
        }
        RequestAttribution::Disabled => inference_request,
    }
}

/// Where one attributed request has got to: the tokens it has written, when
/// the first one arrived, and the newest prefill event. The progress and
/// success lines are all read from here.
struct AttributedRun<'phase> {
    phase_name: &'phase str,
    output_token_count: u16,
    prompt_token_count: u32,
    request_started_at: Instant,
    generated_token_ids: Vec<u32>,
    generation_started_at: Option<Instant>,
    latest_prefill_processed_token_count: u32,
    latest_prefill_elapsed_millis: u64,
}

impl AttributedRun<'_> {
    /// Folds one generated token into the run. Returns whether the request
    /// has finished.
    fn apply_generated_token(
        &mut self,
        generated_token: GeneratedToken,
        end_of_sequence_token_ids: &[u32],
    ) -> bool {
        let phase_name = self.phase_name;
        match generated_token {
            GeneratedToken::TokenId {
                token_id: generated_token_id,
                ..
            } => {
                self.generated_token_ids.push(generated_token_id);
                self.generation_started_at.get_or_insert_with(Instant::now);
                if end_of_sequence_token_ids.contains(&generated_token_id) {
                    return true;
                }
            }
            GeneratedToken::PrefillProgress {
                processed_token_count,
                elapsed_millis,
                ..
            } => {
                self.latest_prefill_processed_token_count = processed_token_count;
                self.latest_prefill_elapsed_millis = elapsed_millis;
                let prefill_tokens_per_second = if elapsed_millis == 0 {
                    0.0
                } else {
                    f64::from(processed_token_count)
                        / (elapsed_millis as f64 / 1000.0).max(f64::EPSILON)
                };
                let prompt_token_count = self.prompt_token_count;
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
            GeneratedToken::EndOfSequence => return true,
        }
        false
    }

    /// The line printed on every progress tick: the prefill's state while no
    /// token has arrived, the generation's afterwards.
    fn print_live_progress(&self) {
        let phase_name = self.phase_name;
        if self.generated_token_ids.is_empty() {
            let prompt_token_count = self.prompt_token_count;
            let latest_prefill_processed_token_count = self.latest_prefill_processed_token_count;
            let prefill_elapsed_seconds = self.latest_prefill_elapsed_millis as f64 / 1000.0;
            let prefill_tokens_per_second = if self.latest_prefill_elapsed_millis == 0 {
                0.0
            } else {
                f64::from(latest_prefill_processed_token_count)
                    / prefill_elapsed_seconds.max(f64::EPSILON)
            };
            eprintln!(
                "[performance-attribution] status=progress phase={phase_name} stage=prefill processed_prompt_tokens={latest_prefill_processed_token_count}/{prompt_token_count} prefill_tokens_per_second={prefill_tokens_per_second:.2} request_elapsed_seconds={:.1}",
                self.request_started_at.elapsed().as_secs_f64()
            );
            return;
        }
        let output_token_count = self.output_token_count;
        let generated_output_token_count = self.generated_token_ids.len();
        let steady_state_output_token_count = generated_output_token_count.saturating_sub(1);
        let generation_elapsed_seconds = self
            .generation_started_at
            .map(|started_at: Instant| started_at.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let output_tokens_per_second =
            steady_state_output_token_count as f64 / generation_elapsed_seconds.max(f64::EPSILON);
        let remaining_output_token_count =
            usize::from(output_token_count).saturating_sub(generated_output_token_count);
        let estimated_remaining_seconds = if steady_state_output_token_count == 0 {
            f64::INFINITY
        } else {
            remaining_output_token_count as f64 / output_tokens_per_second.max(f64::EPSILON)
        };
        eprintln!(
            "[performance-attribution] status=progress phase={phase_name} stage=generation completed_output_tokens={generated_output_token_count}/{output_token_count} steady_state_output_tokens={steady_state_output_token_count} output_tokens_per_second={output_tokens_per_second:.2} ETA_seconds={estimated_remaining_seconds:.1}"
        );
    }

    fn print_success(&self) {
        let phase_name = self.phase_name;
        let steady_state_output_token_count = self.generated_token_ids.len().saturating_sub(1);
        let generation_elapsed_seconds = self
            .generation_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let output_tokens_per_second =
            steady_state_output_token_count as f64 / generation_elapsed_seconds.max(f64::EPSILON);
        eprintln!(
            "[performance-attribution] status=success phase={phase_name} output_tokens={} steady_state_output_tokens={steady_state_output_token_count} generation_elapsed_seconds={generation_elapsed_seconds:.3} output_tokens_per_second={output_tokens_per_second:.3} request_elapsed_seconds={:.3}",
            self.generated_token_ids.len(),
            self.request_started_at.elapsed().as_secs_f64()
        );
    }
}

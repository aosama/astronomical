//! Prompt reading for the owner-thread folded-ternary execution, through the
//! shared Qwen 3.5 steps (issue #2473, Prism step 9).
//!
//! The shared steps choose each chunk (the configured size, clamped at the
//! `astro_ssd_prompt_cache` boundaries), publish the blocks and the partial
//! tail it completes before the cursor moves, and report its progress. What
//! stays here is the folded-ternary forward and its evaluation: a chunk
//! evaluates its final hidden row (and, for the last chunk, the logits) and
//! nothing more, so the decoder state's own arrays are not added as roots.

use std::time::Instant;

use ipc_protocol::RequestId;
use mlx_bridge::MlxArray;

use crate::GeneratedToken;
use crate::InferenceEngineError;
use crate::PerformanceOperation;
use crate::astro_ssd_prompt_cache_boundary_completed_prefill_chunk_tokens;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::{
    AstroSsdPromptCacheBoundaryCheckpoint, AstroSsdPromptCacheBoundaryCheckpointCollector,
};
use text_generation::astro_ssd_prompt_cache_capture::record_astro_ssd_prompt_cache_boundary_checkpoint;
use text_generation::engine_errors::fatal_runtime_error;
use text_generation::generation_loop::prompt_reading_steps::{
    self, PromptStretch, PromptStretchTiming,
};

use super::engine_trait_impl::gpu_error;
use super::{PrismHadamardQwen35ActiveGeneration, PrismHadamardQwen35InferenceExecution};

/// The block boundaries one chunk crosses, relative to its start, and the
/// collector for those before its end; `None` when the chunk ends on its
/// only boundary or crosses none.
struct ChunkBoundaryPlan {
    all_completed_prefill_chunk_tokens: Vec<usize>,
    intermediate_boundary_collector: Option<AstroSsdPromptCacheBoundaryCheckpointCollector>,
}

impl PrismHadamardQwen35InferenceExecution {
    /// Reads the next chunk of the prompt, or `None` once it is all read.
    pub(super) fn advance_prompt_prefill(
        &mut self,
        request_id: RequestId,
        active: &mut PrismHadamardQwen35ActiveGeneration,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        let prompt_token_count = active.input_token_ids.len();
        let Some(prefill_chunk_span) =
            prompt_reading_steps::resolve_next_prefill_chunk_span(self, active, prompt_token_count)
        else {
            return Ok(None);
        };
        let read_stretch = PromptStretch {
            prefill_start: prefill_chunk_span.prefill_start,
            prefill_end: prefill_chunk_span.requested_prefill_chunk_end,
        };
        let forward_chunk_started_at = Instant::now();
        let boundary_checkpoints = self.read_prompt_stretch(active, read_stretch)?;
        let forward_chunk_elapsed_millis =
            u64::try_from(forward_chunk_started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        prompt_reading_steps::publish_and_advance_durable_prefill_state(
            self,
            active,
            read_stretch,
            boundary_checkpoints,
        )?;
        prefill_progress_token(
            self,
            active,
            request_id,
            read_stretch,
            PromptStretchTiming {
                forward_chunk_started_at,
                forward_chunk_elapsed_millis,
            },
        )
        .map(Some)
    }

    /// Runs one chunk's forward from the request's position, evaluates it,
    /// samples the first reply token when the chunk ends the prompt, and
    /// returns the recurrent state at every block boundary the chunk
    /// completes. The position and cursor move only once the shared step has
    /// published those blocks.
    fn read_prompt_stretch(
        &self,
        active: &mut PrismHadamardQwen35ActiveGeneration,
        read_stretch: PromptStretch,
    ) -> Result<Vec<AstroSsdPromptCacheBoundaryCheckpoint>, InferenceEngineError> {
        let model = self.execution_core.loaded_model()?;
        let ChunkBoundaryPlan {
            all_completed_prefill_chunk_tokens,
            mut intermediate_boundary_collector,
        } = self.chunk_boundary_plan(model, active, read_stretch)?;
        let chunk_token_ids: Vec<i32> = active.input_token_ids
            [read_stretch.prefill_start..read_stretch.prefill_end]
            .iter()
            .map(|token_id| *token_id as i32)
            .collect();
        let token_indices = model
            .runtime
            .array_from_i32(&chunk_token_ids, &[1, chunk_token_ids.len() as i32])
            .map_err(gpu_error)?;
        let hidden_states = model
            .forward_hidden_states_recording_boundary_states(
                &token_indices,
                active.next_position_tokens,
                &mut active.request_decoder_state,
                intermediate_boundary_collector.as_mut(),
            )
            .map_err(gpu_error)?;
        let chunk_ends_the_prompt = read_stretch.prefill_end == active.input_token_ids.len();
        let final_logits =
            submit_prefill_state_evaluation(model, active, &hidden_states, chunk_ends_the_prompt)?;
        queue_prefill_first_token(model, active, final_logits.as_ref())?;
        let mut boundary_checkpoints = match intermediate_boundary_collector {
            Some(collector) => collector.complete().map_err(fatal_runtime_error)?,
            None => Vec::new(),
        };
        record_astro_ssd_prompt_cache_boundary_checkpoint(
            active,
            read_stretch.token_count(),
            &all_completed_prefill_chunk_tokens,
            &mut boundary_checkpoints,
        )?;
        Ok(boundary_checkpoints)
    }

    /// The block boundaries a chunk crosses when its blocks will be
    /// published.
    fn chunk_boundary_plan(
        &self,
        model: &PrismHadamardTextModel,
        active: &PrismHadamardQwen35ActiveGeneration,
        read_stretch: PromptStretch,
    ) -> Result<ChunkBoundaryPlan, InferenceEngineError> {
        let astro_ssd_prompt_cache_block_token_count =
            match self.execution_core.astro_ssd_prompt_cache.as_ref() {
                Some(astro_ssd_prompt_cache) if active.can_use_astro_ssd_prompt_cache => {
                    astro_ssd_prompt_cache
                        .model_contract_ref()
                        .block_token_count()
                }
                _ => 0,
            };
        let all_completed_prefill_chunk_tokens =
            astro_ssd_prompt_cache_boundary_completed_prefill_chunk_tokens(
                read_stretch.prefill_start,
                read_stretch.prefill_end,
                astro_ssd_prompt_cache_block_token_count,
            );
        let mut intermediate_completed_prefill_chunk_tokens =
            all_completed_prefill_chunk_tokens.clone();
        if intermediate_completed_prefill_chunk_tokens.last().copied()
            == Some(read_stretch.token_count())
        {
            intermediate_completed_prefill_chunk_tokens.pop();
        }
        let intermediate_boundary_collector =
            if intermediate_completed_prefill_chunk_tokens.is_empty() {
                None
            } else {
                Some(
                    AstroSsdPromptCacheBoundaryCheckpointCollector::new(
                        intermediate_completed_prefill_chunk_tokens,
                        model
                            .astro_decoder_state_cache_layout
                            .boundary_tensor_count(),
                        astro_ssd_prompt_cache_block_token_count,
                    )
                    .map_err(fatal_runtime_error)?,
                )
            };
        Ok(ChunkBoundaryPlan {
            all_completed_prefill_chunk_tokens,
            intermediate_boundary_collector,
        })
    }
}

/// The shared progress token for a read chunk. A folded pack never runs
/// prompt work on the Neural Engine, so there is no chunk report for it.
fn prefill_progress_token(
    execution: &PrismHadamardQwen35InferenceExecution,
    active: &PrismHadamardQwen35ActiveGeneration,
    request_id: RequestId,
    read_stretch: PromptStretch,
    stretch_timing: PromptStretchTiming,
) -> Result<GeneratedToken, InferenceEngineError> {
    prompt_reading_steps::prefill_progress_token(
        execution,
        active,
        request_id,
        read_stretch,
        stretch_timing,
        None,
    )
}

/// Computes the final logits when this chunk exhausted the prompt, then
/// submits the chunk's state graph for asynchronous evaluation. Both
/// submission branches sit inside their own
/// `PrefillStateAsyncEvaluationSubmission` region and return the final
/// logits so the first decode token can be sampled from them.
fn submit_prefill_state_evaluation(
    model: &PrismHadamardTextModel,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    hidden_states: &MlxArray,
    chunk_ends_the_prompt: bool,
) -> Result<Option<MlxArray>, InferenceEngineError> {
    let final_logits = if chunk_ends_the_prompt {
        Some(
            active_generation
                .measure_operation_with_request(
                    PerformanceOperation::FinalLogitsGraphConstruction,
                    |_| model.logits_for_last_hidden_state(hidden_states),
                )
                .map_err(gpu_error)?,
        )
    } else {
        None
    };
    if let Some(final_logits) = final_logits.as_ref() {
        active_generation
            .measure_operation_with_request(
                PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                |_| {
                    model
                        .runtime
                        .evaluate_arrays(&[hidden_states, final_logits])
                },
            )
            .map_err(gpu_error)?;
    } else {
        active_generation
            .measure_operation_with_request(
                PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                |_| model.runtime.evaluate_arrays(&[hidden_states]),
            )
            .map_err(gpu_error)?;
    }
    Ok(final_logits)
}

/// When the chunk completed the prompt, samples the first decode token with
/// the shared sampler inside `DecodeSamplingSpan`, kicks off its asynchronous
/// evaluation, and queues it as the pending token the next decode step reads.
/// The output budget is the shared decode step's to enforce, so the token is
/// always queued.
fn queue_prefill_first_token(
    model: &PrismHadamardTextModel,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    final_logits: Option<&MlxArray>,
) -> Result<(), InferenceEngineError> {
    if let Some(final_logits) = final_logits {
        let first_token = active_generation.measure_operation_with_request(
            PerformanceOperation::DecodeSamplingSpan,
            |active_generation| active_generation.build_generated_token(model, final_logits),
        )?;
        active_generation
            .measure_operation_with_request(
                PerformanceOperation::DecodeAsyncEvaluationSubmission,
                |_| model.runtime.async_eval_arrays(&[&first_token]),
            )
            .map_err(gpu_error)?;
        active_generation.pending_generated_token = Some(first_token);
    }
    Ok(())
}

#[cfg(test)]
#[path = "prefill_chunk_tests.rs"]
mod prefill_chunk_tests;

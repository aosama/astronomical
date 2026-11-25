//! Starts the next stretch of a prompt and finishes it once its last layer
//! has run.
//!
//! Every prompt is read one layer at a time over its spans
//! (`layer_at_a_time_prefill`). The order in this file is a correctness
//! contract:
//!
//! 1. Cut the spans from the cursor: the configured size, clamped at
//!    semantic control-span and durable astro SSD prompt cache boundaries. A stretch
//!    may publish several astro SSD prompt cache blocks, each from the state its span left.
//! 2. Read the stretch one layer at a time, reporting every measured step
//!    to the memory governor.
//! 3. Publish required astro SSD prompt cache state before advancing the in-memory cursor.
//! 4. Synchronize the tape and reclaim MLX buffer cache only above threshold.
//! 5. Emit progress only after all required state transitions succeeded.
//!
//! A capacity failure never retries smaller spans. The retired ladder halved
//! the executable chunk after each failure, which collapsed real requests into
//! dozens of forwards that re-paid complete-layer storage reads while making no
//! visible progress. Instead, a memory failure is redone with room leased
//! from the memory governor (`prefill_memory_recovery`): an expert model
//! runs its failed span again, and a prompt that cannot be put back to a
//! span is read again from the `astro_ssd_prompt_cache`. Either is terminal
//! only once the governor's attempts are spent.

use std::time::Instant;
use text_generation::generation_loop::prompt_reading_steps::{
    self, PromptStretch, PromptStretchTiming,
};

use ipc_protocol::RequestId;

use crate::{GeneratedToken, InferenceEngineError, last_prefill_chunk_demand_weight};

use super::completed_forward_memory;
use super::layer_at_a_time_prefill::PromptPrefillChunkOutcome;
use super::prefill_memory_recovery::PrefillAttemptFailure;
use super::{Qwen3_5EngineState, fatal_engine_error};
impl Qwen3_5EngineState {
    pub(super) fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::StreamingEngineRequest,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        if active_request
            .engine_extension
            .layer_at_a_time_prefill
            .is_some()
        {
            return self.continue_layer_at_a_time_prefill(request_id, active_request);
        }
        // Target-only prefill consumes every prompt token and samples first
        // output from the last chunk.
        let prefill_end_exclusive = active_request.input_token_ids.len();
        let Some(PrefillAttempt {
            prefill_start,
            attempted_prefill_chunk_token_count,
            layer_major_span_plan,
        }) = self.next_prompt_spans_to_read(active_request, prefill_end_exclusive)
        else {
            return Ok(None);
        };
        log_prefill_attempt(
            request_id,
            prefill_start,
            attempted_prefill_chunk_token_count,
            &layer_major_span_plan,
        );
        self.open_expert_demand_window_for_prefill(
            active_request,
            prefill_start,
            attempted_prefill_chunk_token_count,
            prefill_end_exclusive,
        );
        let prefill_end = self
            .prefill_end_for_attempted_chunk(prefill_start, attempted_prefill_chunk_token_count)?;
        let prefill_attempt_failure = PrefillAttemptFailure {
            prefill_start,
            attempted_prefill_chunk_token_count,
        };
        let begin_outcome = self.begin_layer_at_a_time_prefill(
            active_request,
            prefill_start,
            prefill_end,
            layer_major_span_plan,
        );
        match begin_outcome {
            Ok(()) => self.continue_layer_at_a_time_prefill(request_id, active_request),
            Err(prompt_prefill_chunk_attempt_error) => self.recover_failed_prefill_attempt(
                request_id,
                active_request,
                prefill_attempt_failure,
                prompt_prefill_chunk_attempt_error,
            ),
        }
    }

    /// After a forward read its tokens: publishes the astro SSD prompt cache state,
    /// moves the cursor, measures the forward and reports the tokens read.
    pub(super) fn complete_prefill_attempt(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::StreamingEngineRequest,
        prefill_start: usize,
        prefill_end: usize,
        prompt_prefill_chunk_outcome: PromptPrefillChunkOutcome,
        forward_chunk_started_at: Instant,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        // No post-chunk weight reset is needed: decode evidence ignores the
        // prefill multiplier entirely (see `demand_phase`), and the next
        // chunk's window-open sets its own weight before recording.
        let forward_chunk_elapsed_millis =
            prompt_prefill_chunk_outcome.forward_chunk_elapsed_millis;
        self.hand_prompt_stretch_taps_to_the_drafter(active_request);
        let read_stretch = PromptStretch {
            prefill_start,
            prefill_end,
        };
        prompt_reading_steps::publish_and_advance_durable_prefill_state(
            self,
            active_request,
            read_stretch,
            prompt_prefill_chunk_outcome.boundary_checkpoints,
        )?;
        prompt_reading_steps::reclaim_mlx_buffer_cache_after_prefill_chunk(
            self,
            request_id,
            active_request,
            read_stretch,
        )?;
        self.record_completed_forward_memory(
            active_request,
            prompt_prefill_chunk_outcome.measured_forward_token_count,
            prompt_prefill_chunk_outcome.active_memory_bytes_before_forward,
        )?;
        Ok(Some(prompt_reading_steps::prefill_progress_token(
            self,
            active_request,
            request_id,
            read_stretch,
            PromptStretchTiming {
                forward_chunk_started_at,
                forward_chunk_elapsed_millis,
            },
            None,
        )?))
    }
}

fn log_prefill_attempt(
    request_id: RequestId,
    prefill_start: usize,
    attempted_prefill_chunk_token_count: usize,
    layer_major_span_plan: &super::layer_major_span_plan::LayerMajorSpanPlan,
) {
    tracing::info!(
        request_id = request_id.value(),
        prefill_start,
        attempted_prefill_chunk_token_count,
        span_count = layer_major_span_plan.spans.len(),
        "reading prompt tokens one layer at a time"
    );
}

/// What the next prefill attempt runs: the spans from the cursor to where
/// the plan stops, read one layer at a time.
struct PrefillAttempt {
    prefill_start: usize,
    attempted_prefill_chunk_token_count: usize,
    layer_major_span_plan: super::layer_major_span_plan::LayerMajorSpanPlan,
}

impl Qwen3_5EngineState {
    /// The spans the next attempt reads. Every prompt is read one layer at
    /// a time over its spans, so a layer read from storage is read once.
    fn next_prompt_spans_to_read(
        &self,
        active_request: &super::engine_request::StreamingEngineRequest,
        prefill_end_exclusive: usize,
    ) -> Option<PrefillAttempt> {
        let engine_core::astro_ssd_prompt_cache::ResolvedPrefillChunkSpan { prefill_start, .. } =
            prompt_reading_steps::resolve_next_prefill_chunk_span(
                self,
                active_request,
                prefill_end_exclusive,
            )?;
        let layer_major_span_plan =
            self.layer_major_span_plan_for(active_request, prefill_start, prefill_end_exclusive)?;
        Some(PrefillAttempt {
            prefill_start,
            attempted_prefill_chunk_token_count: layer_major_span_plan.group_end_token
                - prefill_start,
            layer_major_span_plan,
        })
    }

    /// Opens the request's one demand-evidence window for prefill and sets the
    /// final chunk's boosted assignment weight. Mirrors the inlined block.
    fn open_expert_demand_window_for_prefill(
        &mut self,
        active_request: &mut super::engine_request::StreamingEngineRequest,
        prefill_start: usize,
        attempted_prefill_chunk_token_count: usize,
        prefill_end_exclusive: usize,
    ) {
        // One prefill demand window per request: the first chunk opens it —
        // discarding the previous request's prompt evidence while lifetime
        // decode evidence persists — and every chunk sets its own weight so
        // earlier chunks record at weight one while the final chunk records
        // boosted, ranking prompt-tail routes by token density for the
        // prefill-to-decode handoff.
        if !active_request
            .engine_extension
            .has_opened_prefill_demand_window
        {
            if let Some(model) = self.core.model.as_ref() {
                model.begin_prefill_demand_window_for_residency();
            }
            active_request
                .engine_extension
                .has_opened_prefill_demand_window = true;
        }
        let prefill_chunk_demand_weight =
            if prefill_start + attempted_prefill_chunk_token_count >= prefill_end_exclusive {
                last_prefill_chunk_demand_weight(
                    u64::try_from(prefill_start).unwrap_or(u64::MAX),
                    u64::try_from(attempted_prefill_chunk_token_count).unwrap_or(u64::MAX),
                )
            } else {
                1
            };
        if let Some(model) = self.core.model.as_ref() {
            model.set_prefill_demand_assignment_weight_for_residency(prefill_chunk_demand_weight);
        }
    }

    /// The exclusive chunk end for one attempted span; overflow is a fatal
    /// generation error, exactly as the inlined checked_add was.
    fn prefill_end_for_attempted_chunk(
        &self,
        prefill_start: usize,
        attempted_prefill_chunk_token_count: usize,
    ) -> Result<usize, InferenceEngineError> {
        prefill_start
            .checked_add(attempted_prefill_chunk_token_count)
            .ok_or_else(|| fatal_engine_error("prefill chunk end overflowed"))
    }

    /// Measures one completed prompt forward and tells the memory governor.
    fn record_completed_forward_memory(
        &mut self,
        active_request: &mut super::engine_request::StreamingEngineRequest,
        measured_forward_token_count: usize,
        active_memory_bytes_before_forward: usize,
    ) -> Result<(), InferenceEngineError> {
        let model = self.core.loaded_model()?;
        completed_forward_memory::measure_completed_forward(
            model,
            completed_forward_memory::MeasuredForward::prompt(measured_forward_token_count),
            active_request.request_decoder_state.payload_byte_count(),
            active_memory_bytes_before_forward,
            &mut active_request.performance_attribution,
        )?;
        Ok(())
    }
}

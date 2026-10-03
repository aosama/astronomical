//! Forward dispatch for a prompt-processing chunk.
//!
//! Selects and executes one of three mutually exclusive forward paths:
//!
//! 1. **Visual prefill** — the current chunk contains image-pad tokens and the
//!    request still owns visual embeddings. A request-level vision tensor alone
//!    is not enough; text-only suffixes must use the text path so cache restore
//!    and cold prefill seed the first generated token the same way.
//! 2. **Terminal history capture** — optional multi-token prediction history
//!    initialization on the final chunk.
//! 3. **Plain text prefill** — the common path; may seed the first generated
//!    token if this is the terminal chunk, or forward with optional prompt-cache
//!    boundary checkpoints.

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::prefill_chunk_planning::PrefillChunkPlan;
use super::terminal_prefill_seed::{
    chunk_requires_visual_embeddings, seed_first_generated_token_from_terminal_prefill_chunk,
    seed_terminal_text_prefill_after_prompt_cache_boundaries,
};

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5::multi_token_prediction::{
    execute_terminal_optional_history_capture_with_performance_attribution,
    record_prompt_history_initialization_fallback,
};

/// Outcome of visual, history-capture, or text prefill.
pub(super) struct ForwardDispatchOutcome {
    pub(super) boundary_checkpoints: Vec<crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

/// Forward-path failure before the orchestrator attaches a request checkpoint.
///
/// GPU execution failures keep their typed `Qwen3_5ExecutionError` so capacity
/// evidence can still drive retry.
pub(super) enum PrefillForwardError {
    Execution(Qwen3_5ExecutionError),
}

impl From<Qwen3_5ExecutionError> for PrefillForwardError {
    fn from(execution_error: Qwen3_5ExecutionError) -> Self {
        Self::Execution(execution_error)
    }
}

impl Qwen3_5EngineState {
    /// Execute the appropriate forward path for this chunk based on the plan.
    ///
    /// Returns execution errors without wrapping in `PromptPrefillChunkAttemptError`
    /// so the orchestrator can attach the checkpoint.
    pub(super) fn dispatch_prefill_forward(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
        plan: &PrefillChunkPlan,
    ) -> Result<ForwardDispatchOutcome, PrefillForwardError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Qwen3_5ExecutionError::InvalidInput {
                description: "Qwen3.5 engine lost its loaded model",
            })?;

        let dispatch_outcome = dispatch_non_speculative_forward(
            active_request,
            model,
            prefill_start,
            prefill_end,
            plan,
        )?;

        Ok(ForwardDispatchOutcome {
            boundary_checkpoints: dispatch_outcome.boundary_checkpoints,
        })
    }
}

struct NonSpeculativeOutcome {
    boundary_checkpoints: Vec<crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

fn dispatch_non_speculative_forward(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5Model,
    prefill_start: usize,
    prefill_end: usize,
    plan: &PrefillChunkPlan,
) -> Result<NonSpeculativeOutcome, PrefillForwardError> {
    let mut boundary_checkpoints = Vec::new();
    // Request-owned visual embeddings stay attached after every image-pad in this
    // prompt has already been consumed. Dispatching on that tensor made a cached
    // suffix take text prefill with first-token seeding while the cold suffix of
    // the same text took visual prefill without seeding, so the first generated
    // token diverged after a lossless KV restore.
    let chunk_requires_visual_embeddings = chunk_requires_visual_embeddings(
        active_request,
        &active_request.input_token_ids[prefill_start..prefill_end],
    );

    if chunk_requires_visual_embeddings {
        if active_request.visual_embeddings.is_none() {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description:
                    "prefill chunk contains image-pad tokens but visual embeddings are missing",
            }
            .into());
        }
        let (consumed_visual_embedding_count, visual_boundary_checkpoints) =
            dispatch_visual_prefill(
                active_request,
                model,
                prefill_start,
                prefill_end,
                &plan.intermediate_completed_prefill_chunk_tokens,
                plan.persistent_prompt_cache_block_token_count,
            )?;
        boundary_checkpoints = visual_boundary_checkpoints;
        active_request.consumed_visual_embedding_count += consumed_visual_embedding_count;
    } else if plan.is_terminal_optional_history_capture {
        dispatch_terminal_history_capture(active_request, model, prefill_start, prefill_end)?;
    } else {
        boundary_checkpoints = dispatch_text_prefill(
            active_request,
            model,
            prefill_start,
            prefill_end,
            &plan.intermediate_completed_prefill_chunk_tokens,
            plan.persistent_prompt_cache_block_token_count,
        )?;
    }

    Ok(NonSpeculativeOutcome {
        boundary_checkpoints,
    })
}

// ---------------------------------------------------------------------------
// Visual prefill
// ---------------------------------------------------------------------------

fn dispatch_visual_prefill(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5Model,
    prefill_start: usize,
    prefill_end: usize,
    intermediate_completed_prefill_chunk_tokens: &[usize],
    persistent_prompt_cache_block_token_count: Option<usize>,
) -> Result<
    (
        usize,
        Vec<crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    ),
    Qwen3_5ExecutionError,
> {
    // Extract visual embeddings reference before borrowing mutably.
    let visual_embeddings = active_request
        .visual_embeddings
        .as_ref()
        .expect("dispatch_visual_prefill called when visual_embeddings is None");

    if intermediate_completed_prefill_chunk_tokens.is_empty() {
        model
            .prefill_chunk_with_visual_embeddings_and_performance_attribution(
                &active_request.input_token_ids[prefill_start..prefill_end],
                active_request.next_position_tokens,
                visual_embeddings,
                active_request.consumed_visual_embedding_count,
                &mut active_request.request_decoder_state,
                active_request.image_pad_token_id,
                &mut active_request.performance_attribution,
            )
            .map(|consumed_visual_embedding_count| (consumed_visual_embedding_count, Vec::new()))
    } else {
        let persistent_prompt_cache_block_token_count = persistent_prompt_cache_block_token_count
            .ok_or_else(|| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "planned visual prompt-cache checkpoints have no block contract",
            }
        })?;
        model
            .prefill_chunk_with_visual_embeddings_and_boundary_checkpoints_with_performance_attribution(
                &active_request.input_token_ids[prefill_start..prefill_end],
                active_request.next_position_tokens,
                visual_embeddings,
                active_request.consumed_visual_embedding_count,
                &mut active_request.request_decoder_state,
                active_request.image_pad_token_id,
                intermediate_completed_prefill_chunk_tokens.to_vec(),
                persistent_prompt_cache_block_token_count,
                &mut active_request.performance_attribution,
            )
            .map(|checkpoint_outcome| {
                (
                    checkpoint_outcome.consumed_visual_embedding_count,
                    checkpoint_outcome.boundary_checkpoints,
                )
            })
    }
}

// ---------------------------------------------------------------------------
// Terminal history capture
// ---------------------------------------------------------------------------

fn dispatch_terminal_history_capture(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5Model,
    prefill_start: usize,
    prefill_end: usize,
) -> Result<usize, Qwen3_5ExecutionError> {
    let optional_history_capture_result =
        execute_terminal_optional_history_capture_with_performance_attribution(
            model,
            prefill_start,
            prefill_end,
            active_request,
        );
    match optional_history_capture_result {
        Ok(history_token_count) => Ok(history_token_count),
        Err(optional_history_capture_error) => {
            // Only terminal optional-prefill fallback errors are recoverable.
            // Non-fallback errors propagate up; the orchestrator wraps them
            // with the checkpoint.
            if super::prompt_prefill_errors::terminal_optional_prefill_error_is_fallback(
                &optional_history_capture_error,
            ) {
                tracing::warn!(
                    request_id = active_request.request_id.value(),
                    error = %optional_history_capture_error,
                    "optional terminal history initialization failed; continuing target-only"
                );
                active_request.clear_optional_prediction_session();
                record_prompt_history_initialization_fallback(active_request);
                Ok(0)
            } else {
                Err(optional_history_capture_error)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plain text prefill
// ---------------------------------------------------------------------------

fn dispatch_text_prefill(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5Model,
    prefill_start: usize,
    prefill_end: usize,
    intermediate_completed_prefill_chunk_tokens: &[usize],
    persistent_prompt_cache_block_token_count: Option<usize>,
) -> Result<Vec<crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint>, Qwen3_5ExecutionError> {
    let chunk_includes_final_prompt_token = prefill_end == active_request.input_token_ids.len();
    if chunk_includes_final_prompt_token && intermediate_completed_prefill_chunk_tokens.is_empty() {
        seed_first_generated_token_from_terminal_prefill_chunk(
            model,
            active_request,
            prefill_start,
            prefill_end,
        )?;
        Ok(Vec::new())
    } else if intermediate_completed_prefill_chunk_tokens.is_empty() {
        model
            .prefill_chunk_with_performance_attribution(
                &active_request.input_token_ids[prefill_start..prefill_end],
                active_request.next_position_tokens,
                &mut active_request.request_decoder_state,
                &mut active_request.performance_attribution,
            )
            .map(|()| Vec::new())
    } else {
        let persistent_prompt_cache_block_token_count = persistent_prompt_cache_block_token_count
            .ok_or_else(|| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "planned text prompt-cache checkpoints have no block contract",
            }
        })?;
        if chunk_includes_final_prompt_token {
            return seed_terminal_text_prefill_after_prompt_cache_boundaries(
                active_request,
                model,
                prefill_start,
                prefill_end,
                intermediate_completed_prefill_chunk_tokens,
                persistent_prompt_cache_block_token_count,
            );
        }
        model
            .prefill_chunk_with_boundary_checkpoints_and_performance_attribution(
                &active_request.input_token_ids[prefill_start..prefill_end],
                active_request.next_position_tokens,
                &mut active_request.request_decoder_state,
                intermediate_completed_prefill_chunk_tokens.to_vec(),
                persistent_prompt_cache_block_token_count,
                &mut active_request.performance_attribution,
            )
            .map(|checkpoint_outcome| checkpoint_outcome.boundary_checkpoints)
    }
}

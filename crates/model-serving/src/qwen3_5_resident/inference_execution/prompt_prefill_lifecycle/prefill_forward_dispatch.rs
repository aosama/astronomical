//! Forward dispatch for a prompt-processing chunk.
//!
//! Selects and executes one of two mutually exclusive forward paths:
//!
//! 1. **Visual prefill** — the current chunk contains image-pad tokens and the
//!    request still owns visual embeddings. A request-level vision tensor alone
//!    is not enough; text-only suffixes must use the text path so cache restore
//!    and cold prefill seed the first generated token the same way.
//! 2. **Plain text prefill** — the common path; may seed the first generated
//!    token if this is the terminal chunk, or forward with optional prompt-cache
//!    boundary checkpoints.

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::prefill_chunk_planning::PrefillChunkPlan;
use super::terminal_prefill_seed::{
    chunk_requires_visual_embeddings, seed_first_generated_token_from_terminal_prefill_chunk,
    seed_terminal_text_prefill_after_prompt_cache_boundaries,
};

use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use crate::qwen3_5_resident::model::VerificationLogitsMode;

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
        let model = self.model.as_ref().ok_or_else(
            super::prefill_invalid_input_error_constructors::engine_lost_loaded_model_invalid_input,
        )?;

        let dispatch_outcome = if active_request.speculative_session_admitted {
            dispatch_speculative_forward(self, active_request, model, prefill_start, prefill_end)?
        } else {
            dispatch_non_speculative_forward(
                active_request,
                model,
                prefill_start,
                prefill_end,
                plan,
            )?
        };

        Ok(ForwardDispatchOutcome {
            boundary_checkpoints: dispatch_outcome.boundary_checkpoints,
        })
    }
}

/// The speculative counterpart of the non-speculative dispatch: every
/// admitted text chunk runs through the tap-capturing verification forward
/// so the draft's context key-value cache grows from the target's hidden
/// states chunk by chunk. The terminal chunk additionally seeds the first
/// generated token exactly as the classic terminal path does. Boundary
/// checkpoints are not produced: a speculative session forgoes prompt-cache
/// capture in both directions by decision at admission.
fn dispatch_speculative_forward(
    engine_state: &Qwen3_5EngineState,
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5ResidentModel,
    prefill_start: usize,
    prefill_end: usize,
) -> Result<ChunkForwardOutcome, PrefillForwardError> {
    let draft_runtime = engine_state.speculative_draft_runtime.as_ref().ok_or(
        Qwen3_5ExecutionError::InvalidInput {
            description: "speculative prefill requires the resident draft runtime",
        },
    )?;
    // Admission guarantees a text-only session (visual inputs stay classic),
    // so every admitted chunk takes the plain text path — no image-pad
    // tokens can appear here.
    let chunk_includes_final_prompt_token = prefill_end == active_request.input_token_ids.len();
    let logits_mode = if chunk_includes_final_prompt_token {
        VerificationLogitsMode::LastRowOnly
    } else {
        VerificationLogitsMode::NoLogits
    };
    let chunk_token_ids = active_request.input_token_ids[prefill_start..prefill_end].to_vec();
    let chunk_output = model.build_speculative_verification_forward(
        &chunk_token_ids,
        active_request.next_position_tokens,
        &mut active_request.request_decoder_state,
        draft_runtime.draft_configuration.target_layer_ids(),
        logits_mode,
        &mut active_request.performance_attribution,
    )?;
    let mut draft_key_value_states = active_request
        .speculative_draft_key_value_states
        .take()
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative prefill requires the request's draft key-value states",
        })?;
    let conditioned_states =
        crate::qwen3_5_resident::dflash_draft::target_hidden_state_conditioning::condition_target_hidden_states(
            model.runtime(),
            &draft_runtime.draft.weights,
            &chunk_output.hidden_state_taps,
            draft_runtime.draft_configuration.rms_norm_epsilon(),
        )
        .map_err(Qwen3_5ExecutionError::from)?;
    crate::qwen3_5_resident::dflash_draft::draft_context_kv_projection::append_conditioned_context_key_values(
        model.runtime(),
        &draft_runtime.draft.weights,
        &draft_runtime.draft_configuration,
        &mut draft_key_value_states,
        &conditioned_states,
        prefill_start as i32,
    )
    .map_err(Qwen3_5ExecutionError::from)?;
    active_request.speculative_draft_key_value_states = Some(draft_key_value_states);
    if chunk_includes_final_prompt_token {
        let terminal_logits =
            chunk_output
                .per_position_logits
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "the terminal speculative chunk must produce its seeding logits",
                })?;
        let first_generated_token = active_request
            .build_generated_token(model, &terminal_logits)
            .map_err(
                super::prefill_invalid_input_error_constructors::terminal_prefill_first_generated_token_sampling_failed_invalid_input,
            )?;
        active_request.set_pending_generated_token(first_generated_token);
    }
    Ok(ChunkForwardOutcome {
        boundary_checkpoints: Vec::new(),
    })
}

/// What either prefill dispatch path produced for one chunk. Named for the
/// payload, not either path: the speculative branch returns the same shape
/// (its checkpoint list is always empty — admitted sessions forfeit
/// prompt-cache capture by decision at admission).
struct ChunkForwardOutcome {
    boundary_checkpoints: Vec<crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

fn dispatch_non_speculative_forward(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5ResidentModel,
    prefill_start: usize,
    prefill_end: usize,
    plan: &PrefillChunkPlan,
) -> Result<ChunkForwardOutcome, PrefillForwardError> {
    let boundary_checkpoints;
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

    Ok(ChunkForwardOutcome {
        boundary_checkpoints,
    })
}

// ---------------------------------------------------------------------------
// Visual prefill
// ---------------------------------------------------------------------------

fn dispatch_visual_prefill(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5ResidentModel,
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
.ok_or_else(super::prefill_invalid_input_error_constructors::visual_prompt_cache_checkpoints_have_no_block_contract_invalid_input)?;
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
// Plain text prefill
// ---------------------------------------------------------------------------

fn dispatch_text_prefill(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5ResidentModel,
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
.ok_or_else(super::prefill_invalid_input_error_constructors::text_prompt_cache_checkpoints_have_no_block_contract_invalid_input)?;
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

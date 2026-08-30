//! First-token seeding after the last prompt-processing chunk.
//!
//! Prefill of the complete prompt already produced last-prompt logits. Sampling
//! those logits is the first generated token. Re-forwarding the last prompt token
//! at the next position is a different graph and diverges from a cache restore
//! whose leftover tail is short enough to seed directly.

use super::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model, Qwen3_5TargetForwardOutput};

pub(super) fn chunk_requires_visual_embeddings(
    active_request: &Qwen3_5EngineRequest,
    chunk_token_ids: &[u32],
) -> bool {
    // The artifact's image-pad vocabulary ID can occur in ordinary ChatML token
    // sequences. Treat it as a visual row only when the request actually supplied
    // an image; otherwise text-only requests would be forced through vision.
    active_request.has_visual_inputs
        && chunk_token_ids
            .iter()
            .any(|token_id| *token_id == active_request.image_pad_token_id)
}

pub(super) fn seed_terminal_text_prefill_after_prompt_cache_boundaries(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5Model,
    prefill_start: usize,
    prefill_end: usize,
    intermediate_completed_prefill_chunck_tokens: &[usize],
    persistent_prompt_cache_block_token_count: usize,
) -> Result<Vec<crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint>, Qwen3_5ExecutionError> {
    // A terminal text chunk can still contain cache-block boundaries. The terminal
    // forward builds the vocabulary head while also collecting boundary checkpoints,
    // so the chunk is a single forward pass. This avoids the prefix+tail split that
    // would create an extra expert streaming pass for SSD-paged models.
    let terminal_outcome = model
        .terminal_prefill_chunck_with_boundary_checkpoints_and_performance_attribution(
            &active_request.input_token_ids[prefill_start..prefill_end],
            active_request.next_position_tokens,
            &mut active_request.request_decoder_state,
            intermediate_completed_prefill_chunck_tokens.to_vec(),
            persistent_prompt_cache_block_token_count,
            &mut active_request.performance_attribution,
        )?;
    let prompt_token_count = prefill_end - prefill_start;
    seed_first_generated_token_from_terminal_forward_output(
        model,
        active_request,
        &terminal_outcome.target_forward_output,
        prompt_token_count,
    )?;
    Ok(terminal_outcome.boundary_checkpoints)
}

pub(super) fn seed_first_generated_token_from_terminal_prefill_chunk(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    prefill_start: usize,
    prefill_end: usize,
) -> Result<(), Qwen3_5ExecutionError> {
    let prompt_token_ids = active_request.input_token_ids[prefill_start..prefill_end].to_vec();
    if active_request.has_optional_prediction_session() {
        let target_forward_output = model
            .forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                &prompt_token_ids,
                active_request.next_position_tokens,
                &mut active_request.request_decoder_state,
                &mut active_request.performance_attribution,
            )?;
        let last_hidden_row_index = i32::try_from(prompt_token_ids.len().saturating_sub(1))
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "terminal prefill token count exceeds the MLX int32 range",
            })?;
        let last_prompt_hidden_state = target_forward_output
            .pre_final_normalization_hidden_state_at(model.runtime(), last_hidden_row_index)
            .map_err(Qwen3_5ExecutionError::from)?;
        let first_generated_token =
            active_request
                .build_generated_token(model, target_forward_output.final_logits())
                .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                    description:
                        "failed to sample the first generated token from terminal prefill logits",
                })?;
        active_request.set_pending_generated_token(first_generated_token);
        if let Some(prediction_session) = active_request.optional_prediction_session_mut() {
            prediction_session.set_target_hidden_states(Some(last_prompt_hidden_state));
        }
        return Ok(());
    }
    let final_prompt_logits = model.forward_chunk_with_performance_attribution(
        &prompt_token_ids,
        active_request.next_position_tokens,
        &mut active_request.request_decoder_state,
        &mut active_request.performance_attribution,
    )?;
    let first_generated_token = active_request
        .build_generated_token(model, &final_prompt_logits)
        .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
            description: "failed to sample the first generated token from terminal prefill logits",
        })?;
    active_request.set_pending_generated_token(first_generated_token);
    Ok(())
}

/// Seeds the first generated token from a terminal forward output that already
/// carries both logits and pre-normalization hidden states.
///
/// Used by the terminal checkpointed prefill path, which produces logits and
/// cache checkpoints in a single forward. The hidden states are evaluated
/// explicitly when an optional prediction session needs them for MTP.
fn seed_first_generated_token_from_terminal_forward_output(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    target_forward_output: &Qwen3_5TargetForwardOutput,
    prompt_token_count: usize,
) -> Result<(), Qwen3_5ExecutionError> {
    if active_request.has_optional_prediction_session() {
        model
            .runtime()
            .evaluate_arrays(&[target_forward_output.pre_final_normalization_hidden_states()])
            .map_err(Qwen3_5ExecutionError::from)?;
        let last_hidden_row_index =
            i32::try_from(prompt_token_count.saturating_sub(1)).map_err(|_| {
                Qwen3_5ExecutionError::InvalidInput {
                    description: "terminal prefill token count exceeds the MLX int32 range",
                }
            })?;
        let last_prompt_hidden_state = target_forward_output
            .pre_final_normalization_hidden_state_at(model.runtime(), last_hidden_row_index)
            .map_err(Qwen3_5ExecutionError::from)?;
        let first_generated_token =
            active_request
                .build_generated_token(model, target_forward_output.final_logits())
                .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                    description:
                        "failed to sample the first generated token from terminal prefill logits",
                })?;
        active_request.set_pending_generated_token(first_generated_token);
        if let Some(prediction_session) = active_request.optional_prediction_session_mut() {
            prediction_session.set_target_hidden_states(Some(last_prompt_hidden_state));
        }
        return Ok(());
    }
    let first_generated_token = active_request
        .build_generated_token(model, target_forward_output.final_logits())
        .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
            description: "failed to sample the first generated token from terminal prefill logits",
        })?;
    active_request.set_pending_generated_token(first_generated_token);
    Ok(())
}

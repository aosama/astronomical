//! One speculative decoding round: draft, verify, accept or correct.
//!
//! A round takes the bonus token (the last token the target itself
//! endorsed), drafts `k` proposals with the DFlash trunk in one parallel
//! pass, then asks the target to verify the block in a single multi-token
//! forward that also captures the conditioning taps. Under greedy
//! verification the accepted prefix is the longest run of proposals the
//! target's argmax agrees with.
//!
//! The target's decoder-state rollback uses the transactional checkpoint
//! the state stack has always owned: checkpoint before the verification
//! forward, and on rejection restore and re-forward only the accepted
//! prefix (at most `k + 1` tokens). The full-attention layers could truncate
//! in place, but the gated-delta linear layers are recurrent and cannot be
//! partially rewound, so the uniform transactional path is the correct
//! first implementation; layer-split rollback is a measured optimization
//! for later.

use astronomical_ipc_protocol::RequestId;
use mlx_c_rust::MlxArray;

use crate::performance_attribution::PerformanceAttribution;
use crate::qwen3_5_core::decoder::request_state::RequestDecoderStateStack;
use crate::qwen3_5_resident::dflash_draft;
use crate::qwen3_5_resident::model::Qwen3_5ExecutionError;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use crate::qwen3_5_resident::model::VerificationLogitsMode;

use super::verification_decisions::{
    GreedyVerificationDecision, decide_greedy_verification, emitted_tokens_for_round,
};

/// Everything one round needs besides the owners it mutates.
#[derive(Debug, Clone, Copy)]
pub struct SpeculationRoundRequest {
    pub request_id: RequestId,
    /// The token the target itself endorsed last; it seeds the draft block.
    pub bonus_token_id: u32,
    /// How many proposals the draft drafts this round (`k`). This is the
    /// configured speculative decoding block length clamped by the
    /// session's remaining output budget — the configured ceiling, not
    /// necessarily the per-round value.
    pub speculation_token_count: usize,
    /// The shared context length: target and draft key-value offsets.
    pub context_length_tokens: u32,
}

/// What one round produced.
#[derive(Debug, Clone)]
pub struct SpeculationRoundOutcome {
    /// Tokens the round emits, in order (bonus, accepted proposals, and the
    /// target's correction token when the draft was caught out).
    pub emitted_token_ids: Vec<u32>,
    /// The token the next round seeds from: the target's own preference.
    pub next_bonus_token_id: u32,
    /// How many of the drafted proposals the target confirmed.
    pub accepted_proposal_count: usize,
    /// The context length after the round, for the caller's bookkeeping.
    pub context_length_after_round: u32,
}

/// Runs one speculative round. Mutates the target's decoder-state stack and
/// the draft's key-value states in lockstep; both end at the accepted
/// context length.
#[allow(clippy::too_many_arguments)]
pub fn run_speculative_round(
    target_model: &Qwen3_5ResidentModel,
    target_decoder_state: &mut RequestDecoderStateStack,
    draft_weights: &dflash_draft::draft_weights::DFlashDraftWeights,
    draft_configuration: &dflash_draft::draft_configuration::DFlashDraftConfiguration,
    draft_key_value_states: &mut dflash_draft::draft_key_value_state::DFlashDraftKeyValueStates,
    round_request: &SpeculationRoundRequest,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<SpeculationRoundOutcome, Qwen3_5ExecutionError> {
    let context_length_tokens = round_request.context_length_tokens;
    if draft_key_value_states.offset_tokens() != context_length_tokens as i32 {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "the draft key-value context length must match the target context length before a speculative round",
        });
    }

    let proposals = draft_proposals(
        target_model,
        draft_weights,
        draft_configuration,
        draft_key_value_states,
        round_request,
        performance_attribution,
    )?;

    // The draft key-value state now holds the discarded block scratch at
    // positions after the context; roll it back before the round ends so
    // the next round starts from the clean context boundary.
    let draft_checkpoint_offset = context_length_tokens as i32;

    let verification_tokens =
        assemble_verification_tokens(round_request.bonus_token_id, &proposals);
    let target_checkpoint = target_decoder_state
        .checkpoint()
        .map_err(Qwen3_5ExecutionError::from)?;
    let verification_output = target_model.build_speculative_verification_forward(
        &verification_tokens,
        context_length_tokens,
        target_decoder_state,
        draft_configuration.target_layer_ids(),
        VerificationLogitsMode::AllPositions,
        performance_attribution,
    )?;
    let verification_logits = verification_output.per_position_logits.as_ref().ok_or(
        Qwen3_5ExecutionError::InvalidInput {
            description: "a speculative verification round requires per-position logits",
        },
    )?;
    let target_preferred_token_ids = argmax_rows_to_host(target_model, verification_logits)?;

    let verification_decision = decide_greedy_verification(&proposals, &target_preferred_token_ids);
    if verification_decision.accepted_every_proposal {
        finish_accepted_round(
            target_model,
            draft_weights,
            draft_configuration,
            draft_key_value_states,
            round_request,
            &verification_output.hidden_state_taps,
            &proposals,
            &target_preferred_token_ids,
            &verification_decision,
            draft_checkpoint_offset,
        )
    } else {
        // Partial or empty acceptance: restore the transactional checkpoint
        // and replay only the accepted prefix, which rebuilds the recurrent
        // linear-attention states the rejection would otherwise corrupt.
        target_decoder_state
            .restore_checkpoint(target_checkpoint)
            .map_err(Qwen3_5ExecutionError::from)?;
        let replay_tokens = assemble_replay_tokens(
            round_request.bonus_token_id,
            &proposals,
            &verification_decision,
        );
        if !replay_tokens.is_empty() {
            let _ = target_model.build_speculative_verification_forward(
                &replay_tokens,
                context_length_tokens,
                target_decoder_state,
                &[],
                VerificationLogitsMode::NoLogits,
                performance_attribution,
            )?;
        }
        finish_accepted_round(
            target_model,
            draft_weights,
            draft_configuration,
            draft_key_value_states,
            round_request,
            &verification_output.hidden_state_taps,
            &proposals,
            &target_preferred_token_ids,
            &verification_decision,
            draft_checkpoint_offset,
        )
    }
}

/// Drafts `k` proposals with one parallel trunk pass. The block's own
/// key-value scratch is left in the draft states past the context boundary;
/// the caller truncates it once the round's acceptance is known.
fn draft_proposals(
    target_model: &Qwen3_5ResidentModel,
    draft_weights: &dflash_draft::draft_weights::DFlashDraftWeights,
    draft_configuration: &dflash_draft::draft_configuration::DFlashDraftConfiguration,
    draft_key_value_states: &mut dflash_draft::draft_key_value_state::DFlashDraftKeyValueStates,
    round_request: &SpeculationRoundRequest,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Vec<u32>, Qwen3_5ExecutionError> {
    let speculation_token_count = round_request.speculation_token_count;
    let block_token_ids = assemble_block_token_ids(
        round_request.bonus_token_id,
        draft_configuration.mask_token_id(),
        speculation_token_count,
    );
    let block_embeddings = embed_block_tokens(target_model, &block_token_ids)?;
    let block_hidden_states = dflash_draft::draft_forward::forward_draft_block(
        &target_model.runtime,
        draft_weights,
        draft_configuration,
        draft_key_value_states,
        &block_embeddings,
        round_request.context_length_tokens as i32,
        performance_attribution,
    )
    .map_err(Qwen3_5ExecutionError::from)?;
    // Score the draft's hidden states with the target's language-model head —
    // the draft ships no head of its own.
    let block_logits = target_model.quantized_linear(
        &block_hidden_states,
        &target_model.weights.language_model_head_weights,
    )?;
    let block_logits = target_model
        .runtime
        .astype(&block_logits, mlx_c_rust::MlxDtype::Float32)?;
    // Row zero is the bonus position; the proposals live in rows one onward.
    let proposal_logits = target_model.runtime.slice(
        &block_logits,
        &[0, 1, 0],
        &[
            1,
            1 + speculation_token_count as i32,
            block_logits.shape()[2],
        ],
        &[1, 1, 1],
    )?;
    let proposal_argmax = argmax_rows_to_host(target_model, &proposal_logits)?;
    Ok(proposal_argmax)
}

#[allow(clippy::too_many_arguments)]
fn finish_accepted_round(
    target_model: &Qwen3_5ResidentModel,
    draft_weights: &dflash_draft::draft_weights::DFlashDraftWeights,
    draft_configuration: &dflash_draft::draft_configuration::DFlashDraftConfiguration,
    draft_key_value_states: &mut dflash_draft::draft_key_value_state::DFlashDraftKeyValueStates,
    round_request: &SpeculationRoundRequest,
    hidden_state_taps: &[MlxArray],
    proposals: &[u32],
    target_preferred_token_ids: &[u32],
    verification_decision: &GreedyVerificationDecision,
    draft_checkpoint_offset: i32,
) -> Result<SpeculationRoundOutcome, Qwen3_5ExecutionError> {
    let accepted_context_growth =
        super::verification_decisions::accepted_context_growth(verification_decision);
    // Discard the draft's block scratch (context entries are re-derived from
    // the target's taps, never the draft's own projections).
    draft_key_value_states
        .truncate_to_offset(draft_checkpoint_offset)
        .map_err(Qwen3_5ExecutionError::from)?;
    // Grow the draft's context key-value from the accepted prefix of the
    // target's taps: rows zero..accepted growth of every tap.
    let accepted_taps =
        slice_taps_to_prefix(target_model, hidden_state_taps, accepted_context_growth)?;
    let conditioned_states =
        dflash_draft::target_hidden_state_conditioning::condition_target_hidden_states(
            &target_model.runtime,
            draft_weights,
            &accepted_taps,
            draft_configuration.rms_norm_epsilon(),
        )
        .map_err(Qwen3_5ExecutionError::from)?;
    dflash_draft::draft_context_kv_projection::append_conditioned_context_key_values(
        &target_model.runtime,
        draft_weights,
        draft_configuration,
        draft_key_value_states,
        &conditioned_states,
        round_request.context_length_tokens as i32,
    )
    .map_err(Qwen3_5ExecutionError::from)?;

    let emitted_token_ids = emitted_tokens_for_round(
        round_request.bonus_token_id,
        proposals,
        target_preferred_token_ids,
        verification_decision,
    );
    let next_bonus_token_id =
        *emitted_token_ids
            .last()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "a speculative round must emit at least the bonus token",
            })?;
    Ok(SpeculationRoundOutcome {
        emitted_token_ids,
        next_bonus_token_id,
        accepted_proposal_count: verification_decision.accepted_proposal_count,
        context_length_after_round: round_request.context_length_tokens
            + accepted_context_growth as u32,
    })
}

fn assemble_block_token_ids(
    bonus_token_id: u32,
    mask_token_id: i32,
    speculation_token_count: usize,
) -> Vec<u32> {
    let mut block_token_ids = Vec::with_capacity(speculation_token_count + 1);
    block_token_ids.push(bonus_token_id);
    let mask_token_id_u32 = u32::try_from(mask_token_id).unwrap_or(0);
    for _ in 0..speculation_token_count {
        block_token_ids.push(mask_token_id_u32);
    }
    block_token_ids
}

/// Test-visible assembly mirrors: the production helpers stay private to
/// keep the round's token flow in one place; these thin re-exports exist
/// so the assembly tests lock the exact token orders without graphics
/// machinery.
#[cfg(test)]
pub(super) fn assemble_verification_tokens_for_tests(
    bonus_token_id: u32,
    proposals: &[u32],
) -> Vec<u32> {
    assemble_verification_tokens(bonus_token_id, proposals)
}

#[cfg(test)]
pub(super) fn assemble_replay_tokens_for_tests(
    bonus_token_id: u32,
    proposals: &[u32],
    verification_decision: &GreedyVerificationDecision,
) -> Vec<u32> {
    assemble_replay_tokens(bonus_token_id, proposals, verification_decision)
}

#[cfg(test)]
pub(super) fn assemble_block_token_ids_for_tests(
    bonus_token_id: u32,
    mask_token_id: i32,
    speculation_token_count: usize,
) -> Vec<u32> {
    assemble_block_token_ids(bonus_token_id, mask_token_id, speculation_token_count)
}

fn assemble_verification_tokens(bonus_token_id: u32, proposals: &[u32]) -> Vec<u32> {
    assemble_replay_tokens(
        bonus_token_id,
        proposals,
        &GreedyVerificationDecision {
            accepted_proposal_count: proposals.len(),
            accepted_every_proposal: true,
        },
    )
}

fn assemble_replay_tokens(
    bonus_token_id: u32,
    proposals: &[u32],
    verification_decision: &GreedyVerificationDecision,
) -> Vec<u32> {
    let mut replay_tokens = Vec::with_capacity(verification_decision.accepted_proposal_count + 1);
    replay_tokens.push(bonus_token_id);
    replay_tokens.extend_from_slice(&proposals[..verification_decision.accepted_proposal_count]);
    replay_tokens
}

fn embed_block_tokens(
    target_model: &Qwen3_5ResidentModel,
    block_token_ids: &[u32],
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let signed_token_ids = block_token_ids
        .iter()
        .map(|token_id| {
            i32::try_from(*token_id).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "token ID exceeds the MLX int32 range",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let token_indices = target_model
        .runtime
        .array_from_i32(&signed_token_ids, &[1, block_token_ids.len() as i32])?;
    target_model.embedding_lookup(&token_indices)
}

/// Reads one argmax token per logit row onto the host.
fn argmax_rows_to_host(
    target_model: &Qwen3_5ResidentModel,
    logits: &MlxArray,
) -> Result<Vec<u32>, Qwen3_5ExecutionError> {
    let row_count = logits.shape()[1];
    let row_argmax = target_model.runtime.argmax_axis(logits, -1)?;
    target_model.runtime.evaluate_arrays(&[&row_argmax])?;
    let mut argmax_token_ids = Vec::with_capacity(row_count as usize);
    for row_index in 0..row_count {
        let row_value = target_model.runtime.slice(
            &row_argmax,
            &[0, row_index],
            &[1, row_index + 1],
            &[1, 1],
        )?;
        argmax_token_ids.push(row_value.item_u32()?);
    }
    Ok(argmax_token_ids)
}

/// Slices every tap to its first `prefix_token_count` rows.
fn slice_taps_to_prefix<'a>(
    target_model: &Qwen3_5ResidentModel,
    hidden_state_taps: &'a [MlxArray],
    prefix_token_count: usize,
) -> Result<Vec<MlxArray>, Qwen3_5ExecutionError> {
    let prefix_tokens_i32 =
        i32::try_from(prefix_token_count).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
            description: "accepted tap prefix exceeds the MLX int32 range",
        })?;
    let mut accepted_taps = Vec::with_capacity(hidden_state_taps.len());
    for tap in hidden_state_taps {
        let tap_hidden_size = tap.shape()[2];
        let accepted_tap = target_model.runtime.slice(
            tap,
            &[0, 0, 0],
            &[1, prefix_tokens_i32, tap_hidden_size],
            &[1, 1, 1],
        )?;
        accepted_taps.push(accepted_tap);
    }
    Ok(accepted_taps)
}

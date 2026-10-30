//! The finishing half of one speculative round: truncate the draft's block
//! scratch, condition the draft's context key-value on the target's accepted
//! taps, and compute the round's emitted tokens. Split from
//! `speculation_round` only by the source-size gate; the round's flow is
//! still draft (in `speculation_round`) then finish (here).

use mlx_c_rust::MlxArray;

use std::time::Instant;

use crate::PerformanceOperation;
use crate::performance_attribution::PerformanceAttribution;
use crate::qwen3_5_resident::dflash_draft;
use crate::qwen3_5_resident::model::Qwen3_5ExecutionError;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;

use super::speculation_round::{SpeculationRoundOutcome, SpeculationRoundParticipants};
use super::verification_decisions::{TokenEqualityVerificationDecision, emitted_tokens_for_round};

/// The verification artifacts the finishing half of a round consumes.
pub(super) struct VerificationArtifacts<'a> {
    pub(super) hidden_state_taps: &'a [MlxArray],
    pub(super) proposals: &'a [u32],
    pub(super) target_chosen_token_ids: &'a [u32],
    pub(super) verification_decision: &'a TokenEqualityVerificationDecision,
}

/// Finishes the round, keeping the draft key-value conditioning
/// measurement span whole around the finisher.
pub(super) fn run_measured_accepted_round_finish(
    participants: &SpeculationRoundParticipants<'_>,
    draft_key_value_states: &mut dflash_draft::draft_key_value_state::DFlashDraftKeyValueStates,
    verification_artifacts: &VerificationArtifacts<'_>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<SpeculationRoundOutcome, crate::InferenceEngineError> {
    let conditioning_started_at = Instant::now();
    let finished_round =
        finish_accepted_round(participants, draft_key_value_states, verification_artifacts);
    performance_attribution.record_operation_interval(
        PerformanceOperation::SpeculativeDraftKeyValueConditioning,
        conditioning_started_at,
        Instant::now(),
    );
    finished_round
}

fn finish_accepted_round(
    participants: &SpeculationRoundParticipants<'_>,
    draft_key_value_states: &mut dflash_draft::draft_key_value_state::DFlashDraftKeyValueStates,
    verification_artifacts: &VerificationArtifacts<'_>,
) -> Result<SpeculationRoundOutcome, crate::InferenceEngineError> {
    let accepted_context_growth = super::verification_decisions::accepted_context_growth(
        verification_artifacts.verification_decision,
    );
    // The draft key-value state now holds the discarded block scratch at
    // positions after the context; roll it back before the round ends so
    // the next round starts from the clean context boundary.
    let draft_checkpoint_offset = participants.round_request.context_length_tokens as i32;
    // Discard the draft's block scratch (context entries are re-derived from
    // the target's taps, never the draft's own projections).
    draft_key_value_states
        .truncate_to_offset(draft_checkpoint_offset)
        .map_err(Qwen3_5ExecutionError::from)?;
    // Grow the draft's context key-value from the accepted prefix of the
    // target's taps: rows zero..accepted growth of every tap.
    let accepted_taps = slice_taps_to_prefix(
        participants.target_model,
        verification_artifacts.hidden_state_taps,
        accepted_context_growth,
    )?;
    let conditioned_states =
        dflash_draft::target_hidden_state_conditioning::condition_target_hidden_states(
            &participants.target_model.runtime,
            participants.draft_weights,
            &accepted_taps,
            participants.draft_configuration.rms_norm_epsilon(),
        )
        .map_err(Qwen3_5ExecutionError::from)?;
    dflash_draft::draft_context_kv_projection::append_conditioned_context_key_values(
        &participants.target_model.runtime,
        participants.draft_weights,
        participants.draft_configuration,
        draft_key_value_states,
        &conditioned_states,
        participants.round_request.context_length_tokens as i32,
    )
    .map_err(Qwen3_5ExecutionError::from)?;

    let emitted_token_ids = emitted_tokens_for_round(
        participants.round_request.bonus_token_id,
        verification_artifacts.proposals,
        verification_artifacts.target_chosen_token_ids,
        verification_artifacts.verification_decision,
    );
    let next_bonus_token_id =
        *emitted_token_ids
            .last()
            .ok_or(crate::InferenceEngineError::from(
                Qwen3_5ExecutionError::InvalidInput {
                    description: "a speculative round must emit at least the bonus token",
                },
            ))?;
    Ok(SpeculationRoundOutcome {
        emitted_token_ids,
        next_bonus_token_id,
        accepted_proposal_count: verification_artifacts
            .verification_decision
            .accepted_proposal_count,
        context_length_after_round: participants.round_request.context_length_tokens
            + accepted_context_growth as u32,
    })
}

/// Slices every tap to its first `prefix_token_count` rows.
fn slice_taps_to_prefix(
    target_model: &Qwen3_5ResidentModel,
    hidden_state_taps: &[MlxArray],
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

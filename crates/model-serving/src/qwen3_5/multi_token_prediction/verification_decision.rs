use thiserror::Error;

use crate::memory::{MtpDepthDowngradeReason, MtpDraftDepth};
/// Target-authoritative outcome of one fixed-depth MTP verification window.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MtpVerificationDecision {
    effective_depth: MtpDraftDepth,
    proposed_count: u8,
    accepted_count: u8,
    pending_target_token_id: Option<u32>,
    was_eos_truncated: bool,
    operational_fallback: bool,
}

impl MtpVerificationDecision {
    #[must_use]
    pub const fn effective_depth(&self) -> MtpDraftDepth {
        self.effective_depth
    }

    #[must_use]
    pub const fn proposed_count(&self) -> u8 {
        self.proposed_count
    }

    #[must_use]
    pub const fn accepted_count(&self) -> u8 {
        self.accepted_count
    }

    #[must_use]
    pub const fn pending_target_token_id(&self) -> Option<u32> {
        self.pending_target_token_id
    }

    #[must_use]
    pub const fn was_eos_truncated(&self) -> bool {
        self.was_eos_truncated
    }

    #[must_use]
    pub const fn is_operational_fallback(&self) -> bool {
        self.operational_fallback
    }

    #[cfg(feature = "direct-mlx")]
    pub(crate) const fn operational_fallback(effective_depth: MtpDraftDepth) -> Self {
        Self {
            effective_depth,
            proposed_count: 0,
            accepted_count: 0,
            pending_target_token_id: None,
            was_eos_truncated: false,
            operational_fallback: true,
        }
    }
}

/// Invalid bounded inputs supplied to a pure verifier decision boundary.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MtpVerificationDecisionError {
    #[error("MTP verification vectors do not match the effective draft depth")]
    VectorLengthMismatch,
    #[error("an untruncated sampled verification window requires the residual or bonus token")]
    MissingPostPrefixToken,
}

/// Speculative acceptance `min(1, p/q)` for one drafted token.
///
/// `p` is the target distribution's probability of the drafted token and `q` is
/// the draft distribution's probability of emitting it. The ratio caps at one so
/// a target that favors the draft never adds rejection cost, and a token the
/// draft distribution could not have produced is an automatic rejection.
#[must_use]
pub fn qwen3_5_mtp_sampled_acceptance_probability(
    target_token_probability: f32,
    draft_token_probability: f32,
) -> f32 {
    if draft_token_probability <= 0.0 || !draft_token_probability.is_finite() {
        return 0.0;
    }
    if !target_token_probability.is_finite() || target_token_probability >= draft_token_probability
    {
        return 1.0;
    }
    target_token_probability / draft_token_probability
}

/// Computes longest-prefix acceptance and truncates accepted output at the first EOS.
pub fn qwen3_5_mtp_verification_decision(
    effective_depth: MtpDraftDepth,
    draft_token_ids: &[u32],
    target_token_ids: &[u32],
    end_of_sequence_token_ids: &[u32],
) -> Result<MtpVerificationDecision, MtpVerificationDecisionError> {
    let proposed_count = effective_depth.get();
    if draft_token_ids.len() != usize::from(proposed_count)
        || target_token_ids.len() != usize::from(proposed_count) + 1
    {
        return Err(MtpVerificationDecisionError::VectorLengthMismatch);
    }

    let matched_prefix_count = draft_token_ids
        .iter()
        .zip(target_token_ids)
        .take_while(|(draft_token_id, target_token_id)| draft_token_id == target_token_id)
        .count();
    let coin_accepted_prefix_count = matched_prefix_count;
    let first_accepted_eos_position = draft_token_ids[..coin_accepted_prefix_count]
        .iter()
        .position(|draft_token_id| end_of_sequence_token_ids.contains(draft_token_id));
    let accepted_count = first_accepted_eos_position
        .map_or(coin_accepted_prefix_count, |eos_position| eos_position + 1);
    let was_eos_truncated = first_accepted_eos_position.is_some();
    let pending_target_token_id = if was_eos_truncated {
        None
    } else {
        target_token_ids.get(accepted_count).copied()
    };

    Ok(MtpVerificationDecision {
        effective_depth,
        proposed_count,
        accepted_count: accepted_count as u8,
        pending_target_token_id,
        was_eos_truncated,
        operational_fallback: false,
    })
}

/// Coin-backed sampled verification of one fixed-depth MTP window.
///
/// `accepted_coin_flags` carries one acceptance outcome per drafted token in
/// draft order; `post_prefix_token_id` is the residual re-sample after a
/// rejection or the sampled bonus after a fully accepted window. Accepted-prefix
/// and EOS-truncation semantics mirror [`qwen3_5_mtp_verification_decision`]
/// exactly, so both decoding modes share one commit and emission path.
///
/// The distribution-equivalence guarantee of speculative sampling is preserved
/// because the acceptance coin at every position is drawn as Bernoulli of
/// [`qwen3_5_mtp_sampled_acceptance_probability`] and a rejection resamples the
/// emitted token from mass proportional to `max(0, p − q)` over the same
/// position distribution.
#[must_use]
pub fn qwen3_5_mtp_sampled_verification_decision(
    effective_depth: MtpDraftDepth,
    draft_token_ids: &[u32],
    accepted_coin_flags: &[bool],
    post_prefix_token_id: Option<u32>,
    end_of_sequence_token_ids: &[u32],
) -> Result<MtpVerificationDecision, MtpVerificationDecisionError> {
    let proposed_count = effective_depth.get();
    if draft_token_ids.len() != usize::from(proposed_count)
        || accepted_coin_flags.len() != usize::from(proposed_count)
    {
        return Err(MtpVerificationDecisionError::VectorLengthMismatch);
    }
    let post_prefix_token_id =
        post_prefix_token_id.ok_or(MtpVerificationDecisionError::MissingPostPrefixToken)?;

    let coin_accepted_prefix_count = accepted_coin_flags
        .iter()
        .take_while(|accepted| **accepted)
        .count();
    let first_accepted_eos_position = draft_token_ids[..coin_accepted_prefix_count]
        .iter()
        .position(|draft_token_id| end_of_sequence_token_ids.contains(draft_token_id));
    let (accepted_count, was_eos_truncated, pending_target_token_id) =
        match first_accepted_eos_position {
            Some(eos_position) => (eos_position + 1, true, None),
            None => (
                coin_accepted_prefix_count,
                false,
                Some(post_prefix_token_id),
            ),
        };

    Ok(MtpVerificationDecision {
        effective_depth,
        proposed_count,
        accepted_count: accepted_count as u8,
        pending_target_token_id,
        was_eos_truncated,
        operational_fallback: false,
    })
}

/// Clamps one resolved depth before any target or predictor state is mutated.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn qwen3_5_mtp_effective_depth_for_windows(
    requested_depth: MtpDraftDepth,
    generated_token_count: u16,
    maximum_output_tokens: u16,
    next_position_tokens: u32,
    maximum_position_count: usize,
    is_inside_thinking: bool,
    thinking_token_count: u16,
    thinking_budget: Option<u16>,
) -> Option<MtpDraftDepth> {
    qwen3_5_mtp_effective_depth_and_reason_for_windows(
        requested_depth,
        generated_token_count,
        maximum_output_tokens,
        next_position_tokens,
        maximum_position_count,
        is_inside_thinking,
        thinking_token_count,
        thinking_budget,
    )
    .0
}

/// Resolves the bounded request depth and attributes the first limiting user-visible window.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn qwen3_5_mtp_effective_depth_and_reason_for_windows(
    requested_depth: MtpDraftDepth,
    generated_token_count: u16,
    maximum_output_tokens: u16,
    next_position_tokens: u32,
    maximum_position_count: usize,
    is_inside_thinking: bool,
    thinking_token_count: u16,
    thinking_budget: Option<u16>,
) -> (Option<MtpDraftDepth>, Option<MtpDepthDowngradeReason>) {
    let remaining_output_tokens = maximum_output_tokens.saturating_sub(generated_token_count);
    let remaining_context_tokens =
        maximum_position_count.saturating_sub(next_position_tokens as usize);
    let remaining_thinking_tokens = if is_inside_thinking {
        thinking_budget.map_or(u16::MAX, |budget| {
            budget.saturating_sub(thinking_token_count)
        })
    } else {
        u16::MAX
    };
    // Every verification window can publish D accepted drafts plus one target correction/bonus,
    // so each independent window must reserve one non-draft token before depths are compared.
    let output_draft_capacity = usize::from(remaining_output_tokens).saturating_sub(1);
    let context_draft_capacity = remaining_context_tokens.saturating_sub(1);
    let thinking_draft_capacity = usize::from(remaining_thinking_tokens).saturating_sub(1);
    let requested_draft_count = usize::from(requested_depth.get());
    let feasible_draft_count = requested_draft_count
        .min(output_draft_capacity)
        .min(context_draft_capacity)
        .min(thinking_draft_capacity);
    let effective_depth = u8::try_from(feasible_draft_count)
        .ok()
        .and_then(|depth| MtpDraftDepth::new(depth).ok());
    let downgrade_reason = if feasible_draft_count == requested_draft_count {
        None
    } else if output_draft_capacity == feasible_draft_count {
        Some(MtpDepthDowngradeReason::OutputWindow)
    } else if context_draft_capacity == feasible_draft_count {
        Some(MtpDepthDowngradeReason::ContextWindow)
    } else {
        Some(MtpDepthDowngradeReason::ThinkingWindow)
    };
    (effective_depth, downgrade_reason)
}

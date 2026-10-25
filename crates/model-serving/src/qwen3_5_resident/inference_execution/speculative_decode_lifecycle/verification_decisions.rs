//! Pure acceptance decisions for speculative verification.
//!
//! Everything here is values in, values out: no arrays, no runtime, no
//! state. The stochastic sibling (GPU acceptance coins plus residual
//! correction) already exists in the sampling module; this file carries the
//! greedy contract the proof harness measures and the boundary rules every
//! mode shares.

/// The outcome of one greedy verification decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GreedyVerificationDecision {
    /// How many of the drafted proposals the target confirmed, counting from
    /// the first: `accepted_proposal_count == proposals.len()` means the
    /// target agreed with the whole draft.
    pub accepted_proposal_count: usize,
    /// Whether the target confirmed every proposal. On full acceptance the
    /// verifier's own next-token preference extends the block at zero cost.
    pub accepted_every_proposal: bool,
}

/// Decides how many drafted proposals a greedy target confirms.
///
/// `target_preferred_token_ids[i]` is the target's argmax token for the slot
/// proposal `i` occupies (the row the verifier produced one position before
/// that proposal). A proposal is confirmed when it equals the target's
/// preference; the count is the longest confirmed prefix.
pub fn decide_greedy_verification(
    proposals: &[u32],
    target_preferred_token_ids: &[u32],
) -> GreedyVerificationDecision {
    let comparable_slot_count = proposals.len().min(target_preferred_token_ids.len());
    let mut accepted_proposal_count = 0_usize;
    for proposal_slot in 0..comparable_slot_count {
        if proposals[proposal_slot] == target_preferred_token_ids[proposal_slot] {
            accepted_proposal_count = proposal_slot + 1;
        } else {
            break;
        }
    }
    let accepted_every_proposal = accepted_proposal_count == proposals.len()
        && target_preferred_token_ids.len() >= proposals.len();
    GreedyVerificationDecision {
        accepted_proposal_count,
        accepted_every_proposal,
    }
}

/// How many context entries the verifier keeps after one round: the bonus
/// token plus every accepted proposal. Rejected proposals and (on rejection)
/// the verifier's correction token are not part of the kept context.
#[must_use]
pub fn accepted_context_growth(decision: &GreedyVerificationDecision) -> usize {
    decision.accepted_proposal_count + 1
}

/// The tokens one round emits under greedy verification.
///
/// On full acceptance: bonus plus every proposal — the verifier's next
/// preference becomes the next round's bonus outside this function. On
/// rejection: bonus plus the accepted prefix plus the verifier's correction
/// token (the target's preferred token at the first rejected slot), which
/// the caller must also use as the next round's bonus.
#[must_use]
pub fn emitted_tokens_for_round(
    bonus_token_id: u32,
    proposals: &[u32],
    target_preferred_token_ids: &[u32],
    decision: &GreedyVerificationDecision,
) -> Vec<u32> {
    let mut emitted_tokens = Vec::with_capacity(proposals.len() + 2);
    emitted_tokens.push(bonus_token_id);
    emitted_tokens.extend_from_slice(&proposals[..decision.accepted_proposal_count]);
    if !decision.accepted_every_proposal
        && decision.accepted_proposal_count < target_preferred_token_ids.len()
    {
        emitted_tokens.push(target_preferred_token_ids[decision.accepted_proposal_count]);
    }
    emitted_tokens
}

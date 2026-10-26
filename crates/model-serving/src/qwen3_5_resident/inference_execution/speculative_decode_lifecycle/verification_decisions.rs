//! Pure acceptance decisions for speculative verification.
//!
//! Everything here is values in, values out: no arrays, no runtime, no
//! state. The acceptance rule is token equality against the target's
//! chosen token per position — argmax under greedy decoding, one tempered
//! sample under sampled decoding — so the same rule serves both modes and
//! every emitted token is always a token the target itself chose. The
//! stochastic Leviathan sibling (GPU acceptance coins plus residual
//! correction) already exists in the sampling module for a future
//! draft-sampling contract; this file carries the token-equality contract
//! the proof harness measures and the boundary rules every mode shares.

/// How the target chooses one token per verification position.
///
/// The draft always proposes greedily: under token-equality acceptance the
/// per-position acceptance probability is the probability the target's
/// choice equals the draft's proposal, `sum(q(x) * p(x))`, which the
/// degenerate argmax proposal distribution maximizes at `p(argmax)` —
/// sampling the draft strictly lowers acceptance. The target's choice is
/// what carries the user's sampling semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationTokenChoice {
    /// The target's argmax per position: the greedy contract. Emitted
    /// tokens are exactly the classic greedy decode.
    HighestLogit,
    /// One independent tempered sample per position through the same
    /// temperature, top-k, and top-p transform the classic decode path
    /// applies. Every emitted token is the target's own sample — accepted
    /// proposals merely equal it — so the output law is exactly the
    /// classic sampled law at the requested temperature.
    TopKTopP {
        temperature_thousandths: u16,
        top_k: u16,
        top_p_thousandths: u16,
    },
}

impl VerificationTokenChoice {
    /// Whether this choice needs the request's randomness stream.
    #[must_use]
    pub fn consumes_random_state(&self) -> bool {
        matches!(self, Self::TopKTopP { .. })
    }
}

/// The outcome of one token-equality verification decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenEqualityVerificationDecision {
    /// How many of the drafted proposals the target confirmed, counting from
    /// the first: `accepted_proposal_count == proposals.len()` means the
    /// target's choice agreed with the whole draft.
    pub accepted_proposal_count: usize,
    /// Whether the target's choice confirmed every proposal. On full
    /// acceptance the verifier's own next-token choice extends the block
    /// at zero cost.
    pub accepted_every_proposal: bool,
}

/// Decides how many drafted proposals the target's choices confirm.
///
/// `target_chosen_token_ids[i]` is the target's chosen token for the slot
/// proposal `i` occupies (the row the verifier produced one position before
/// that proposal) — its argmax under greedy decoding or its tempered sample
/// under sampled decoding. A proposal is confirmed when it equals the
/// target's choice; the count is the longest confirmed prefix. Under either
/// mode the first divergence emits the target's own choice, so the draft's
/// distribution never enters the output law.
pub fn decide_token_equality_verification(
    proposals: &[u32],
    target_chosen_token_ids: &[u32],
) -> TokenEqualityVerificationDecision {
    let comparable_slot_count = proposals.len().min(target_chosen_token_ids.len());
    let mut accepted_proposal_count = 0_usize;
    for proposal_slot in 0..comparable_slot_count {
        if proposals[proposal_slot] == target_chosen_token_ids[proposal_slot] {
            accepted_proposal_count = proposal_slot + 1;
        } else {
            break;
        }
    }
    let accepted_every_proposal = accepted_proposal_count == proposals.len()
        && target_chosen_token_ids.len() >= proposals.len();
    TokenEqualityVerificationDecision {
        accepted_proposal_count,
        accepted_every_proposal,
    }
}

/// How many context entries the verifier keeps after one round: the bonus
/// token plus every accepted proposal. Rejected proposals and (on rejection)
/// the verifier's correction token are not part of the kept context.
#[must_use]
pub fn accepted_context_growth(decision: &TokenEqualityVerificationDecision) -> usize {
    decision.accepted_proposal_count + 1
}

/// The tokens one round emits under token-equality verification.
///
/// On full acceptance: bonus plus every proposal — the verifier's next
/// choice becomes the next round's bonus outside this function. On
/// rejection: bonus plus the accepted prefix plus the verifier's correction
/// token (the target's chosen token at the first rejected slot), which the
/// caller must also use as the next round's bonus. In every case each
/// emitted token beyond the bonus equals a token the target itself chose.
#[must_use]
pub fn emitted_tokens_for_round(
    bonus_token_id: u32,
    proposals: &[u32],
    target_chosen_token_ids: &[u32],
    decision: &TokenEqualityVerificationDecision,
) -> Vec<u32> {
    let mut emitted_tokens = Vec::with_capacity(proposals.len() + 2);
    emitted_tokens.push(bonus_token_id);
    emitted_tokens.extend_from_slice(&proposals[..decision.accepted_proposal_count]);
    if !decision.accepted_every_proposal
        && decision.accepted_proposal_count < target_chosen_token_ids.len()
    {
        emitted_tokens.push(target_chosen_token_ids[decision.accepted_proposal_count]);
    }
    emitted_tokens
}

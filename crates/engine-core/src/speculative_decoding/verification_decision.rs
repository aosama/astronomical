//! Which drafted tokens the target keeps, and the token it adds after them.
//!
//! Greedy decoding keeps drafts while they equal the target's argmax. Sampled
//! decoding uses speculative rejection sampling (Leviathan et al. 2023, as in
//! z-lab's reference): a draft token `x` drawn from the drafter's `q` is kept
//! with probability `min(1, p(x) / q(x))`, and on the first refusal the target
//! samples from the leftover mass `max(0, p - q)`. Every emitted token is then
//! distributed exactly as the target's own sampler would have chosen it, which
//! is what makes drafting lossless at temperature 0.7 or 1.0.
//!
//! This module is pure host arithmetic over the few probabilities read back:
//! the uniform random draws come in as arguments, so a decision is a
//! deterministic function of its inputs and is unit-tested without a GPU.

/// One drafted token and the drafter's distribution it was drawn from. DFlash2
/// samples among a slot's kept candidates only, so `q` is sparse.
pub struct DraftedSlot<'slot> {
    pub token_id: u32,
    pub candidate_ids: &'slot [u32],
    /// `q` over `candidate_ids`, in the same order; sums to one.
    pub candidate_probabilities: &'slot [f32],
}

/// The verdict of one verify pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationDecision {
    /// Leading drafts the target keeps.
    pub kept_draft_count: usize,
    /// The target's own token after the kept drafts: a correction where a
    /// draft was refused, or the next token when every draft was kept.
    pub next_token_id: u32,
}

/// Greedy: keep drafts while they equal the target's argmax at the same row.
/// `target_argmax_ids` has one more row than `drafted_ids`.
pub fn decide_by_token_equality(
    drafted_ids: &[u32],
    target_argmax_ids: &[u32],
) -> VerificationDecision {
    let mut kept_draft_count = 0;
    while kept_draft_count < drafted_ids.len()
        && drafted_ids[kept_draft_count] == target_argmax_ids[kept_draft_count]
    {
        kept_draft_count += 1;
    }
    VerificationDecision {
        kept_draft_count,
        next_token_id: target_argmax_ids[kept_draft_count],
    }
}

/// The target's sampling distribution at one verified row, as the token ids
/// its top-k cut keeps and their probabilities after top-p and temperature.
/// Every other token has probability zero under that law, so a handful of
/// entries per row is exact and only those are read back from the GPU.
pub struct TargetDistribution<'row> {
    pub token_ids: &'row [u32],
    pub probabilities: &'row [f32],
}

impl TargetDistribution<'_> {
    fn probability_of(&self, token_id: u32) -> f64 {
        let mut probability = 0.0;
        for (candidate_id, candidate_probability) in self.token_ids.iter().zip(self.probabilities) {
            if *candidate_id == token_id {
                probability += f64::from(*candidate_probability);
            }
        }
        probability
    }
}

/// Sampled: rejection sampling against the target's distribution at each
/// verified row (drafts plus one). `uniform_draws` needs one value in [0, 1)
/// per draft plus one.
pub fn decide_by_rejection_sampling(
    drafted_slots: &[DraftedSlot<'_>],
    target_distributions: &[TargetDistribution<'_>],
    uniform_draws: &[f64],
) -> VerificationDecision {
    for (slot_index, drafted_slot) in drafted_slots.iter().enumerate() {
        let target_distribution = &target_distributions[slot_index];
        let target_probability = target_distribution.probability_of(drafted_slot.token_id);
        let drafter_probability = sparse_probability(drafted_slot, drafted_slot.token_id);
        // Kept when u * q < p: the reference's form of u < min(1, p / q).
        if uniform_draws[slot_index] * drafter_probability < target_probability {
            continue;
        }
        let leftover = leftover_weights(target_distribution, drafted_slot);
        return VerificationDecision {
            kept_draft_count: slot_index,
            next_token_id: sample_from(
                target_distribution.token_ids,
                &leftover,
                uniform_draws[drafted_slots.len()],
            ),
        };
    }
    let final_distribution = &target_distributions[drafted_slots.len()];
    let final_weights: Vec<f64> = final_distribution
        .probabilities
        .iter()
        .map(|value| f64::from(*value))
        .collect();
    VerificationDecision {
        kept_draft_count: drafted_slots.len(),
        next_token_id: sample_from(
            final_distribution.token_ids,
            &final_weights,
            uniform_draws[drafted_slots.len()],
        ),
    }
}

fn sparse_probability(drafted_slot: &DraftedSlot<'_>, token_id: u32) -> f64 {
    let mut probability = 0.0;
    for (candidate_id, candidate_probability) in drafted_slot
        .candidate_ids
        .iter()
        .zip(drafted_slot.candidate_probabilities)
    {
        if *candidate_id == token_id {
            probability += f64::from(*candidate_probability);
        }
    }
    probability
}

/// `max(0, p - q)` over the target's tokens (elsewhere p is zero, so the
/// leftover is too); the target's own weights when nothing is left (q covers
/// p everywhere, which a refusal makes vanishingly rare).
fn leftover_weights(
    target_distribution: &TargetDistribution<'_>,
    drafted_slot: &DraftedSlot<'_>,
) -> Vec<f64> {
    let mut leftover = Vec::with_capacity(target_distribution.token_ids.len());
    for (token_id, probability) in target_distribution
        .token_ids
        .iter()
        .zip(target_distribution.probabilities)
    {
        leftover
            .push((f64::from(*probability) - sparse_probability(drafted_slot, *token_id)).max(0.0));
    }
    if leftover.iter().sum::<f64>() <= 0.0 {
        return target_distribution
            .probabilities
            .iter()
            .map(|value| f64::from(*value))
            .collect();
    }
    leftover
}

/// Inverse-CDF draw of one of `token_ids` from its unnormalized `weights`
/// with one uniform value.
fn sample_from(token_ids: &[u32], weights: &[f64], uniform_draw: f64) -> u32 {
    let total: f64 = weights.iter().sum();
    let threshold = uniform_draw * total;
    let mut cumulative = 0.0;
    let mut last_positive_index = 0;
    for (entry_index, weight) in weights.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        cumulative += weight;
        last_positive_index = entry_index;
        if threshold < cumulative {
            return token_ids[entry_index];
        }
    }
    // Rounding can leave the threshold a hair above the final cumulative sum.
    token_ids[last_positive_index]
}

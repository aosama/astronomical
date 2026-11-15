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
//! This module is pure host arithmetic over probabilities already read back:
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

/// Sampled: rejection sampling against the target's tempered, top-k and top-p
/// masked probabilities. `target_probability_rows` holds one row per verified
/// position (drafts plus one), each over the whole vocabulary.
/// `uniform_draws` needs one value in [0, 1) per draft plus one.
pub fn decide_by_rejection_sampling(
    drafted_slots: &[DraftedSlot<'_>],
    target_probability_rows: &[&[f32]],
    uniform_draws: &[f64],
) -> VerificationDecision {
    for (slot_index, drafted_slot) in drafted_slots.iter().enumerate() {
        let target_row = target_probability_rows[slot_index];
        let target_probability = f64::from(target_row[drafted_slot.token_id as usize]);
        let drafter_probability = sparse_probability(drafted_slot, drafted_slot.token_id);
        // Kept when u * q < p: the reference's form of u < min(1, p / q).
        if uniform_draws[slot_index] * drafter_probability < target_probability {
            continue;
        }
        let leftover = leftover_distribution(target_row, drafted_slot);
        return VerificationDecision {
            kept_draft_count: slot_index,
            next_token_id: sample_from(&leftover, uniform_draws[drafted_slots.len()]),
        };
    }
    let final_row: Vec<f64> = target_probability_rows[drafted_slots.len()]
        .iter()
        .map(|value| f64::from(*value))
        .collect();
    VerificationDecision {
        kept_draft_count: drafted_slots.len(),
        next_token_id: sample_from(&final_row, uniform_draws[drafted_slots.len()]),
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

/// `max(0, p - q)` normalized; the target row itself when nothing is left
/// (q covers p everywhere, which a refusal makes vanishingly rare).
fn leftover_distribution(target_row: &[f32], drafted_slot: &DraftedSlot<'_>) -> Vec<f64> {
    let mut leftover: Vec<f64> = target_row.iter().map(|value| f64::from(*value)).collect();
    for (candidate_id, candidate_probability) in drafted_slot
        .candidate_ids
        .iter()
        .zip(drafted_slot.candidate_probabilities)
    {
        let entry = &mut leftover[*candidate_id as usize];
        *entry = (*entry - f64::from(*candidate_probability)).max(0.0);
    }
    let leftover_mass: f64 = leftover.iter().sum();
    if leftover_mass <= 0.0 {
        return target_row.iter().map(|value| f64::from(*value)).collect();
    }
    leftover
}

/// Inverse-CDF draw from unnormalized `weights` with one uniform value.
fn sample_from(weights: &[f64], uniform_draw: f64) -> u32 {
    let total: f64 = weights.iter().sum();
    let threshold = uniform_draw * total;
    let mut cumulative = 0.0;
    let mut last_positive_index = 0;
    for (token_index, weight) in weights.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        cumulative += weight;
        last_positive_index = token_index;
        if threshold < cumulative {
            return token_index as u32;
        }
    }
    // Rounding can leave the threshold a hair above the final cumulative sum.
    last_positive_index as u32
}

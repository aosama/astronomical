//! Predictor-driven warm-table retention (issue #539).
//!
//! The native router still decides what to execute. The predictor only ranks
//! which leftover experts the warm table should refuse to evict. Wrong
//! predictions therefore cannot displace a page the current route needs:
//! predicted identifiers join the protected set, and a miss that does not fit
//! leftover slots is dropped.

/// Returns the `top_k` expert identifiers with the highest logits, highest
/// first. Ties keep the lower identifier so the ranking is deterministic.
#[must_use]
pub fn select_top_expert_ids_from_logits(logits: &[f32], top_k: usize) -> Vec<usize> {
    if top_k == 0 || logits.is_empty() {
        return Vec::new();
    }
    let mut expert_id_order: Vec<usize> = (0..logits.len()).collect();
    expert_id_order.sort_unstable_by(|left, right| {
        logits[*right]
            .partial_cmp(&logits[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    expert_id_order.truncate(top_k.min(logits.len()));
    expert_id_order
}

/// Merges predicted expert identifiers into the insert's protected set.
///
/// The result is sorted and unique so the insert's binary search stays valid.
/// Predicted identifiers that are already protected are not duplicated.
#[must_use]
pub fn merge_predicted_experts_into_protected_set(
    protected_expert_ids: &[usize],
    predicted_expert_ids: &[usize],
) -> Vec<usize> {
    let mut merged = Vec::with_capacity(protected_expert_ids.len() + predicted_expert_ids.len());
    merged.extend_from_slice(protected_expert_ids);
    merged.extend_from_slice(predicted_expert_ids);
    merged.sort_unstable();
    merged.dedup();
    merged
}

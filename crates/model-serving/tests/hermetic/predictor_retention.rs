//! Predictor retention-hint ranking: leftover protection, never eviction.

use astronomical_model_serving::{
    merge_predicted_experts_into_protected_set, select_top_expert_ids_from_logits,
};

#[test]
fn selects_the_highest_logits_and_breaks_ties_by_lower_expert_id() {
    let logits = [0.1_f32, 0.9, 0.9, 0.2];
    assert_eq!(select_top_expert_ids_from_logits(&logits, 2), vec![1, 2]);
}

#[test]
fn empty_or_zero_budget_selects_nothing() {
    assert!(select_top_expert_ids_from_logits(&[], 3).is_empty());
    assert!(select_top_expert_ids_from_logits(&[1.0, 2.0], 0).is_empty());
}

#[test]
fn truncates_to_the_logit_count_when_the_budget_is_larger() {
    let logits = [0.4_f32, 0.1];
    assert_eq!(select_top_expert_ids_from_logits(&logits, 8), vec![0, 1]);
}

#[test]
fn merging_predicted_experts_into_the_protected_set_never_drops_a_demanded_id() {
    let protected_expert_ids = [3_usize, 1];
    let predicted_expert_ids = [4_usize, 1, 5];
    let merged =
        merge_predicted_experts_into_protected_set(&protected_expert_ids, &predicted_expert_ids);
    assert_eq!(merged, vec![1, 3, 4, 5]);
    assert!(
        merged.binary_search(&3).is_ok(),
        "a demanded page must stay protected after the predictor hint is merged"
    );
}

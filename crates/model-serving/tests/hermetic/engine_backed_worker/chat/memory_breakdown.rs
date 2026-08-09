use super::*;

#[test]
fn should_reconcile_shared_payload_views_to_the_measured_mlx_active_total() {
    let memory_breakdown = MlxActiveMemoryBreakdown::reconcile(100, 60, 30, 20);

    assert_eq!(memory_breakdown.expert_payload_bytes, 60);
    assert_eq!(memory_breakdown.model_core_payload_bytes, 30);
    assert_eq!(memory_breakdown.context_state_payload_bytes, 10);
    assert_eq!(memory_breakdown.speculative_prefill_draft_memory_bytes, 0);
    assert_eq!(
        memory_breakdown
            .expert_payload_bytes
            .saturating_add(memory_breakdown.model_core_payload_bytes)
            .saturating_add(memory_breakdown.context_state_payload_bytes),
        100
    );
}

#[test]
fn should_assign_every_remaining_draft_scoring_byte_to_the_standalone_drafter() {
    let memory_breakdown =
        MlxActiveMemoryBreakdown::reconcile_with_speculative_prefill_draft(100, 20, 30, 10, 25);

    assert_eq!(memory_breakdown.expert_payload_bytes, 20);
    assert_eq!(memory_breakdown.model_core_payload_bytes, 30);
    assert_eq!(memory_breakdown.context_state_payload_bytes, 10);
    assert_eq!(memory_breakdown.speculative_prefill_draft_memory_bytes, 40);
    assert_eq!(
        memory_breakdown
            .expert_payload_bytes
            .saturating_add(memory_breakdown.model_core_payload_bytes)
            .saturating_add(memory_breakdown.context_state_payload_bytes)
            .saturating_add(memory_breakdown.speculative_prefill_draft_memory_bytes),
        100
    );
}

#[test]
fn should_clamp_target_owners_before_assigning_remaining_memory_to_the_drafter() {
    let memory_breakdown =
        MlxActiveMemoryBreakdown::reconcile_with_speculative_prefill_draft(100, 80, 40, 10, 50);

    assert_eq!(memory_breakdown.expert_payload_bytes, 80);
    assert_eq!(memory_breakdown.model_core_payload_bytes, 20);
    assert_eq!(memory_breakdown.context_state_payload_bytes, 0);
    assert_eq!(memory_breakdown.speculative_prefill_draft_memory_bytes, 0);
}

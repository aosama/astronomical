use super::*;

#[test]
fn should_reconcile_shared_payload_views_to_the_measured_mlx_active_total() {
    let memory_breakdown = MlxActiveMemoryBreakdown::reconcile(100, 60, 30, 20);

    assert_eq!(memory_breakdown.expert_payload_bytes, 60);
    assert_eq!(memory_breakdown.model_core_payload_bytes, 30);
    assert_eq!(memory_breakdown.context_state_payload_bytes, 10);
    assert_eq!(
        memory_breakdown
            .expert_payload_bytes
            .saturating_add(memory_breakdown.model_core_payload_bytes)
            .saturating_add(memory_breakdown.context_state_payload_bytes),
        100
    );
}

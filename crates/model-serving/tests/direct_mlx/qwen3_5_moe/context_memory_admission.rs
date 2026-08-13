use astronomical_model_serving::ContextAdmissionRequirements;

#[test]
fn should_reserve_the_maximum_routed_expert_page_during_request_admission() {
    let active_memory_bytes_before_expert_page = 9_906_269_322;
    let maximum_routed_expert_page_bytes = 299_040_768;

    let projected_active_memory_bytes = ContextAdmissionRequirements {
        current_active_memory_bytes: active_memory_bytes_before_expert_page,
        context_growth_bytes: 0,
        expert_page_reservation_bytes: maximum_routed_expert_page_bytes,
        temporary_workspace_bytes: 0,
        retained_expert_payload_bytes: 0,
        active_memory_ceiling_bytes: 10_000_000_000,
        complete_experts_are_resident: false,
    }
    .projected_active_memory_bytes();

    assert_eq!(projected_active_memory_bytes, Some(10_205_310_090));
    assert!(projected_active_memory_bytes > Some(10_000_000_000));
}

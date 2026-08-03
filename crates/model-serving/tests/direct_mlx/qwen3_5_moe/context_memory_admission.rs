use astronomical_model_serving::context_memory_admission_projected_active_memory_bytes;

#[test]
fn should_reserve_the_maximum_routed_expert_page_during_request_admission() {
    let active_memory_bytes_before_expert_page = 9_906_269_322;
    let maximum_routed_expert_page_bytes = 299_040_768;

    let projected_active_memory_bytes = context_memory_admission_projected_active_memory_bytes(
        active_memory_bytes_before_expert_page,
        0,
        maximum_routed_expert_page_bytes,
    );

    assert_eq!(projected_active_memory_bytes, Some(10_205_310_090));
    assert!(projected_active_memory_bytes > Some(10_000_000_000));
}

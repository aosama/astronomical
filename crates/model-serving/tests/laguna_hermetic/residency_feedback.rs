use astronomical_model_serving::{
    MlxRamBudget, MlxRamBudgetModelGeometry, laguna_retained_expert_budget_after_completed_forward,
};

#[test]
fn should_admit_complete_laguna_experts_when_measured_forward_work_fits() {
    let mlx_ram_budget = test_ram_budget(1_000_000_000, 650_000_000);

    let retained_expert_budget = laguna_retained_expert_budget_after_completed_forward(
        &mlx_ram_budget,
        600_000_000,
        650_000_000,
        400_000_000,
        650_000_000,
    );

    assert_eq!(retained_expert_budget, 650_000_000);
}

#[test]
fn should_reduce_laguna_expert_retention_when_measured_forward_work_exceeds_the_ceiling() {
    let mlx_ram_budget = test_ram_budget(1_000_000_000, 800_000_000);

    let retained_expert_budget = laguna_retained_expert_budget_after_completed_forward(
        &mlx_ram_budget,
        950_000_000,
        1_056_000_000,
        700_000_000,
        800_000_000,
    );

    assert_eq!(retained_expert_budget, 580_000_000);
}

#[test]
fn should_keep_laguna_request_finalization_cleanup_in_the_execution_path() {
    let execution_source = include_str!("../../src/laguna/engine/execution.rs");

    assert!(execution_source.contains("synchronize_gpu_stream_and_clear_allocator_cache"));
}

fn test_ram_budget(
    mlx_memory_ceiling_bytes: u64,
    complete_expert_payload_bytes: u64,
) -> MlxRamBudget {
    MlxRamBudget::with_bootstrap_context_window_reserve_bytes(
        mlx_memory_ceiling_bytes,
        MlxRamBudgetModelGeometry {
            model_core_payload_bytes: 100_000_000,
            complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes: 10_000_000,
            largest_routed_expert_page_bytes: 1_000_000,
        },
        100_000_000,
    )
    .expect("the Laguna memory test budget should be valid")
}

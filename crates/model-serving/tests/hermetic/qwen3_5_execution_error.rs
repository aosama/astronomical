use astronomical_model_serving::{
    ExpertPagingError, MemoryBoundary, MlxAllocationAdmissionError, Qwen3_5ExecutionError,
};
use astronomical_runtime_integration::MlxRuntimeError;

#[test]
fn should_classify_direct_and_paged_native_capacity_errors_as_the_same_recoverable_pressure() {
    let expected_capacity_evidence = (9_900_000_000, 300_000_000, 10_100_000_000);
    let direct_capacity_error =
        Qwen3_5ExecutionError::from(MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes: expected_capacity_evidence.0,
            attempted_allocation_bytes: expected_capacity_evidence.1,
            allowed_active_memory_bytes: expected_capacity_evidence.2,
        });
    let paged_capacity_error = Qwen3_5ExecutionError::from(ExpertPagingError::NativeRuntime(
        MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes: expected_capacity_evidence.0,
            attempted_allocation_bytes: expected_capacity_evidence.1,
            allowed_active_memory_bytes: expected_capacity_evidence.2,
        },
    ));

    assert_eq!(
        direct_capacity_error.active_memory_limit_exceeded_evidence(),
        Some(expected_capacity_evidence)
    );
    assert_eq!(
        paged_capacity_error.active_memory_limit_exceeded_evidence(),
        Some(expected_capacity_evidence),
        "paged expert loading must enter the same checkpoint, reclamation, and chunk-reduction recovery path as direct MLX execution"
    );
}

#[test]
fn should_classify_rust_expert_budget_rejection_as_recoverable_capacity_pressure() {
    let expected_capacity_evidence = (29_151_197_454, 855_638_016, 30_000_000_000);
    let paged_budget_error = Qwen3_5ExecutionError::from(ExpertPagingError::MemoryBudget(
        MlxAllocationAdmissionError::Rejected {
            stage: "rust_streamed_expert_layer_39".to_owned(),
            boundary: MemoryBoundary::AllocationProjection,
            shortfall_bytes: 6_835_470,
            active_memory_bytes: expected_capacity_evidence.0 as u64,
            pending_allocation_bytes: expected_capacity_evidence.1 as u64,
            active_memory_ceiling_bytes: expected_capacity_evidence.2 as u64,
        },
    ));

    assert_eq!(
        paged_budget_error.active_memory_limit_exceeded_evidence(),
        Some(expected_capacity_evidence),
        "Rust expert-budget rejection must enter checkpoint restoration and reclamation"
    );
}

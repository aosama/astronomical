use astronomical_model_serving::{
    ExpertPagingError, InferenceEngineError, MemoryBoundary, MlxAllocationAdmissionError,
    Qwen3_5ExecutionError,
};
use astronomical_runtime_integration::MlxRuntimeError;

#[test]
fn should_translate_expert_page_memory_budget_rejection_to_an_invalid_request() {
    let qwen3_5_execution_error = Qwen3_5ExecutionError::from(ExpertPagingError::MemoryBudget(
        MlxAllocationAdmissionError::Rejected {
            stage: "expert_page_layer_20".to_owned(),
            boundary: MemoryBoundary::AllocationProjection,
            shortfall_bytes: 548_786_530,
            active_memory_bytes: 19_944_999_178,
            pending_allocation_bytes: 20_804_683_864,
            active_memory_ceiling_bytes: 40_200_896_512,
        },
    ));

    let inference_engine_error = InferenceEngineError::from(qwen3_5_execution_error);

    assert!(matches!(
        inference_engine_error,
        InferenceEngineError::InvalidRequest { reason }
            if reason.contains("expert_page_layer_20")
    ));
}

#[test]
fn should_keep_expert_page_memory_counter_failure_fatal() {
    let qwen3_5_execution_error = Qwen3_5ExecutionError::from(ExpertPagingError::MemoryBudget(
        MlxAllocationAdmissionError::MlxRuntime(MlxRuntimeError::RuntimeOperation {
            operation: "read MLX memory counters",
            description: "simulated counter failure".to_owned(),
        }),
    ));

    let inference_engine_error = InferenceEngineError::from(qwen3_5_execution_error);

    assert!(matches!(
        inference_engine_error,
        InferenceEngineError::Fatal { reason }
            if reason.contains("simulated counter failure")
    ));
}

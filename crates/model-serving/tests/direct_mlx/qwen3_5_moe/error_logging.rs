use astronomical_model_serving::{
    ExpertPagingError, InferenceEngineError, MemoryBudgetError, Qwen3_5ExecutionError,
};
use astronomical_runtime_integration::MlxRuntimeError;

#[test]
fn should_translate_expert_page_memory_budget_rejection_to_an_invalid_request() {
    let qwen3_5_execution_error = Qwen3_5ExecutionError::from(ExpertPagingError::MemoryBudget(
        MemoryBudgetError::BudgetExceeded {
            stage: "expert_page_layer_20".to_owned(),
            projected_bytes: 40_749_683_042,
            active_bytes: 19_944_999_178,
            allocator_cache_bytes: 15_874_672_728,
            configured_cap_bytes: 40_200_896_512,
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
        MemoryBudgetError::MlxRuntime(MlxRuntimeError::RuntimeOperation {
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

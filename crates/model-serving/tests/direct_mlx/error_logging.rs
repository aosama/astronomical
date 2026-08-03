use astronomical_model_serving::{
    ExpertPagingError, InferenceEngineError, MemoryBudgetError, Qwen3_5MoEExecutionError,
    WorkerRuntimeError,
};
use astronomical_runtime_integration::MlxRuntimeError;

#[test]
fn should_include_underlying_mlx_operation_in_worker_fatal_error_text() {
    let mlx_runtime_error = MlxRuntimeError::RuntimeOperation {
        operation: "evaluate Qwen3.5-MoE gated-delta decay graph",
        description: "simulated native MLX graph execution failure".to_owned(),
    };
    let qwen3_5_moe_execution_error = Qwen3_5MoEExecutionError::from(mlx_runtime_error);
    let inference_engine_error = InferenceEngineError::from(qwen3_5_moe_execution_error);
    let worker_runtime_error = WorkerRuntimeError::InferenceEngineGenerationFailed {
        reason: inference_engine_error.to_string(),
    };

    let worker_runtime_error_text = worker_runtime_error.to_string();

    assert!(
        worker_runtime_error_text.contains("direct MLX execution failed"),
        "worker fatal error should identify that the direct MLX layer failed: {worker_runtime_error_text}"
    );
    assert!(
        worker_runtime_error_text.contains("evaluate Qwen3.5-MoE gated-delta decay graph"),
        "worker fatal error should include the failed MLX operation: {worker_runtime_error_text}"
    );
    assert!(
        worker_runtime_error_text.contains("simulated native MLX graph execution failure"),
        "worker fatal error should include the native MLX failure description: {worker_runtime_error_text}"
    );
}

#[test]
fn should_translate_expert_page_memory_budget_rejection_to_an_invalid_request() {
    let qwen3_5_moe_execution_error = Qwen3_5MoEExecutionError::from(
        ExpertPagingError::MemoryBudget(MemoryBudgetError::BudgetExceeded {
            stage: "expert_page_layer_20".to_owned(),
            projected_bytes: 40_749_683_042,
            active_bytes: 19_944_999_178,
            allocator_cache_bytes: 15_874_672_728,
            configured_cap_bytes: 40_200_896_512,
        }),
    );

    let inference_engine_error = InferenceEngineError::from(qwen3_5_moe_execution_error);

    assert!(matches!(
        inference_engine_error,
        InferenceEngineError::InvalidRequest { reason }
            if reason.contains("expert_page_layer_20")
    ));
}

#[test]
fn should_translate_native_mlx_capacity_rejection_to_an_invalid_request() {
    let qwen3_5_moe_execution_error =
        Qwen3_5MoEExecutionError::from(MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 9_900_000_000,
            attempted_allocation_bytes: 300_000_000,
            allowed_active_memory_bytes: 10_100_000_000,
        });

    let inference_engine_error = InferenceEngineError::from(qwen3_5_moe_execution_error);

    assert!(matches!(
        inference_engine_error,
        InferenceEngineError::InvalidRequest { reason }
            if reason == "generation cannot fit under the configured MLX memory ceiling"
    ));
}

#[test]
fn should_keep_expert_page_memory_counter_failure_fatal() {
    let qwen3_5_moe_execution_error =
        Qwen3_5MoEExecutionError::from(ExpertPagingError::MemoryBudget(
            MemoryBudgetError::MlxRuntime(MlxRuntimeError::RuntimeOperation {
                operation: "read MLX memory counters",
                description: "simulated counter failure".to_owned(),
            }),
        ));

    let inference_engine_error = InferenceEngineError::from(qwen3_5_moe_execution_error);

    assert!(matches!(
        inference_engine_error,
        InferenceEngineError::Fatal { reason }
            if reason.contains("simulated counter failure")
    ));
}

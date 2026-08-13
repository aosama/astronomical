use astronomical_model_serving::{
    ExpertPagingError, InferenceEngineError, Qwen3_5ExecutionError, WorkerRuntimeError,
};
use astronomical_runtime_integration::MlxRuntimeError;

#[test]
fn should_include_underlying_mlx_operation_in_worker_fatal_error_text() {
    let mlx_runtime_error = MlxRuntimeError::RuntimeOperation {
        operation: "evaluate Qwen3.5-MoE gated-delta decay graph",
        description: "simulated native MLX graph execution failure".to_owned(),
    };
    let qwen3_5_execution_error = Qwen3_5ExecutionError::from(mlx_runtime_error);
    let inference_engine_error = InferenceEngineError::from(qwen3_5_execution_error);
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
fn should_translate_native_mlx_capacity_rejection_to_an_invalid_request() {
    let qwen3_5_execution_error =
        Qwen3_5ExecutionError::from(MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 9_900_000_000,
            attempted_allocation_bytes: 300_000_000,
            allowed_active_memory_bytes: 10_100_000_000,
        });

    let inference_engine_error = InferenceEngineError::from(qwen3_5_execution_error);

    assert!(matches!(
        inference_engine_error,
        InferenceEngineError::InvalidRequest { reason }
            if reason == "generation cannot fit under the configured MLX memory ceiling"
    ));
}

#[test]
fn should_translate_prefixed_native_capacity_rejection_to_an_invalid_request() {
    let qwen3_5_execution_error = Qwen3_5ExecutionError::from(ExpertPagingError::NativeRuntime(
        MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 35_346_771_214,
            attempted_allocation_bytes: 3_538_944,
            allowed_active_memory_bytes: 35_350_000_000,
        },
    ));

    let inference_engine_error = InferenceEngineError::from(qwen3_5_execution_error);

    assert!(matches!(
        inference_engine_error,
        InferenceEngineError::InvalidRequest { reason }
            if reason == "generation cannot fit under the configured MLX memory ceiling"
    ));
}

//! Direct MLX contracts for Laguna's typed centralized-memory integration.

use astronomical_model_serving::LagunaExecutionError;
use astronomical_runtime_integration::MlxRuntimeError;

#[test]
fn should_preserve_typed_active_memory_pressure_for_engine_recovery() {
    let execution_error = LagunaExecutionError::from(MlxRuntimeError::ActiveMemoryLimitExceeded {
        active_memory_bytes: 900,
        attempted_allocation_bytes: 125,
        allowed_active_memory_bytes: 1_000,
    });

    assert!(matches!(
        execution_error,
        LagunaExecutionError::Runtime(MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 900,
            attempted_allocation_bytes: 125,
            allowed_active_memory_bytes: 1_000,
        })
    ));
}

#[test]
fn should_classify_only_typed_capacity_failures_as_recoverable() {
    let active_limit_error =
        LagunaExecutionError::from(MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 900,
            attempted_allocation_bytes: 125,
            allowed_active_memory_bytes: 1_000,
        });
    let structural_runtime_error = LagunaExecutionError::from(MlxRuntimeError::RuntimeOperation {
        operation: "test Laguna operation",
        description: "a malformed tensor shape".to_owned(),
    });

    assert!(active_limit_error.is_recoverable_memory_pressure());
    assert!(!structural_runtime_error.is_recoverable_memory_pressure());
}

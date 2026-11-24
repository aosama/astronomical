use crate::mlx_runtime_types::{MlxMemorySnapshot, MlxRuntimeError};

#[test]
fn should_classify_only_the_metal_command_buffer_out_of_memory_error_as_recoverable() {
    let graphics_processor_out_of_memory_error = MlxRuntimeError::RuntimeOperation {
        operation: "synchronize the MLX GPU stream",
        description: "[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).".to_owned(),
    };
    let unrelated_insufficient_memory_error = MlxRuntimeError::RuntimeOperation {
        operation: "load model weights",
        description: "Insufficient Memory".to_owned(),
    };

    assert!(
        graphics_processor_out_of_memory_error.is_recoverable_graphics_processor_out_of_memory()
    );
    assert!(!unrelated_insufficient_memory_error.is_recoverable_graphics_processor_out_of_memory());
}

#[test]
fn should_raise_a_snapshot_peak_to_a_higher_measured_one() {
    let snapshot = MlxMemorySnapshot {
        active_memory_bytes: 10,
        mlx_buffer_cache_bytes: 3,
        peak_memory_bytes: 20,
    };
    let raised_snapshot = snapshot.with_peak_at_least(35);
    assert_eq!(raised_snapshot.peak_memory_bytes(), 35);
    assert_eq!(raised_snapshot.active_memory_bytes(), 10);
    assert_eq!(raised_snapshot.mlx_buffer_cache_bytes(), 3);
}

#[test]
fn should_keep_a_snapshot_peak_that_is_already_the_higher() {
    let snapshot = MlxMemorySnapshot {
        active_memory_bytes: 10,
        mlx_buffer_cache_bytes: 3,
        peak_memory_bytes: 20,
    };
    assert_eq!(snapshot.with_peak_at_least(20).peak_memory_bytes(), 20);
    assert_eq!(snapshot.with_peak_at_least(0).peak_memory_bytes(), 20);
}

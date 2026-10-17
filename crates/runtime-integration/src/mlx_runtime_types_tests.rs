use crate::mlx_runtime_types::MlxRuntimeError;

#[test]
fn should_classify_only_the_metal_command_buffer_out_of_memory_error_as_recoverable() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

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

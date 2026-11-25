use crate::mlx_runtime_types::{
    MLX_BUFFER_CACHE_MEASURED_LIMIT_FLOOR_BYTES, MlxMemoryLimits, MlxMemorySnapshot,
    MlxRuntimeError, measured_mlx_buffer_cache_limit_bytes,
    mlx_buffer_cache_limit_after_active_limit_change,
};

const GIB: usize = 1_073_741_824;

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

#[test]
fn should_bound_the_mlx_buffer_cache_limit_to_one_measured_step() {
    assert_eq!(
        measured_mlx_buffer_cache_limit_bytes(36 * GIB, 3 * GIB as u64),
        3 * GIB
    );
}

#[test]
fn should_raise_a_small_measured_step_to_the_floor() {
    assert_eq!(
        measured_mlx_buffer_cache_limit_bytes(36 * GIB, 952_571_664),
        MLX_BUFFER_CACHE_MEASURED_LIMIT_FLOOR_BYTES
    );
    assert_eq!(
        measured_mlx_buffer_cache_limit_bytes(36 * GIB, 0),
        MLX_BUFFER_CACHE_MEASURED_LIMIT_FLOOR_BYTES
    );
}

#[test]
fn should_never_bound_the_mlx_buffer_cache_above_the_active_limit() {
    assert_eq!(
        measured_mlx_buffer_cache_limit_bytes(GIB / 2, 3 * GIB as u64),
        GIB / 2
    );
    assert_eq!(
        measured_mlx_buffer_cache_limit_bytes(8 * GIB, u64::MAX),
        8 * GIB
    );
}

#[test]
fn should_keep_following_the_active_limit_when_the_cache_limit_did() {
    let following_limits = MlxMemoryLimits::new(34 * GIB, 34 * GIB).expect("valid limits");
    assert_eq!(
        mlx_buffer_cache_limit_after_active_limit_change(following_limits, 36 * GIB),
        36 * GIB
    );
}

#[test]
fn should_keep_a_measured_bound_across_an_active_limit_change() {
    let bounded_limits = MlxMemoryLimits::new(34 * GIB, GIB).expect("valid limits");
    assert_eq!(
        mlx_buffer_cache_limit_after_active_limit_change(bounded_limits, 36 * GIB),
        GIB
    );
    assert_eq!(
        mlx_buffer_cache_limit_after_active_limit_change(bounded_limits, GIB / 2),
        GIB / 2,
        "a lowered active limit caps the bound"
    );
}

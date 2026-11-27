use model_primitives::synthetic_weights_test_support::initialize_test_runtime;
use serial_test::file_serial;

use super::{reclaim_mlx_buffer_cache, reclaimed_pass_count};
use crate::decision_models::training::gpu_math::GpuMath;

// --- positive ---

#[test]
#[file_serial]
fn should_report_a_clear_when_the_mlx_buffer_cache_reaches_the_threshold() {
    let runtime = initialize_test_runtime();
    // Any MLX buffer cache, an empty one included, reaches a zero threshold.
    assert!(matches!(
        runtime.reclaim_mlx_buffer_cache_if_above_threshold(0),
        Ok(true)
    ));
}

// --- edge / boundary ---

#[test]
#[file_serial]
fn should_never_count_fewer_reclaimed_passes_than_before() {
    let runtime = initialize_test_runtime();
    let count_before = reclaimed_pass_count();

    reclaim_mlx_buffer_cache(GpuMath { runtime: &runtime }).unwrap();

    assert!(reclaimed_pass_count() >= count_before);
}

// --- negative ---

#[test]
#[file_serial]
fn should_report_no_clear_below_the_threshold() {
    let runtime = initialize_test_runtime();
    assert!(matches!(
        runtime.reclaim_mlx_buffer_cache_if_above_threshold(usize::MAX),
        Ok(false)
    ));
}

use super::{device_memory_ceiling, memory_limits_within_device, native_memory_limits};
use crate::MlxMemoryLimits;

const GIGABYTE_BYTES: usize = 1 << 30;

fn requested_limits(active_gigabytes: usize, mlx_buffer_cache_gigabytes: usize) -> MlxMemoryLimits {
    MlxMemoryLimits::new(
        active_gigabytes * GIGABYTE_BYTES,
        mlx_buffer_cache_gigabytes * GIGABYTE_BYTES,
    )
    .unwrap()
}

// --- positive ---

#[test]
fn should_cap_both_limits_at_a_smaller_gpu() {
    let device_memory_ceiling_bytes = 3 * GIGABYTE_BYTES;
    let native_limits =
        memory_limits_within_device(requested_limits(12, 6), Some(device_memory_ceiling_bytes));
    assert_eq!(
        native_limits.active_memory_limit_bytes(),
        device_memory_ceiling_bytes
    );
    assert_eq!(
        native_limits.mlx_buffer_cache_limit_bytes(),
        device_memory_ceiling_bytes
    );
}

#[test]
fn should_take_mlxs_own_share_of_an_idle_gpus_memory() {
    // 95% of 4 GB, exactly.
    assert_eq!(
        device_memory_ceiling(4 * GIGABYTE_BYTES, 4 * GIGABYTE_BYTES),
        Ok(4_080_218_931)
    );
}

#[test]
fn should_hand_mlx_the_capped_limits_on_this_build() {
    // On a build without the CUDA backend nothing changes; on a CUDA build
    // the limits come back no larger than requested and below the GPU's
    // memory, which no machine's 1 TB request fits under.
    let requested = requested_limits(1024, 2);
    let native_limits = native_memory_limits(requested).unwrap();
    if mlx_bridge::cuda_is_available().unwrap() {
        assert!(native_limits.active_memory_limit_bytes() < requested.active_memory_limit_bytes());
        assert!(
            native_limits.mlx_buffer_cache_limit_bytes()
                <= requested.mlx_buffer_cache_limit_bytes()
        );
    } else {
        assert_eq!(native_limits, requested);
    }
}

// --- edge / boundary ---

#[test]
fn should_keep_limits_that_already_fit_the_gpu() {
    let requested = requested_limits(2, 1);
    assert_eq!(
        memory_limits_within_device(requested, Some(2 * GIGABYTE_BYTES)),
        requested
    );
}

#[test]
fn should_cap_only_the_active_limit_when_the_mlx_buffer_cache_already_fits() {
    let device_memory_ceiling_bytes = 3 * GIGABYTE_BYTES;
    let native_limits =
        memory_limits_within_device(requested_limits(12, 1), Some(device_memory_ceiling_bytes));
    assert_eq!(
        native_limits.active_memory_limit_bytes(),
        device_memory_ceiling_bytes
    );
    assert_eq!(native_limits.mlx_buffer_cache_limit_bytes(), GIGABYTE_BYTES);
}

#[test]
fn should_hold_the_ceiling_to_what_another_process_left_free() {
    // A GPU that also drives a display, with a gigabyte already taken.
    assert_eq!(
        device_memory_ceiling(4 * GIGABYTE_BYTES, 3 * GIGABYTE_BYTES),
        Ok(3 * GIGABYTE_BYTES)
    );
}

// --- negative ---

#[test]
fn should_change_nothing_without_a_cuda_gpu() {
    let requested = requested_limits(12, 6);
    assert_eq!(memory_limits_within_device(requested, None), requested);
}

#[test]
fn should_refuse_a_gpu_reported_with_no_memory() {
    assert!(device_memory_ceiling(0, 0).is_err());
}

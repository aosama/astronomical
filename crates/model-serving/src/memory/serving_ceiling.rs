const BYTES_PER_GIBIBYTE: u64 = 1024 * 1024 * 1024;
const MINIMUM_GPU_SERVING_HEADROOM_BYTES: u64 = 2 * BYTES_PER_GIBIBYTE;
const MAXIMUM_GPU_SERVING_HEADROOM_BYTES: u64 = 8 * BYTES_PER_GIBIBYTE;
const GPU_WORKING_SET_HEADROOM_DIVISOR: u64 = 8;

/// Leaves bounded allocation headroom below the recommended GPU working set.
#[must_use]
pub const fn recommended_gpu_serving_memory_ceiling_bytes(
    recommended_gpu_working_set_bytes: u64,
) -> u64 {
    let proportional_headroom_bytes =
        recommended_gpu_working_set_bytes / GPU_WORKING_SET_HEADROOM_DIVISOR;
    let reserved_headroom_bytes =
        if proportional_headroom_bytes < MINIMUM_GPU_SERVING_HEADROOM_BYTES {
            MINIMUM_GPU_SERVING_HEADROOM_BYTES
        } else if proportional_headroom_bytes > MAXIMUM_GPU_SERVING_HEADROOM_BYTES {
            MAXIMUM_GPU_SERVING_HEADROOM_BYTES
        } else {
            proportional_headroom_bytes
        };
    recommended_gpu_working_set_bytes.saturating_sub(reserved_headroom_bytes)
}

/// Resolves the effective ceiling under the configured, physical, and serving limits.
#[must_use]
pub const fn effective_mlx_serving_memory_ceiling_bytes(
    configured_mlx_memory_ceiling_bytes: Option<u64>,
    machine_mlx_memory_ceiling_bytes: u64,
    serving_mlx_memory_ceiling_bytes: u64,
) -> u64 {
    let requested_mlx_memory_ceiling_bytes = match configured_mlx_memory_ceiling_bytes {
        Some(configured_mlx_memory_ceiling_bytes) => configured_mlx_memory_ceiling_bytes,
        None => machine_mlx_memory_ceiling_bytes,
    };
    let machine_bounded_ceiling_bytes =
        if requested_mlx_memory_ceiling_bytes < machine_mlx_memory_ceiling_bytes {
            requested_mlx_memory_ceiling_bytes
        } else {
            machine_mlx_memory_ceiling_bytes
        };
    if machine_bounded_ceiling_bytes < serving_mlx_memory_ceiling_bytes {
        machine_bounded_ceiling_bytes
    } else {
        serving_mlx_memory_ceiling_bytes
    }
}

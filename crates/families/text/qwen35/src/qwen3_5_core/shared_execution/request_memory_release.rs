//! Final synchronized MLX allocator cleanup after request owners are dropped.

use ipc_protocol::RequestId;
use runtime_integration::MlxMemorySnapshot;

use crate::qwen3_5_core::model::base::Qwen3_5ModelBase;

/// Synchronizes submitted request work, then returns reusable MLX allocations.
///
/// The finalization caller must drop request-owned arrays first. Keeping that
/// ownership step outside this helper, which sees only the model, makes it
/// impossible to report cleanup while the request still holds decoder or
/// snapshot arrays. With no model loaded there is nothing to release.
pub fn release_request_memory(
    loaded_model: Option<&Qwen3_5ModelBase>,
    request_id: RequestId,
) -> Option<MlxMemorySnapshot> {
    let runtime = loaded_model?.runtime();
    if let Err(mlx_buffer_cache_error) = runtime.synchronize_gpu_stream_and_clear_mlx_buffer_cache()
    {
        tracing::warn!(request_id = request_id.value(), error = %mlx_buffer_cache_error,
            "failed to release reclaimable MLX request memory");
        return None;
    }
    match runtime.memory_snapshot() {
        Ok(mlx_memory_snapshot) => {
            tracing::info!(
                request_id = request_id.value(),
                mlx_active_bytes = mlx_memory_snapshot.active_memory_bytes(),
                mlx_buffer_cache_bytes = mlx_memory_snapshot.mlx_buffer_cache_bytes(),
                mlx_peak_bytes = mlx_memory_snapshot.peak_memory_bytes(),
                "released reclaimable MLX request memory"
            );
            Some(mlx_memory_snapshot)
        }
        Err(snapshot_error) => {
            tracing::warn!(
                request_id = request_id.value(), error = %snapshot_error,
                "released MLX request memory but could not sample allocator metrics"
            );
            None
        }
    }
}

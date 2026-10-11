//! Final synchronized MLX allocator cleanup after request owners are dropped.

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxMemorySnapshot;

use super::Qwen3_5EngineState;

impl Qwen3_5EngineState {
    /// Synchronizes submitted request work, then returns reusable MLX allocations.
    ///
    /// The finalization caller must drop request-owned arrays first. Keeping that
    /// ownership step outside this immutable model helper makes it impossible to
    /// report cleanup while the request still holds decoder or snapshot arrays.
    pub(crate) fn release_request_memory(
        &self,
        request_id: RequestId,
        should_capture_memory_snapshot: bool,
    ) -> Option<MlxMemorySnapshot> {
        let model = self.model.as_ref()?;
        if let Err(allocator_cache_error) = model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
        {
            tracing::warn!(request_id = request_id.value(), error = %allocator_cache_error,
                "failed to release reclaimable MLX request memory");
            return None;
        }
        if !should_capture_memory_snapshot {
            return None;
        }
        match model.runtime().memory_snapshot() {
            Ok(mlx_memory_snapshot) => {
                tracing::info!(
                    request_id = request_id.value(),
                    mlx_active_bytes = mlx_memory_snapshot.active_memory_bytes(),
                    mlx_allocator_cache_bytes = mlx_memory_snapshot.allocator_cache_memory_bytes(),
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
}

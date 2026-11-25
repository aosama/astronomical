//! The MLX sample taken after a forward.

use runtime_integration::MlxMemorySnapshot;

use crate::InferenceEngineError;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;

use super::fatal_runtime_error;

/// One MLX sample taken on the owner thread after a forward. The residency
/// claim published with it is derived from the breakdown of this exact
/// sample, so the two can never describe different instants (issue #337).
#[derive(Debug)]
pub(in crate::qwen3_5_resident) struct CompletedForwardMemoryObservation {
    pub(in crate::qwen3_5_resident) mlx_memory_snapshot: MlxMemorySnapshot,
}

/// Captures the raw MLX snapshot at one instant.
pub(in crate::qwen3_5_resident) fn capture_completed_forward_memory_observation(
    model: &Qwen3_5ResidentModel,
) -> Result<CompletedForwardMemoryObservation, InferenceEngineError> {
    let mlx_memory_snapshot = model
        .runtime()
        .memory_snapshot()
        .map_err(fatal_runtime_error)?;
    Ok(CompletedForwardMemoryObservation {
        mlx_memory_snapshot,
    })
}

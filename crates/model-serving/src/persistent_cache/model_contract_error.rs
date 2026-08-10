use thiserror::Error;

use crate::DecoderCacheLayoutError;

/// A model and its live resource budgets could not form a safe persistent-state contract.
#[derive(Debug, Error)]
pub enum PersistentPromptCacheModelContractError {
    #[error("persistent model-state storage requires a nonempty model ID")]
    EmptyModelId,
    #[error("persistent model-state storage requires a nonempty model revision")]
    EmptyModelRevision,
    #[error("persistent model-state storage requires a positive maximum context")]
    ZeroMaximumContextTokenCount,
    #[error("decoder-cache layout declares neither sequence nor boundary state")]
    NoPersistentState,
    #[error("decoder-cache storage geometry used a zero divisor")]
    ZeroStorageGeometryDivisor,
    #[error("persistent model-state block token count overflowed")]
    BlockTokenCountOverflow,
    #[error("persistent sequence-state block payload byte count overflowed")]
    SequenceStateBlockPayloadByteCountOverflow,
    #[error("persistent model-state capture payload byte count overflowed")]
    CapturePayloadByteCountOverflow,
    #[error(
        "boundary snapshot requires {boundary_snapshot_bytes} bytes, above SSD quota {global_ssd_quota_bytes} bytes"
    )]
    BoundarySnapshotExceedsSsdQuota {
        boundary_snapshot_bytes: u64,
        global_ssd_quota_bytes: u64,
    },
    #[error(
        "persistent model-state capture requires {capture_memory_bytes} memory bytes, above MLX ceiling {effective_mlx_memory_ceiling_bytes} bytes"
    )]
    CaptureExceedsMlxMemoryCeiling {
        capture_memory_bytes: u64,
        effective_mlx_memory_ceiling_bytes: u64,
    },
    #[error(
        "persistent model-state block files require {block_file_bytes} SSD bytes, above quota {global_ssd_quota_bytes} bytes"
    )]
    BlockFilesExceedSsdQuota {
        block_file_bytes: u64,
        global_ssd_quota_bytes: u64,
    },
    #[error("failed to derive persistent model-state storage geometry")]
    SerializeStorageGeometry(#[source] serde_json::Error),
    #[error(transparent)]
    DecoderCacheLayout(#[from] DecoderCacheLayoutError),
}

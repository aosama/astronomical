use thiserror::Error;

/// Errors returned by the architecture-neutral inference engine seam.
#[derive(Debug, Error)]
pub enum InferenceEngineError {
    /// The engine cannot accept another generation request right now.
    #[error("engine is busy")]
    EngineBusy,

    /// The request cannot execute safely but the loaded engine remains reusable.
    #[error("invalid generation request: {reason}")]
    InvalidRequest {
        /// Bounded human-readable reason suitable for the worker protocol.
        reason: String,
    },

    /// The requested live MLX ceiling is below the loaded model's safe idle minimum.
    #[error(
        "requested MLX memory ceiling {requested_mlx_memory_ceiling_bytes} bytes is below the safe minimum {minimum_mlx_memory_ceiling_bytes} bytes: {reason}"
    )]
    MlxMemoryLimitRejected {
        /// Requested effective ceiling in bytes.
        requested_mlx_memory_ceiling_bytes: u64,
        /// Exact loaded-model safe idle minimum in bytes.
        minimum_mlx_memory_ceiling_bytes: u64,
        /// Bounded explanation suitable for the worker protocol.
        reason: String,
    },

    /// The engine encountered a fatal condition.
    #[error("fatal engine error: {reason}")]
    Fatal {
        /// Context-rich fatal reason.
        reason: String,
    },
}

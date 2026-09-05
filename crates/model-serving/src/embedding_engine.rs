//! Architecture-neutral lifecycle contract implemented by a concrete embedding runtime.
//!
//! Embedding inference is a single synchronous encoder forward pass, so unlike
//! the image runtime there is no step loop and no mid-request cancellation:
//! one command either returns complete vectors or a bounded failure.

use astronomical_ipc_protocol::{
    EmbeddingsCommand, EmbeddingsFailureReason, WorkerEmbeddingCapabilities,
};

use crate::MlxMemoryTelemetry;

/// A loaded embedding runtime that turns validated text into unit-norm vectors.
pub trait EmbeddingEngine: 'static {
    /// Loads model resources and returns the exact identity and limits to advertise.
    fn load(&mut self) -> Result<EmbeddingEngineLoadResult, EmbeddingsFailureReason>;

    /// Computes one unit-norm vector per input, in request order.
    fn embed(
        &mut self,
        embeddings_command: &EmbeddingsCommand,
    ) -> Result<EmbeddingEngineOutput, EmbeddingsFailureReason>;

    /// Returns the final MLX observation captured after request cleanup, when available.
    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        None
    }

    /// Collects an idle memory observation without adding generation methods.
    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        None
    }
}

/// Loaded embedding identity and advertised request bounds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddingEngineLoadResult {
    model_id: String,
    capabilities: WorkerEmbeddingCapabilities,
    minimum_mlx_memory_ceiling_bytes: u64,
}

impl EmbeddingEngineLoadResult {
    #[must_use]
    pub fn new(model_id: impl Into<String>, capabilities: WorkerEmbeddingCapabilities) -> Self {
        Self {
            model_id: model_id.into(),
            capabilities,
            minimum_mlx_memory_ceiling_bytes: 1,
        }
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[must_use]
    pub const fn capabilities(&self) -> &WorkerEmbeddingCapabilities {
        &self.capabilities
    }

    #[must_use]
    pub const fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.minimum_mlx_memory_ceiling_bytes
    }
}

/// One completed embeddings computation for one worker command.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingEngineOutput {
    pub embeddings: Vec<Vec<f32>>,
    pub input_token_counts: Vec<u32>,
    pub elapsed_millis: u64,
}

/// Default embedding slot used by existing chat-only and image-only workers.
#[doc(hidden)]
pub struct EmbeddingUnavailableEngine;

impl EmbeddingEngine for EmbeddingUnavailableEngine {
    fn load(&mut self) -> Result<EmbeddingEngineLoadResult, EmbeddingsFailureReason> {
        Err(EmbeddingsFailureReason::FatalExecution {
            reason: "the selected model does not support embeddings".to_owned(),
        })
    }

    fn embed(
        &mut self,
        _embeddings_command: &EmbeddingsCommand,
    ) -> Result<EmbeddingEngineOutput, EmbeddingsFailureReason> {
        Err(EmbeddingsFailureReason::FatalExecution {
            reason: "the selected model does not support embeddings".to_owned(),
        })
    }
}

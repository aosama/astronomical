//! Architecture-neutral lifecycle contract implemented by a concrete embedding runtime.
//!
//! Embedding inference is a single synchronous encoder forward pass, so unlike
//! the image runtime there is no step loop and no mid-request cancellation:
//! one command either returns complete vectors or a bounded failure.

use ipc_protocol::{EmbeddingsCommand, EmbeddingsFailureReason, WorkerEmbeddingCapabilities};

use crate::memory::telemetry::MlxMemoryTelemetry;
use memory_policy::live_memory_limit::LiveMemoryLimitEngine;

/// A loaded embedding runtime that turns validated text into unit-norm vectors.
///
/// `Any` lets a caller holding a boxed engine ask which family it is; every
/// engine is `'static`, so it costs nothing.
pub trait EmbeddingEngine: std::any::Any + LiveMemoryLimitEngine {
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

/// A boxed engine is an engine, so the inference worker holds whichever
/// embedding family its factory built and a second family needs no change in
/// the worker's type.
impl EmbeddingEngine for Box<dyn EmbeddingEngine> {
    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.as_ref().collect_mlx_memory_telemetry()
    }

    fn embed(
        &mut self,
        embeddings_command: &EmbeddingsCommand,
    ) -> Result<EmbeddingEngineOutput, EmbeddingsFailureReason> {
        self.as_mut().embed(embeddings_command)
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.as_mut().take_post_cleanup_memory_telemetry()
    }

    fn load(&mut self) -> Result<EmbeddingEngineLoadResult, EmbeddingsFailureReason> {
        self.as_mut().load()
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

/// How the worker builds an embedding family's unloaded engine (issue
/// #2473, Step 5): the family names its builder, and the worker builds
/// every embedding family the same way, with no code of its own per family.
pub trait EmbeddingFamilyConstruction: 'static {
    type Engine: EmbeddingEngine + 'static;
    /// The unloaded engine for the folder; it reads only its configuration
    /// and tokenizer, and MLX work waits for `load` on the worker's thread.
    const BUILD_ENGINE: fn(
        &model_primitives::FamilyEngineSettings,
        String,
        &std::path::Path,
    ) -> Result<Self::Engine, EmbeddingsFailureReason>;
}

/// Default embedding slot used by existing chat-only and image-only workers.
#[doc(hidden)]
pub struct EmbeddingUnavailableEngine;

impl LiveMemoryLimitEngine for EmbeddingUnavailableEngine {}

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

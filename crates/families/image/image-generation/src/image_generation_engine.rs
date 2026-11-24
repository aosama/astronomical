//! Architecture-neutral lifecycle contract implemented by a concrete image runtime such as FLUX.

use ipc_protocol::{
    GeneratedImage, ImageGenerationCapabilities, ImageGenerationCommand,
    ImageGenerationFailureReason, ImageGenerationPhase, ImageGenerationResultMetadata, RequestId,
};

use memory_policy::telemetry::MlxMemoryLimitAdjustment;
use memory_policy::telemetry::MlxMemoryTelemetry;

/// A loaded image runtime advanced through bounded, independently cancellable steps.
///
/// Lifecycle methods stay synchronous because native image arrays are owned by the
/// worker runtime thread and must never be moved through a `Send` future.
///
/// `Any` lets a caller holding a boxed engine ask which family it is, as the
/// worker factory's tests do; every engine is `'static`, so it costs nothing.
pub trait ImageGenerationEngine: std::any::Any {
    /// Loads model resources and returns the exact identity and limits to advertise.
    fn load(&mut self) -> Result<ImageGenerationEngineLoadResult, ImageGenerationFailureReason>;

    /// Creates request-scoped image state from an independently validated command.
    fn start_generation(
        &mut self,
        generation_command: ImageGenerationCommand,
    ) -> Result<(), ImageGenerationFailureReason>;

    /// Performs at most one prompt, transformer block-group, decoding, or encoding boundary.
    fn advance_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason>;

    /// Releases request-scoped state without unloading reusable model resources.
    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), ImageGenerationFailureReason>;

    /// Returns the final MLX observation captured after request cleanup, when available.
    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        None
    }

    /// Collects an idle memory observation without adding autoregressive token methods.
    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        None
    }

    /// Applies a live process memory ceiling while no image request is active.
    fn update_mlx_memory_limit(
        &mut self,
        _requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, ImageGenerationFailureReason> {
        Err(ImageGenerationFailureReason::FatalExecution {
            reason: "this image engine does not support live MLX memory limits".to_owned(),
        })
    }
}

/// A boxed engine is an engine, so the inference worker can hold whichever
/// image family its factory built without naming each family in its type.
impl ImageGenerationEngine for Box<dyn ImageGenerationEngine> {
    fn load(&mut self) -> Result<ImageGenerationEngineLoadResult, ImageGenerationFailureReason> {
        self.as_mut().load()
    }

    fn start_generation(
        &mut self,
        generation_command: ImageGenerationCommand,
    ) -> Result<(), ImageGenerationFailureReason> {
        self.as_mut().start_generation(generation_command)
    }

    fn advance_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        self.as_mut().advance_generation(request_id)
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), ImageGenerationFailureReason> {
        self.as_mut().cancel_generation(request_id)
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.as_ref().collect_mlx_memory_telemetry()
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.as_mut().take_post_cleanup_memory_telemetry()
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, ImageGenerationFailureReason> {
        self.as_mut()
            .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
    }
}

/// Loaded image identity and model-specific capability bounds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageGenerationEngineLoadResult {
    model_id: String,
    capabilities: ImageGenerationCapabilities,
    minimum_mlx_memory_ceiling_bytes: u64,
}

impl ImageGenerationEngineLoadResult {
    #[must_use]
    pub fn new(model_id: impl Into<String>, capabilities: ImageGenerationCapabilities) -> Self {
        Self {
            model_id: model_id.into(),
            capabilities,
            minimum_mlx_memory_ceiling_bytes: 1,
        }
    }

    /// Records the safe loaded-model floor without reducing it to an untyped factory string.
    #[must_use]
    pub const fn with_minimum_mlx_memory_ceiling_bytes(
        mut self,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> Self {
        self.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
        self
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[must_use]
    pub const fn capabilities(&self) -> &ImageGenerationCapabilities {
        &self.capabilities
    }

    #[must_use]
    pub const fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.minimum_mlx_memory_ceiling_bytes
    }
}

/// One bounded result from a concrete image engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageGenerationEngineStep {
    Progress {
        phase: ImageGenerationPhase,
        completed_steps: u16,
        total_steps: u16,
        elapsed_millis: u64,
    },
    Completed {
        generated_image: GeneratedImage,
        result_metadata: ImageGenerationResultMetadata,
    },
}

/// Default image slot used by existing chat-only workers.
#[doc(hidden)]
pub struct ImageGenerationUnavailableEngine;

impl ImageGenerationEngine for ImageGenerationUnavailableEngine {
    fn load(&mut self) -> Result<ImageGenerationEngineLoadResult, ImageGenerationFailureReason> {
        Err(ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration)
    }

    fn start_generation(
        &mut self,
        _generation_command: ImageGenerationCommand,
    ) -> Result<(), ImageGenerationFailureReason> {
        Err(ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration)
    }

    fn advance_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        Err(ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration)
    }

    fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<(), ImageGenerationFailureReason> {
        Ok(())
    }
}

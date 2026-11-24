//! The [`ImageGenerationEngine`] every image family runs behind: load once,
//! serve one request at a time, one boundary per advance, and publish only
//! after cleanup succeeds. A family supplies only what differs, through
//! [`ImageFamily`]: how a request is checked and started, and how it advances
//! through the family's own render steps. FLUX.2 Klein and Qwen-Image-2.1 each
//! carried their own copy of this lifecycle before; a fix to it now reaches
//! both, and a new image family writes none of it.

use std::time::Instant;

use ipc_protocol::{
    GeneratedImage, ImageGenerationCapabilities, ImageGenerationCommand,
    ImageGenerationFailureReason, ImageGenerationPhase, ImageGenerationResultMetadata, RequestId,
};
use memory_policy::telemetry::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
use model_primitives::performance_attribution::PerformanceAttributionOutcome;

use crate::image_generation_engine::{
    ImageGenerationEngine, ImageGenerationEngineLoadResult, ImageGenerationEngineStep,
};

/// Loaded identity supplied by the native components or a scripted lifecycle fake.
#[doc(hidden)]
pub struct ImageComponentLoad {
    pub(crate) model_id: String,
    pub(crate) revision: String,
    pub(crate) capabilities: ImageGenerationCapabilities,
    pub(crate) minimum_mlx_memory_ceiling_bytes: u64,
}

impl ImageComponentLoad {
    /// Records the identity, advertisement, and memory floor one load established.
    #[doc(hidden)]
    #[must_use]
    pub fn new(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        capabilities: ImageGenerationCapabilities,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            revision: revision.into(),
            capabilities,
            minimum_mlx_memory_ceiling_bytes,
        }
    }
}

/// What every image family's component owner does, whatever its render steps.
///
/// The seam keeps lifecycle tests scripted (a fake owner drives the whole
/// lifecycle without MLX) while production keeps one concrete owner. It
/// deliberately has no `Send` bound because MLX component state is
/// runtime-thread-affine.
#[doc(hidden)]
pub trait ImageEngineComponents {
    /// Validates the artifact and reports the serving identity, advertisement, and memory floor.
    fn load(&mut self) -> Result<ImageComponentLoad, String>;

    /// Closes the request: releases its state, writes attribution, and
    /// captures the post-cleanup memory observation.
    fn finalize_request(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        encoded_bytes: Option<u64>,
        failure_description: Option<&str>,
    ) -> Result<(), String>;

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        None
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        None
    }

    /// Records the highest ceiling a load may plan at to reach its floor
    /// (#1902), so a load whose floor is above the ceiling plans at its floor
    /// instead of refusing, unless the user set a lower limit. Components that
    /// never refuse at load ignore it.
    fn adopt_floor_raise_limit(&mut self, _floor_raise_limit_bytes: u64) {}

    fn update_mlx_memory_limit(
        &mut self,
        _requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, String> {
        Err("live MLX memory limits are unavailable".to_owned())
    }
}

/// What the shell needs from a family's record of the request in progress.
pub trait ActiveImageRequest {
    fn request_id(&self) -> RequestId;
    fn started_at(&self) -> Instant;
    /// The step count progress reports are measured against.
    fn total_steps(&self) -> u16;
    /// What the finished image reports about how it was made.
    fn result_metadata(&self, elapsed_millis: u64) -> ImageGenerationResultMetadata;
}

/// What an image family adds to the shared lifecycle.
pub trait ImageFamily: Sized + 'static {
    /// The family's component owner, usually a trait object over its own
    /// component trait, which extends [`ImageEngineComponents`].
    type Components: ?Sized + ImageEngineComponents;
    type ActiveRequest: ActiveImageRequest;
    /// The name every public message uses, such as "FLUX.2 Klein".
    const FAMILY_NAME: &'static str;

    /// Checks a command and starts it on the components. The engine is
    /// loaded and idle when this runs; a component failure is reported
    /// through [`ImageEngineShell::fail_active_request`].
    fn start_request(
        engine: &mut ImageEngineShell<Self>,
        generation_command: ImageGenerationCommand,
    ) -> Result<Self::ActiveRequest, ImageGenerationFailureReason>;

    /// Runs one boundary of the request. A request that continues goes back
    /// through [`ImageEngineShell::restore_with_progress`]; a finished one
    /// through [`ImageEngineShell::complete_rendered_request`].
    fn advance_request(
        engine: &mut ImageEngineShell<Self>,
        active_request: Self::ActiveRequest,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason>;
}

/// An image family's engine: its components and the request in progress.
pub struct ImageEngineShell<Family: ImageFamily> {
    pub serving_model_id: String,
    pub components: Box<Family::Components>,
    loaded_revision: Option<String>,
    active_request: Option<Family::ActiveRequest>,
}

impl<Family: ImageFamily> ImageEngineShell<Family> {
    /// Constructor over any component owner.
    pub fn from_components(
        serving_model_id: impl Into<String>,
        components: Box<Family::Components>,
    ) -> Self {
        Self {
            serving_model_id: serving_model_id.into(),
            components,
            loaded_revision: None,
            active_request: None,
        }
    }

    /// Tells the engine how far a load may plan above its ceiling to reach its
    /// floor (#1902): the user's own limit, or unbounded when they set none.
    #[must_use]
    pub fn with_floor_raise_limit_bytes(mut self, floor_raise_limit_bytes: u64) -> Self {
        self.components
            .adopt_floor_raise_limit(floor_raise_limit_bytes);
        self
    }

    #[must_use]
    pub fn loaded_revision(&self) -> Option<&str> {
        self.loaded_revision.as_deref()
    }

    pub fn elapsed_millis(active_request: &Family::ActiveRequest) -> u64 {
        u64::try_from(active_request.started_at().elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// Ends the request in progress as failed and returns the public reason.
    /// Concrete component failures are path-free and bounded here, so a user
    /// can tell malformed tensors, unsupported operations and memory pressure
    /// apart.
    pub fn fail_active_request(
        &mut self,
        failure_description: String,
        public_failure_description: &'static str,
    ) -> ImageGenerationFailureReason {
        self.active_request = None;
        match self.components.finalize_request(
            PerformanceAttributionOutcome::Failed,
            None,
            Some(&failure_description),
        ) {
            Ok(()) => ImageGenerationFailureReason::FatalExecution {
                reason: format!("{public_failure_description}: {failure_description}")
                    .chars()
                    .take(512)
                    .collect(),
            },
            Err(_cleanup_failure) => ImageGenerationFailureReason::FatalExecution {
                reason: format!(
                    "{} image generation and cleanup failed",
                    Family::FAMILY_NAME
                ),
            },
        }
    }

    pub fn progress(
        active_request: &Family::ActiveRequest,
        phase: ImageGenerationPhase,
        completed_steps: u16,
    ) -> ImageGenerationEngineStep {
        ImageGenerationEngineStep::Progress {
            phase,
            completed_steps,
            total_steps: active_request.total_steps(),
            elapsed_millis: Self::elapsed_millis(active_request),
        }
    }

    /// Keeps a still-running request in progress and reports where it stands.
    pub fn restore_with_progress(
        &mut self,
        active_request: Family::ActiveRequest,
        phase: ImageGenerationPhase,
        completed_steps: u16,
    ) -> ImageGenerationEngineStep {
        let progress = Self::progress(&active_request, phase, completed_steps);
        self.active_request = Some(active_request);
        progress
    }

    /// Finalizes a request whose image is encoded and returns it. A failed
    /// finalize after a successful encode is fatal: the image exists, but the
    /// accounting contract cannot be honored.
    pub fn complete_rendered_request(
        &mut self,
        active_request: Family::ActiveRequest,
        png_bytes: Vec<u8>,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        let elapsed_millis = Self::elapsed_millis(&active_request);
        let encoded_byte_count = u64::try_from(png_bytes.len()).unwrap_or(u64::MAX);
        if self
            .components
            .finalize_request(
                PerformanceAttributionOutcome::Success,
                Some(encoded_byte_count),
                None,
            )
            .is_err()
        {
            return Err(ImageGenerationFailureReason::FatalExecution {
                reason: format!(
                    "{} could not finalize the generated image",
                    Family::FAMILY_NAME
                ),
            });
        }
        Ok(ImageGenerationEngineStep::Completed {
            generated_image: GeneratedImage {
                mime_type: "image/png".to_owned(),
                encoded_bytes: png_bytes,
            },
            result_metadata: active_request.result_metadata(elapsed_millis),
        })
    }

    /// The request in progress, if `request_id` names it. A mismatch leaves
    /// the request in progress untouched.
    fn take_matching_active_request(
        &mut self,
        request_id: RequestId,
    ) -> Result<Family::ActiveRequest, ImageGenerationFailureReason> {
        let Some(active_request) = self.active_request.take() else {
            return Err(ImageGenerationFailureReason::invalid_request(format!(
                "no {} request is active",
                Family::FAMILY_NAME
            )));
        };
        if active_request.request_id() != request_id {
            self.active_request = Some(active_request);
            return Err(mismatched_request_failure(Family::FAMILY_NAME));
        }
        Ok(active_request)
    }
}

fn mismatched_request_failure(family_name: &str) -> ImageGenerationFailureReason {
    ImageGenerationFailureReason::invalid_request(format!(
        "request identifier does not match the active {family_name} request"
    ))
}

impl<Family: ImageFamily> ImageGenerationEngine for ImageEngineShell<Family> {
    fn load(&mut self) -> Result<ImageGenerationEngineLoadResult, ImageGenerationFailureReason> {
        if self.active_request.is_some() {
            return Err(ImageGenerationFailureReason::EngineBusy);
        }
        let loaded = match self.components.load() {
            Ok(loaded) => loaded,
            // Component errors are path-free by contract; keeping their
            // bounded detail makes artifact and memory admission failures
            // actionable at the public load boundary.
            Err(reason) => {
                return Err(ImageGenerationFailureReason::FatalExecution {
                    reason: format!(
                        "{} could not load the selected model: {reason}",
                        Family::FAMILY_NAME
                    )
                    .chars()
                    .take(384)
                    .collect(),
                });
            }
        };
        if loaded.model_id != self.serving_model_id {
            return Err(ImageGenerationFailureReason::FatalExecution {
                reason: format!(
                    "{} loaded under an unexpected serving identity",
                    Family::FAMILY_NAME
                ),
            });
        }
        self.loaded_revision = Some(loaded.revision);
        Ok(
            ImageGenerationEngineLoadResult::new(loaded.model_id, loaded.capabilities)
                .with_minimum_mlx_memory_ceiling_bytes(loaded.minimum_mlx_memory_ceiling_bytes),
        )
    }

    fn start_generation(
        &mut self,
        generation_command: ImageGenerationCommand,
    ) -> Result<(), ImageGenerationFailureReason> {
        if self.loaded_revision.is_none() {
            return Err(ImageGenerationFailureReason::FatalExecution {
                reason: format!(
                    "{} must be loaded before generation starts",
                    Family::FAMILY_NAME
                ),
            });
        }
        if self.active_request.is_some() {
            return Err(ImageGenerationFailureReason::EngineBusy);
        }
        let active_request = Family::start_request(self, generation_command)?;
        self.active_request = Some(active_request);
        Ok(())
    }

    fn advance_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        let active_request = self.take_matching_active_request(request_id)?;
        Family::advance_request(self, active_request)
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), ImageGenerationFailureReason> {
        let Some(active_request) = self.active_request.as_ref() else {
            return Ok(());
        };
        if active_request.request_id() != request_id {
            return Err(mismatched_request_failure(Family::FAMILY_NAME));
        }
        self.active_request = None;
        match self
            .components
            .finalize_request(PerformanceAttributionOutcome::Cancelled, None, None)
        {
            Ok(()) => Ok(()),
            Err(_reason) => Err(ImageGenerationFailureReason::FatalExecution {
                reason: format!(
                    "{} could not cleanly cancel image generation",
                    Family::FAMILY_NAME
                ),
            }),
        }
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.components.take_post_cleanup_memory_telemetry()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.components.collect_mlx_memory_telemetry()
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, ImageGenerationFailureReason> {
        if self.active_request.is_some() {
            return Err(ImageGenerationFailureReason::EngineBusy);
        }
        match self
            .components
            .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
        {
            Ok(adjustment) => Ok(adjustment),
            Err(reason) => Err(ImageGenerationFailureReason::FatalExecution {
                reason: reason.chars().take(256).collect(),
            }),
        }
    }
}

#[cfg(test)]
#[path = "image_engine_shell_tests.rs"]
mod image_engine_shell_tests;

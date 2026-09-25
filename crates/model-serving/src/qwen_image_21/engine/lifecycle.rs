//! The Qwen-Image-2.1 lifecycle owner: load once, serve one request at a time, one boundary
//! per advance, and publish only after cleanup succeeds.

#[cfg(feature = "direct-mlx")]
use std::path::PathBuf;
use std::time::Instant;

use astronomical_ipc_protocol::{
    GeneratedImage, ImageGenerationCommand, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, RequestId,
};

#[cfg(feature = "direct-mlx")]
use crate::QWEN_IMAGE_21_OFFICIAL_MODEL_ID;
use crate::{
    ImageGenerationEngine, ImageGenerationEngineLoadResult, ImageGenerationEngineStep,
    MlxMemoryLimitAdjustment, MlxMemoryTelemetry, PerformanceAttributionOutcome,
    qwen_image_21::engine::components::{
        QwenImage21ComponentLoad, QwenImage21EngineComponents, QwenImage21EngineRequest,
    },
    qwen_image_21::engine::request_validation::validate_official_request,
    qwen_image_21::official_profile::QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
    qwen_image_21::render_contract::QwenImage21RenderAdvance,
};

/// Official Qwen-Image-2.1 engine composed behind the architecture-neutral worker contract.
pub struct QwenImage21ImageEngine {
    serving_model_id: String,
    components: Box<dyn QwenImage21EngineComponents>,
    loaded_revision: Option<String>,
    active_request: Option<ActiveRequest>,
}

impl QwenImage21ImageEngine {
    /// Factory-facing constructor over any component owner.
    pub fn from_components(
        serving_model_id: impl Into<String>,
        components: Box<dyn QwenImage21EngineComponents>,
    ) -> Self {
        Self {
            serving_model_id: serving_model_id.into(),
            components,
            loaded_revision: None,
            active_request: None,
        }
    }

    /// Worker-factory constructor over the native MLX component owner.
    #[cfg(feature = "direct-mlx")]
    #[must_use]
    pub fn from_model_family_factory(
        model_directory: impl Into<PathBuf>,
        provenance: crate::qwen_image_21::artifact::QwenImage21ArtifactProvenance,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        performance_attribution_enabled: bool,
        performance_attribution_log_path: PathBuf,
    ) -> Self {
        Self::from_components(
            QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
            Box::new(
                crate::qwen_image_21::engine::mlx_components::QwenImage21MlxComponents::new(
                    model_directory,
                    provenance,
                    effective_mlx_memory_ceiling_bytes,
                    allocator_cache_memory_limit_bytes,
                    performance_attribution_enabled,
                    performance_attribution_log_path,
                ),
            ),
        )
    }

    /// Test seam: a fake components owner drives the same lifecycle as the native owner.
    #[doc(hidden)]
    pub fn with_components_for_tests(
        serving_model_id: impl Into<String>,
        components: Box<dyn QwenImage21EngineComponents>,
    ) -> Self {
        Self::from_components(serving_model_id, components)
    }

    #[must_use]
    pub fn loaded_revision(&self) -> Option<&str> {
        self.loaded_revision.as_deref()
    }

    fn elapsed_millis(active_request: &ActiveRequest) -> u64 {
        u64::try_from(active_request.started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn fail_active_request(
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
                // Concrete component failures are path-free and bounded here so a user can
                // distinguish artifact, memory, and execution pressure.
                reason: format!("{public_failure_description}: {failure_description}")
                    .chars()
                    .take(512)
                    .collect(),
            },
            Err(_cleanup_failure) => ImageGenerationFailureReason::FatalExecution {
                reason: "Qwen-Image-2.1 image generation and cleanup failed".to_owned(),
            },
        }
    }

    fn progress(
        active_request: &ActiveRequest,
        phase: ImageGenerationPhase,
        completed_steps: u16,
    ) -> ImageGenerationEngineStep {
        ImageGenerationEngineStep::Progress {
            phase,
            completed_steps,
            total_steps: active_request.steps,
            elapsed_millis: Self::elapsed_millis(active_request),
        }
    }
}

impl ImageGenerationEngine for QwenImage21ImageEngine {
    fn load(&mut self) -> Result<ImageGenerationEngineLoadResult, ImageGenerationFailureReason> {
        if self.active_request.is_some() {
            return Err(ImageGenerationFailureReason::EngineBusy);
        }
        let loaded: QwenImage21ComponentLoad = self.components.load().map_err(|reason| {
            ImageGenerationFailureReason::FatalExecution {
                // Component errors are path-free by contract; retaining their bounded detail
                // makes artifact and memory admission failures actionable at the load boundary.
                reason: format!("Qwen-Image-2.1 could not load the selected model: {reason}")
                    .chars()
                    .take(384)
                    .collect(),
            }
        })?;
        if loaded.model_id != self.serving_model_id {
            return Err(ImageGenerationFailureReason::FatalExecution {
                reason: "Qwen-Image-2.1 loaded under an unexpected serving identity".to_owned(),
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
                reason: "Qwen-Image-2.1 must be loaded before generation starts".to_owned(),
            });
        }
        if self.active_request.is_some() {
            return Err(ImageGenerationFailureReason::EngineBusy);
        }
        validate_official_request(&self.serving_model_id, &generation_command)?;
        let engine_request = QwenImage21EngineRequest::new(
            generation_command.prompt,
            generation_command.settings.width_pixels,
            generation_command.settings.height_pixels,
            generation_command.settings.steps,
            generation_command.settings.seed,
        );
        self.components
            .start_request(generation_command.request_id, engine_request)
            .map_err(|reason| {
                self.fail_active_request(
                    reason,
                    "Qwen-Image-2.1 could not prepare the image request",
                )
            })?;
        self.active_request = Some(ActiveRequest {
            request_id: generation_command.request_id,
            width_pixels: generation_command.settings.width_pixels,
            height_pixels: generation_command.settings.height_pixels,
            steps: generation_command.settings.steps,
            seed: generation_command.settings.seed,
            started_at: Instant::now(),
        });
        Ok(())
    }

    fn advance_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        let active_request = self.active_request.take().ok_or_else(|| {
            ImageGenerationFailureReason::invalid_request("no Qwen-Image-2.1 request is active")
        })?;
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            return Err(ImageGenerationFailureReason::invalid_request(
                "request identifier does not match the active Qwen-Image-2.1 request",
            ));
        }
        let advance = match self.components.advance_render() {
            Ok(advance) => advance,
            Err(reason) => {
                let public_failure_description = active_request.public_failure_description();
                return Err(self.fail_active_request(reason, public_failure_description));
            }
        };
        match advance {
            QwenImage21RenderAdvance::Preparing => {
                let progress = Self::progress(&active_request, ImageGenerationPhase::Preparing, 0);
                self.active_request = Some(active_request);
                Ok(progress)
            }
            QwenImage21RenderAdvance::ConditioningCompleted => {
                let progress =
                    Self::progress(&active_request, ImageGenerationPhase::EncodingPrompt, 0);
                self.active_request = Some(active_request);
                Ok(progress)
            }
            QwenImage21RenderAdvance::NoisePrepared => {
                let progress = Self::progress(&active_request, ImageGenerationPhase::Denoising, 0);
                self.active_request = Some(active_request);
                Ok(progress)
            }
            QwenImage21RenderAdvance::DenoisingStep {
                completed_steps,
                total_steps: _,
            } => {
                let completed_steps = u16::try_from(completed_steps).unwrap_or(u16::MAX);
                let progress = Self::progress(
                    &active_request,
                    ImageGenerationPhase::Denoising,
                    completed_steps,
                );
                self.active_request = Some(active_request);
                Ok(progress)
            }
            QwenImage21RenderAdvance::DecodingCompleted => {
                let progress = Self::progress(
                    &active_request,
                    ImageGenerationPhase::Decoding,
                    active_request.steps,
                );
                self.active_request = Some(active_request);
                Ok(progress)
            }
            QwenImage21RenderAdvance::Rendered(rendered) => {
                let png_bytes = rendered.to_png_bytes().map_err(|error| {
                    let _ = self.fail_active_request(
                        error.to_string(),
                        "Qwen-Image-2.1 could not encode the generated image",
                    );
                    ImageGenerationFailureReason::EncodingFailed {
                        reason: "the rendered pixels could not be encoded as PNG".to_owned(),
                    }
                })?;
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
                        reason: "Qwen-Image-2.1 could not finalize the generated image".to_owned(),
                    });
                }
                Ok(ImageGenerationEngineStep::Completed {
                    generated_image: GeneratedImage {
                        mime_type: "image/png".to_owned(),
                        encoded_bytes: png_bytes,
                    },
                    result_metadata: ImageGenerationResultMetadata {
                        width_pixels: active_request.width_pixels,
                        height_pixels: active_request.height_pixels,
                        steps: active_request.steps,
                        guidance_thousandths: QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
                        seed: active_request.seed,
                        elapsed_millis,
                    },
                })
            }
        }
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), ImageGenerationFailureReason> {
        let Some(active_request) = self.active_request.as_ref() else {
            return Ok(());
        };
        if active_request.request_id != request_id {
            return Err(ImageGenerationFailureReason::invalid_request(
                "request identifier does not match the active Qwen-Image-2.1 request",
            ));
        }
        self.active_request = None;
        self.components
            .finalize_request(PerformanceAttributionOutcome::Cancelled, None, None)
            .map_err(|_reason| ImageGenerationFailureReason::FatalExecution {
                reason: "Qwen-Image-2.1 could not cleanly cancel image generation".to_owned(),
            })
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
        self.components
            .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
            .map_err(|reason| ImageGenerationFailureReason::FatalExecution {
                reason: reason.chars().take(256).collect(),
            })
    }
}

struct ActiveRequest {
    request_id: RequestId,
    width_pixels: u32,
    height_pixels: u32,
    steps: u16,
    seed: u64,
    started_at: Instant,
}

impl ActiveRequest {
    /// The bounded public description for a failure surfacing mid-render.
    ///
    /// The render session owns the fine-grained phase; the engine can only name the coarse
    /// stage, which is exactly what a REST caller can act on.
    const fn public_failure_description(&self) -> &'static str {
        "Qwen-Image-2.1 could not complete image generation"
    }
}

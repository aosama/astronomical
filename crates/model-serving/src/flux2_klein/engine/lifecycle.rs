//! Architecture-neutral phase machine that prevents publication before cleanup.

#[cfg(feature = "direct-mlx")]
use std::path::{Path, PathBuf};
use std::time::Instant;

use astronomical_ipc_protocol::{
    GeneratedImage, ImageGenerationCommand, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, RequestId,
};

use crate::{
    ImageGenerationEngine, ImageGenerationEngineLoadResult, ImageGenerationEngineStep,
    MlxMemoryLimitAdjustment, MlxMemoryTelemetry, PerformanceAttributionOutcome,
};

#[cfg(feature = "direct-mlx")]
use super::super::{FLUX2_KLEIN_OFFICIAL_MODEL_ID, Flux2KleinArtifactProvenance};
use super::super::{Flux2KleinFlowSchedule, Flux2KleinImageDimensions, Flux2KleinOfficialProfile};
use super::components::Flux2KleinEngineComponents;
use super::request_validation::validate_official_request;

const IMAGE_TRANSPORT_LIMIT_BYTES: u64 = 32 * 1024 * 1024;

/// Official FLUX.2 Klein engine composed behind the architecture-neutral worker contract.
pub struct Flux2KleinImageEngine {
    serving_model_id: String,
    components: Box<dyn Flux2KleinEngineComponents>,
    loaded_revision: Option<String>,
    active_request: Option<ActiveRequest>,
}

impl Flux2KleinImageEngine {
    /// Factory-facing constructor for the exact official artifact profile.
    #[cfg(feature = "direct-mlx")]
    pub fn new(
        model_directory: impl AsRef<Path>,
        provenance: Flux2KleinArtifactProvenance,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        performance_attribution_enabled: bool,
        performance_attribution_log_path: PathBuf,
    ) -> Self {
        Self::with_components_for_tests(
            FLUX2_KLEIN_OFFICIAL_MODEL_ID,
            Box::new(super::mlx_components::Flux2KleinMlxComponents::new(
                FLUX2_KLEIN_OFFICIAL_MODEL_ID.to_owned(),
                model_directory.as_ref().to_path_buf(),
                provenance,
                effective_mlx_memory_ceiling_bytes,
                allocator_cache_memory_limit_bytes,
                performance_attribution_enabled,
                performance_attribution_log_path,
            )),
        )
    }

    /// Real-artifact seam for `ModelFamilyFactory`; validation and MLX initialization remain in `load`.
    #[cfg(feature = "direct-mlx")]
    pub fn from_model_family_factory(
        model_directory: impl AsRef<Path>,
        provenance: Flux2KleinArtifactProvenance,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        performance_attribution_enabled: bool,
        performance_attribution_log_path: PathBuf,
    ) -> Self {
        Self::new(
            model_directory,
            provenance,
            effective_mlx_memory_ceiling_bytes,
            allocator_cache_memory_limit_bytes,
            performance_attribution_enabled,
            performance_attribution_log_path,
        )
    }

    #[doc(hidden)]
    pub fn with_components_for_tests(
        serving_model_id: impl Into<String>,
        components: Box<dyn Flux2KleinEngineComponents>,
    ) -> Self {
        Self {
            serving_model_id: serving_model_id.into(),
            components,
            loaded_revision: None,
            active_request: None,
        }
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
                // distinguish malformed tensors, unsupported operations, and memory pressure.
                reason: format!("{public_failure_description}: {failure_description}")
                    .chars()
                    .take(512)
                    .collect(),
            },
            Err(_cleanup_failure) => ImageGenerationFailureReason::FatalExecution {
                reason: "FLUX.2 Klein image generation and cleanup failed".to_owned(),
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
            total_steps: Flux2KleinOfficialProfile::inference_step_count() as u16,
            elapsed_millis: Self::elapsed_millis(active_request),
        }
    }
}

impl ImageGenerationEngine for Flux2KleinImageEngine {
    fn load(&mut self) -> Result<ImageGenerationEngineLoadResult, ImageGenerationFailureReason> {
        if self.active_request.is_some() {
            return Err(ImageGenerationFailureReason::EngineBusy);
        }
        let loaded = self.components.load().map_err(|reason| {
            ImageGenerationFailureReason::FatalExecution {
                // Component errors are path-free by contract; retaining their bounded detail makes
                // artifact and memory admission failures actionable at the public load boundary.
                reason: format!("FLUX.2 Klein could not load the selected model: {reason}")
                    .chars()
                    .take(384)
                    .collect(),
            }
        })?;
        if loaded.model_id != self.serving_model_id {
            return Err(ImageGenerationFailureReason::FatalExecution {
                reason: "FLUX.2 Klein loaded under an unexpected serving identity".to_owned(),
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
                reason: "FLUX.2 Klein must be loaded before generation starts".to_owned(),
            });
        }
        if self.active_request.is_some() {
            return Err(ImageGenerationFailureReason::EngineBusy);
        }
        validate_official_request(&self.serving_model_id, &generation_command)?;
        let dimensions = Flux2KleinImageDimensions::validate(
            generation_command.settings.width_pixels,
            generation_command.settings.height_pixels,
            IMAGE_TRANSPORT_LIMIT_BYTES,
        )
        .map_err(|error| ImageGenerationFailureReason::invalid_request(error.to_string()))?;
        let schedule = match self.components.start_request(
            generation_command.request_id,
            dimensions,
            generation_command.settings.seed,
        ) {
            Ok(schedule) => schedule,
            Err(reason) => {
                return Err(self.fail_active_request(
                    reason,
                    "FLUX.2 Klein could not prepare the image request",
                ));
            }
        };
        self.active_request = Some(ActiveRequest {
            request_id: generation_command.request_id,
            prompt: generation_command.prompt,
            seed: generation_command.settings.seed,
            dimensions,
            schedule,
            phase: RequestPhase::Conditioning,
            started_at: Instant::now(),
        });
        Ok(())
    }

    fn advance_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        let mut active_request = self.active_request.take().ok_or_else(|| {
            ImageGenerationFailureReason::invalid_request("no FLUX.2 Klein request is active")
        })?;
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            return Err(ImageGenerationFailureReason::invalid_request(
                "request identifier does not match the active FLUX.2 Klein request",
            ));
        }
        let mut conditioning_completed = None;
        let mut denoising_advance = None;
        let mut decoding_completed = None;
        let operation = match active_request.phase {
            RequestPhase::Conditioning => self
                .components
                .condition_prompt(&active_request.prompt)
                .map(|did_complete_conditioning| {
                    conditioning_completed = Some(did_complete_conditioning);
                }),
            RequestPhase::Noise => self.components.initialize_keyed_noise(
                active_request.seed,
                active_request.schedule.initial_sigma(),
            ),
            RequestPhase::Denoising(step_index) => self
                .components
                .denoise_euler(step_index, active_request.schedule.steps()[step_index])
                .map(|did_complete_euler_step| {
                    denoising_advance = Some(did_complete_euler_step);
                }),
            RequestPhase::Decoding => {
                self.components
                    .decode_latents()
                    .map(|did_complete_decoding| {
                        decoding_completed = Some(did_complete_decoding);
                    })
            }
            RequestPhase::Encoding => {
                let png_bytes = match self.components.encode_png() {
                    Ok(png_bytes) => png_bytes,
                    Err(reason) => {
                        return Err(self.fail_active_request(
                            reason,
                            "FLUX.2 Klein could not encode the generated image",
                        ));
                    }
                };
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
                        reason: "FLUX.2 Klein could not finalize the generated image".to_owned(),
                    });
                }
                return Ok(ImageGenerationEngineStep::Completed {
                    generated_image: GeneratedImage {
                        mime_type: "image/png".to_owned(),
                        encoded_bytes: png_bytes,
                    },
                    result_metadata: ImageGenerationResultMetadata {
                        width_pixels: active_request.dimensions.width_pixels(),
                        height_pixels: active_request.dimensions.height_pixels(),
                        steps: Flux2KleinOfficialProfile::inference_step_count() as u16,
                        guidance_thousandths: Flux2KleinOfficialProfile::guidance_thousandths(),
                        seed: active_request.seed,
                        elapsed_millis,
                    },
                });
            }
        };
        if let Err(reason) = operation {
            let public_failure_description = active_request.phase.public_failure_description();
            return Err(self.fail_active_request(reason, public_failure_description));
        }
        if conditioning_completed == Some(false) {
            let progress = Self::progress(&active_request, ImageGenerationPhase::EncodingPrompt, 0);
            self.active_request = Some(active_request);
            return Ok(progress);
        }
        if denoising_advance == Some(false) {
            let progress = Self::progress(
                &active_request,
                ImageGenerationPhase::Denoising,
                active_request.completed_denoising_steps(),
            );
            self.active_request = Some(active_request);
            return Ok(progress);
        }
        if decoding_completed == Some(false) {
            let progress = Self::progress(
                &active_request,
                ImageGenerationPhase::Decoding,
                Flux2KleinOfficialProfile::inference_step_count() as u16,
            );
            self.active_request = Some(active_request);
            return Ok(progress);
        }
        let (phase, completed_steps) = active_request.advance_phase();
        let progress = Self::progress(&active_request, phase, completed_steps);
        self.active_request = Some(active_request);
        Ok(progress)
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
                "request identifier does not match the active FLUX.2 Klein request",
            ));
        }
        self.active_request = None;
        self.components
            .finalize_request(PerformanceAttributionOutcome::Cancelled, None, None)
            .map_err(|_reason| ImageGenerationFailureReason::FatalExecution {
                reason: "FLUX.2 Klein could not cleanly cancel image generation".to_owned(),
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
    prompt: String,
    seed: u64,
    dimensions: Flux2KleinImageDimensions,
    schedule: Flux2KleinFlowSchedule,
    phase: RequestPhase,
    started_at: Instant,
}

impl ActiveRequest {
    fn completed_denoising_steps(&self) -> u16 {
        match self.phase {
            RequestPhase::Denoising(step_index) => step_index as u16,
            RequestPhase::Decoding | RequestPhase::Encoding => {
                Flux2KleinOfficialProfile::inference_step_count() as u16
            }
            RequestPhase::Conditioning | RequestPhase::Noise => 0,
        }
    }

    fn advance_phase(&mut self) -> (ImageGenerationPhase, u16) {
        match self.phase {
            RequestPhase::Conditioning => {
                self.phase = RequestPhase::Noise;
                (ImageGenerationPhase::EncodingPrompt, 0)
            }
            RequestPhase::Noise => {
                self.phase = RequestPhase::Denoising(0);
                (ImageGenerationPhase::Denoising, 0)
            }
            RequestPhase::Denoising(step_index) => {
                let completed_steps = step_index as u16 + 1;
                self.phase = if step_index + 1 < Flux2KleinOfficialProfile::inference_step_count() {
                    RequestPhase::Denoising(step_index + 1)
                } else {
                    RequestPhase::Decoding
                };
                (ImageGenerationPhase::Denoising, completed_steps)
            }
            RequestPhase::Decoding => {
                self.phase = RequestPhase::Encoding;
                (
                    ImageGenerationPhase::Decoding,
                    Flux2KleinOfficialProfile::inference_step_count() as u16,
                )
            }
            RequestPhase::Encoding => (
                ImageGenerationPhase::EncodingImage,
                Flux2KleinOfficialProfile::inference_step_count() as u16,
            ),
        }
    }
}

#[derive(Clone, Copy)]
enum RequestPhase {
    Conditioning,
    Noise,
    Denoising(usize),
    Decoding,
    Encoding,
}

impl RequestPhase {
    const fn public_failure_description(self) -> &'static str {
        match self {
            Self::Conditioning => "FLUX.2 Klein could not encode the image prompt",
            Self::Noise => "FLUX.2 Klein could not prepare seeded image noise",
            Self::Denoising(_) => "FLUX.2 Klein could not complete image denoising",
            Self::Decoding => "FLUX.2 Klein could not decode the generated image",
            Self::Encoding => "FLUX.2 Klein could not encode the generated image",
        }
    }
}

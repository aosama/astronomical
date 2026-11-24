//! Qwen-Image-2.1 behind the shared image lifecycle: its request checks and
//! how each render boundary of its session is reported. The lifecycle around
//! it (load once, one request at a time, publish only after cleanup) is the
//! shared [`ImageEngineShell`].

#[cfg(feature = "direct-mlx")]
use std::path::PathBuf;

#[cfg(feature = "direct-mlx")]
use image_generation::{
    BuildImageEngine, ImageFamilyConstruction, ImageProvenanceFromEvidence, VerifyImageDirectory,
};
#[cfg(feature = "direct-mlx")]
use model_primitives::FamilyEngineSettings;
#[cfg(feature = "direct-mlx")]
use qwen_image_21_recognition::verify_qwen_image_21_model_directory;
use std::time::Instant;

use ipc_protocol::{
    ImageGenerationCommand, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, RequestId,
};

#[cfg(feature = "direct-mlx")]
use crate::QWEN_IMAGE_21_OFFICIAL_MODEL_ID;
use crate::{
    engine::components::{QwenImage21EngineComponents, QwenImage21EngineRequest},
    engine::request_validation::validate_official_request,
    official_profile::QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
    render_contract::QwenImage21RenderAdvance,
};
use image_generation::ImageGenerationEngineStep;
use image_generation::image_engine_shell::{ActiveImageRequest, ImageEngineShell, ImageFamily};

/// Official Qwen-Image-2.1 engine composed behind the architecture-neutral worker contract.
pub type QwenImage21ImageEngine = ImageEngineShell<QwenImage21ImageFamily>;

pub struct QwenImage21ImageFamily;

/// The worker factory's constructor over the native MLX component owner.
///
/// A free function rather than an inherent constructor because the engine
/// type is the shared shell, which this family crate does not own.
#[cfg(feature = "direct-mlx")]
#[must_use]
pub fn build_qwen_image_21_image_engine(
    model_directory: PathBuf,
    provenance: crate::artifact::QwenImage21ArtifactProvenance,
    family_engine_settings: &FamilyEngineSettings,
) -> QwenImage21ImageEngine {
    QwenImage21ImageEngine::from_components(
        QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
        Box::new(
            crate::engine::mlx_components::QwenImage21MlxComponents::new(
                model_directory,
                provenance,
                family_engine_settings.effective_mlx_memory_ceiling_bytes,
                family_engine_settings.mlx_buffer_cache_limit_bytes,
                family_engine_settings.performance_attribution_enabled,
                family_engine_settings
                    .performance_attribution_log_path
                    .clone(),
            ),
        ),
    )
}

#[cfg(feature = "direct-mlx")]
impl ImageFamilyConstruction for QwenImage21ImageFamily {
    type Provenance = crate::artifact::QwenImage21ArtifactProvenance;
    type DirectoryError = qwen_image_21_recognition::QwenImage21DirectoryVerificationError;
    const VERIFY_DIRECTORY: VerifyImageDirectory<Self> = verify_qwen_image_21_model_directory;
    const PROVENANCE: ImageProvenanceFromEvidence<Self> = Self::Provenance::from_directory_evidence;
    const BUILD_ENGINE: BuildImageEngine<Self> = build_qwen_image_21_image_engine;
}

impl ImageFamily for QwenImage21ImageFamily {
    type Components = dyn QwenImage21EngineComponents;
    type ActiveRequest = QwenImage21ActiveRequest;
    const FAMILY_NAME: &'static str = "Qwen-Image-2.1";

    fn start_request(
        engine: &mut QwenImage21ImageEngine,
        generation_command: ImageGenerationCommand,
    ) -> Result<QwenImage21ActiveRequest, ImageGenerationFailureReason> {
        validate_official_request(&engine.serving_model_id, &generation_command)?;
        let engine_request = QwenImage21EngineRequest::new(
            generation_command.prompt,
            generation_command.settings.width_pixels,
            generation_command.settings.height_pixels,
            generation_command.settings.steps,
            generation_command.settings.seed,
        );
        if let Err(reason) = engine
            .components
            .start_request(generation_command.request_id, engine_request)
        {
            return Err(engine.fail_active_request(
                reason,
                "Qwen-Image-2.1 could not prepare the image request",
            ));
        }
        Ok(QwenImage21ActiveRequest {
            request_id: generation_command.request_id,
            width_pixels: generation_command.settings.width_pixels,
            height_pixels: generation_command.settings.height_pixels,
            steps: generation_command.settings.steps,
            seed: generation_command.settings.seed,
            started_at: Instant::now(),
        })
    }

    /// The render session owns the fine-grained phase; each boundary it
    /// reports maps to one public progress phase.
    fn advance_request(
        engine: &mut QwenImage21ImageEngine,
        active_request: QwenImage21ActiveRequest,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        let advance = match engine.components.advance_render() {
            Ok(advance) => advance,
            Err(reason) => {
                return Err(engine.fail_active_request(
                    reason,
                    "Qwen-Image-2.1 could not complete image generation",
                ));
            }
        };
        let (phase, completed_steps) = match advance {
            QwenImage21RenderAdvance::Preparing => (ImageGenerationPhase::Preparing, 0),
            QwenImage21RenderAdvance::ConditioningCompleted => {
                (ImageGenerationPhase::EncodingPrompt, 0)
            }
            QwenImage21RenderAdvance::NoisePrepared => (ImageGenerationPhase::Denoising, 0),
            QwenImage21RenderAdvance::DenoisingStep {
                completed_steps,
                total_steps: _,
            } => (
                ImageGenerationPhase::Denoising,
                u16::try_from(completed_steps).unwrap_or(u16::MAX),
            ),
            QwenImage21RenderAdvance::DecodingCompleted => {
                (ImageGenerationPhase::Decoding, active_request.steps)
            }
            QwenImage21RenderAdvance::Rendered(rendered) => {
                let png_bytes = match rendered.to_png_bytes() {
                    Ok(png_bytes) => png_bytes,
                    Err(error) => {
                        let _ = engine.fail_active_request(
                            error.to_string(),
                            "Qwen-Image-2.1 could not encode the generated image",
                        );
                        return Err(ImageGenerationFailureReason::EncodingFailed {
                            reason: "the rendered pixels could not be encoded as PNG".to_owned(),
                        });
                    }
                };
                return engine.complete_rendered_request(active_request, png_bytes);
            }
        };
        Ok(engine.restore_with_progress(active_request, phase, completed_steps))
    }
}

pub struct QwenImage21ActiveRequest {
    request_id: RequestId,
    width_pixels: u32,
    height_pixels: u32,
    steps: u16,
    seed: u64,
    started_at: Instant,
}

impl ActiveImageRequest for QwenImage21ActiveRequest {
    fn request_id(&self) -> RequestId {
        self.request_id
    }

    fn started_at(&self) -> Instant {
        self.started_at
    }

    fn total_steps(&self) -> u16 {
        self.steps
    }

    fn result_metadata(&self, elapsed_millis: u64) -> ImageGenerationResultMetadata {
        ImageGenerationResultMetadata {
            width_pixels: self.width_pixels,
            height_pixels: self.height_pixels,
            steps: self.steps,
            guidance_thousandths: QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
            seed: self.seed,
            elapsed_millis,
        }
    }
}

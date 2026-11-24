//! FLUX.2 Klein behind the shared image lifecycle: its request checks and its
//! phase machine (prompt conditioning, keyed noise, Euler denoising, decoding
//! and PNG encoding). The lifecycle around it, which prevents publication
//! before cleanup, is the shared [`ImageEngineShell`].

#[cfg(feature = "direct-mlx")]
use std::path::PathBuf;
use std::time::Instant;

use ipc_protocol::{
    ImageGenerationCommand, ImageGenerationFailureReason, ImageGenerationPhase,
    ImageGenerationResultMetadata, RequestId,
};

#[cfg(feature = "direct-mlx")]
use flux2_klein_recognition::verify_flux2_klein_model_directory;
use image_generation::ImageGenerationEngineStep;
use image_generation::image_engine_shell::{ActiveImageRequest, ImageEngineShell, ImageFamily};
#[cfg(feature = "direct-mlx")]
use image_generation::{
    BuildImageEngine, ImageFamilyConstruction, ImageProvenanceFromEvidence, VerifyImageDirectory,
};
#[cfg(feature = "direct-mlx")]
use model_primitives::FamilyEngineSettings;

#[cfg(feature = "direct-mlx")]
use super::super::Flux2KleinArtifactProvenance;
use super::super::{Flux2KleinFlowSchedule, Flux2KleinImageDimensions, Flux2KleinOfficialProfile};
use super::components::Flux2KleinEngineComponents;
use super::request_validation;

const IMAGE_TRANSPORT_LIMIT_BYTES: u64 = 32 * 1024 * 1024;

/// Official FLUX.2 Klein engine composed behind the architecture-neutral worker contract.
pub type Flux2KleinImageEngine = ImageEngineShell<Flux2KleinImageFamily>;

pub struct Flux2KleinImageFamily;

/// The worker factory's constructor for either reviewed build; the
/// provenance decides which model id the engine serves under. Artifact
/// validation and MLX initialization wait for `load`.
///
/// A free function rather than an inherent constructor because the engine
/// type is the shared shell, which this family crate does not own.
#[cfg(feature = "direct-mlx")]
pub fn build_flux2_klein_image_engine(
    model_directory: PathBuf,
    provenance: Flux2KleinArtifactProvenance,
    family_engine_settings: &FamilyEngineSettings,
) -> Flux2KleinImageEngine {
    let serving_model_id = provenance.serving_model_id();
    Flux2KleinImageEngine::from_components(
        serving_model_id,
        Box::new(super::mlx_components::Flux2KleinMlxComponents::new(
            serving_model_id.to_owned(),
            model_directory,
            provenance,
            family_engine_settings.effective_mlx_memory_ceiling_bytes,
            family_engine_settings.mlx_buffer_cache_limit_bytes,
            family_engine_settings.performance_attribution_enabled,
            family_engine_settings
                .performance_attribution_log_path
                .clone(),
        )),
    )
}

#[cfg(feature = "direct-mlx")]
impl ImageFamilyConstruction for Flux2KleinImageFamily {
    type Provenance = Flux2KleinArtifactProvenance;
    type DirectoryError = flux2_klein_recognition::Flux2KleinDirectoryVerificationError;
    const VERIFY_DIRECTORY: VerifyImageDirectory<Self> = verify_flux2_klein_model_directory;
    const PROVENANCE: ImageProvenanceFromEvidence<Self> = Self::Provenance::from_directory_evidence;
    const BUILD_ENGINE: BuildImageEngine<Self> = build_flux2_klein_image_engine;
}

impl ImageFamily for Flux2KleinImageFamily {
    type Components = dyn Flux2KleinEngineComponents;
    type ActiveRequest = Flux2KleinActiveRequest;
    const FAMILY_NAME: &'static str = "FLUX.2 Klein";

    fn start_request(
        engine: &mut Flux2KleinImageEngine,
        generation_command: ImageGenerationCommand,
    ) -> Result<Flux2KleinActiveRequest, ImageGenerationFailureReason> {
        request_validation::validate_official_request(
            &engine.serving_model_id,
            &generation_command,
        )?;
        let dimensions = Flux2KleinImageDimensions::validate(
            generation_command.settings.width_pixels,
            generation_command.settings.height_pixels,
            IMAGE_TRANSPORT_LIMIT_BYTES,
        )
        .map_err(|error| ImageGenerationFailureReason::invalid_request(error.to_string()))?;
        let schedule = match engine.components.start_request(
            generation_command.request_id,
            dimensions,
            generation_command.settings.seed,
        ) {
            Ok(schedule) => schedule,
            Err(reason) => {
                return Err(engine.fail_active_request(
                    reason,
                    "FLUX.2 Klein could not prepare the image request",
                ));
            }
        };
        Ok(Flux2KleinActiveRequest {
            request_id: generation_command.request_id,
            prompt: generation_command.prompt,
            seed: generation_command.settings.seed,
            dimensions,
            schedule,
            phase: RequestPhase::Conditioning,
            started_at: Instant::now(),
        })
    }

    fn advance_request(
        engine: &mut Flux2KleinImageEngine,
        mut active_request: Flux2KleinActiveRequest,
    ) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
        let phase_outcome = match active_request.phase {
            RequestPhase::Encoding => return complete_encoded_request(engine, active_request),
            _ => run_phase(engine, &active_request),
        };
        let did_complete_phase = match phase_outcome {
            Ok(did_complete_phase) => did_complete_phase,
            Err(reason) => {
                let public_failure_description = active_request.phase.public_failure_description();
                return Err(engine.fail_active_request(reason, public_failure_description));
            }
        };
        if !did_complete_phase {
            let (phase, completed_steps) = active_request.partial_progress();
            return Ok(engine.restore_with_progress(active_request, phase, completed_steps));
        }
        let (phase, completed_steps) = active_request.advance_phase();
        Ok(engine.restore_with_progress(active_request, phase, completed_steps))
    }
}

/// Runs the request's current phase on the components. `Ok(false)` means
/// the phase advanced but did not finish, so the request stays in it.
fn run_phase(
    engine: &mut Flux2KleinImageEngine,
    active_request: &Flux2KleinActiveRequest,
) -> Result<bool, String> {
    match active_request.phase {
        RequestPhase::Conditioning => engine.components.condition_prompt(&active_request.prompt),
        RequestPhase::Noise => engine
            .components
            .initialize_keyed_noise(active_request.seed, active_request.schedule.initial_sigma())
            .map(|()| true),
        RequestPhase::Denoising(step_index) => engine
            .components
            .denoise_euler(step_index, active_request.schedule.steps()[step_index]),
        RequestPhase::Decoding => engine.components.decode_latents(),
        RequestPhase::Encoding => Ok(true),
    }
}

/// Terminal encoding step: a failed encode fails the request; a successful
/// one is finalized and returned by the shell.
fn complete_encoded_request(
    engine: &mut Flux2KleinImageEngine,
    active_request: Flux2KleinActiveRequest,
) -> Result<ImageGenerationEngineStep, ImageGenerationFailureReason> {
    let png_bytes = match engine.components.encode_png() {
        Ok(png_bytes) => png_bytes,
        Err(reason) => {
            return Err(engine
                .fail_active_request(reason, "FLUX.2 Klein could not encode the generated image"));
        }
    };
    engine.complete_rendered_request(active_request, png_bytes)
}

impl ActiveImageRequest for Flux2KleinActiveRequest {
    fn request_id(&self) -> RequestId {
        self.request_id
    }

    fn started_at(&self) -> Instant {
        self.started_at
    }

    fn total_steps(&self) -> u16 {
        Flux2KleinOfficialProfile::inference_step_count() as u16
    }

    fn result_metadata(&self, elapsed_millis: u64) -> ImageGenerationResultMetadata {
        ImageGenerationResultMetadata {
            width_pixels: self.dimensions.width_pixels(),
            height_pixels: self.dimensions.height_pixels(),
            steps: Flux2KleinOfficialProfile::inference_step_count() as u16,
            guidance_thousandths: Flux2KleinOfficialProfile::guidance_thousandths(),
            seed: self.seed,
            elapsed_millis,
        }
    }
}

pub struct Flux2KleinActiveRequest {
    request_id: RequestId,
    prompt: String,
    seed: u64,
    dimensions: Flux2KleinImageDimensions,
    schedule: Flux2KleinFlowSchedule,
    phase: RequestPhase,
    started_at: Instant,
}

impl Flux2KleinActiveRequest {
    fn completed_denoising_steps(&self) -> u16 {
        match self.phase {
            RequestPhase::Denoising(step_index) => step_index as u16,
            RequestPhase::Decoding | RequestPhase::Encoding => {
                Flux2KleinOfficialProfile::inference_step_count() as u16
            }
            RequestPhase::Conditioning | RequestPhase::Noise => 0,
        }
    }

    /// Where a request stands when its current phase advanced without
    /// finishing.
    fn partial_progress(&self) -> (ImageGenerationPhase, u16) {
        match self.phase {
            RequestPhase::Conditioning | RequestPhase::Noise => {
                (ImageGenerationPhase::EncodingPrompt, 0)
            }
            RequestPhase::Denoising(_) => (
                ImageGenerationPhase::Denoising,
                self.completed_denoising_steps(),
            ),
            RequestPhase::Decoding | RequestPhase::Encoding => (
                ImageGenerationPhase::Decoding,
                Flux2KleinOfficialProfile::inference_step_count() as u16,
            ),
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

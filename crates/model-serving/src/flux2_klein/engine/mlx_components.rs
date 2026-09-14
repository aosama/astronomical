//! Native BF16 component owner with sequential text, transformer, and VAE residency.

mod engine_components;
mod loading;
mod request_execution;
mod state;

pub use request_execution::{
    flux2_klein_euler_update_for_tests, flux2_klein_initial_latents_for_tests,
    flux2_klein_keyed_noise_and_euler_for_tests,
};

use std::path::PathBuf;
use std::time::Instant;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::{
    MemoryCeilingUtilization, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttribution, PerformanceAttributionLog,
    PerformanceAttributionOutcome, PerformanceOperation, ValidatedWeightsFile,
};

use super::super::memory_utilization::{
    Flux2KleinMemoryPhase, compose_flux2_klein_memory_ceiling_utilization,
};
use super::super::transformer::{Flux2KleinForwardAdvance, Flux2KleinForwardState};
use super::super::vae::{Flux2KleinVaeDecodeAdvance, Flux2KleinVaeDecodeState};
use super::super::{
    Flux2KleinArtifactProvenance, Flux2KleinArtifactValidator, Flux2KleinFlowSchedule,
    Flux2KleinFlowScheduler, Flux2KleinFlowStep, Flux2KleinImageDimensions,
    Flux2KleinMemoryAdmission, Flux2KleinMemoryGeometry, Flux2KleinPackedLatentLayout,
    Flux2KleinPngEncoder, Flux2KleinResidencyPlan, Flux2KleinTransformer,
    Flux2KleinTransformerInputs, Flux2KleinVaeDecodeMode, Flux2KleinVaeDecoder,
    ValidatedFlux2KleinArtifact,
};
use super::request_geometry::{
    build_position_ids, memory_geometry, official_capabilities, signed_shape,
};
use super::{
    FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH, Flux2KleinTextConditioner,
    Flux2KleinTextConditioning, Flux2KleinTextConditioningAdvance, Flux2KleinTextConditioningState,
    Flux2KleinTokenizer,
};
use super::{Flux2KleinComponentLoad, Flux2KleinEngineComponents};

const TRANSFORMER_BLOCKS_PER_CANCELLATION_GROUP: usize = 1;
const TEXT_ENCODER_LAYERS_PER_CANCELLATION_GROUP: usize = 1;

pub(super) struct Flux2KleinMlxComponents {
    serving_model_id: String,
    model_directory: PathBuf,
    provenance: Flux2KleinArtifactProvenance,
    effective_mlx_memory_ceiling_bytes: usize,
    original_allocator_cache_memory_limit_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    performance_attribution_enabled: bool,
    performance_attribution_log_path: PathBuf,
    performance_attribution_log: Option<PerformanceAttributionLog>,
    runtime: Option<MlxRuntime>,
    validated_artifact: Option<ValidatedFlux2KleinArtifact>,
    residency_plan: Option<Flux2KleinResidencyPlan>,
    transformer_geometry: Option<super::super::Flux2KleinTransformerGeometry>,
    request_attribution: Option<PerformanceAttribution>,
    request_id: Option<RequestId>,
    request_seed: Option<u64>,
    request_start_memory: Option<(u64, u64, u64)>,
    dimensions: Option<Flux2KleinImageDimensions>,
    latent_layout: Option<Flux2KleinPackedLatentLayout>,
    transformer_file: Option<ValidatedWeightsFile>,
    vae_file: Option<ValidatedWeightsFile>,
    text_conditioning_state: Option<Flux2KleinTextConditioningState>,
    conditioning: Option<Flux2KleinTextConditioning>,
    transformer: Option<Flux2KleinTransformer>,
    forward_state: Option<Flux2KleinForwardState>,
    forward_step_index: Option<usize>,
    denoising_step_started_at: Option<Instant>,
    latents: Option<MlxArray>,
    image_position_ids: Option<MlxArray>,
    text_position_ids: Option<MlxArray>,
    vae_decoder: Option<Flux2KleinVaeDecoder>,
    vae_decode_state: Option<Flux2KleinVaeDecodeState>,
    decoded_rgb: Option<MlxArray>,
    memory_geometry: Option<Flux2KleinMemoryGeometry>,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
}

impl Flux2KleinMlxComponents {
    fn new_attribution(&self) -> PerformanceAttribution {
        if self.performance_attribution_enabled {
            PerformanceAttribution::enabled()
        } else {
            PerformanceAttribution::disabled()
        }
    }
    /// The sequential image phase a memory measurement belongs to.
    ///
    /// Request state drives the answer: whichever request-scoped owner is
    /// currently populated marks the phase, and a request with no populated
    /// owner is in the transformer streaming window between conditioning and
    /// the first denoising step. Without a request the engine is idle and no
    /// transient work is promised.
    fn current_memory_phase(&self) -> Flux2KleinMemoryPhase {
        if self.decoded_rgb.is_some() {
            Flux2KleinMemoryPhase::Encoding
        } else if self.vae_decode_state.is_some() {
            Flux2KleinMemoryPhase::VaeDecoding
        } else if self.forward_state.is_some() {
            Flux2KleinMemoryPhase::Denoising
        } else if self.text_conditioning_state.is_some() {
            Flux2KleinMemoryPhase::TextConditioning
        } else if self.request_id.is_some() {
            Flux2KleinMemoryPhase::Denoising
        } else {
            Flux2KleinMemoryPhase::Idle
        }
    }

    /// Component weights the engine knows are resident at this instant.
    ///
    /// The transformer reports its resident payload exactly, the VAE decoder
    /// is all-or-nothing, and completed conditioning taps are durable request
    /// state. The streamed text encoder stays unattributed on purpose: its
    /// streamed layer weights are transient work inside the very reserve the
    /// plan promises for the conditioning phase.
    fn resident_attributed_bytes(&self) -> u64 {
        let Some(memory_geometry) = self.memory_geometry.as_ref() else {
            return 0;
        };
        let transformer_resident_bytes = self.transformer.as_ref().map_or(0, |transformer| {
            transformer.weights().resident_payload_bytes()
        });
        let vae_resident_bytes = if self.vae_decoder.is_some() {
            memory_geometry.vae_payload_bytes
        } else {
            0
        };
        let conditioning_tap_bytes = if self.conditioning.is_some() {
            memory_geometry.conditioning_bytes
        } else {
            0
        };
        transformer_resident_bytes
            .saturating_add(vae_resident_bytes)
            .saturating_add(conditioning_tap_bytes)
    }

    /// Composes the image-lane unused-headroom split for one measurement.
    fn memory_ceiling_utilization(
        &self,
        active_memory_bytes: u64,
    ) -> Option<MemoryCeilingUtilization> {
        let memory_geometry = self.memory_geometry.as_ref()?;
        Some(compose_flux2_klein_memory_ceiling_utilization(
            self.effective_mlx_memory_ceiling_bytes as u64,
            active_memory_bytes,
            self.resident_attributed_bytes(),
            self.current_memory_phase(),
            memory_geometry,
        ))
    }

    fn runtime(&self) -> Result<&MlxRuntime, String> {
        self.runtime
            .as_ref()
            .ok_or_else(|| "the MLX runtime is unavailable".to_owned())
    }

    fn request_attribution(&mut self) -> Result<PerformanceAttribution, String> {
        self.request_attribution
            .take()
            .ok_or_else(|| "the FLUX.2 Klein request attribution owner is unavailable".to_owned())
    }

    fn restore_request_attribution(&mut self, attribution: PerformanceAttribution) {
        self.request_attribution = Some(attribution);
    }

    fn load_transformer_if_needed(
        &mut self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), String> {
        if self.transformer.is_some() {
            return Ok(());
        }
        let retained_block_indices = self
            .residency_plan
            .as_ref()
            .ok_or_else(|| "the FLUX.2 Klein residency plan is unavailable".to_owned())?
            .retained_transformer_block_indices()
            .to_vec();
        let runtime = self
            .runtime
            .take()
            .ok_or_else(|| "the MLX runtime is unavailable".to_owned())?;
        let transformer_file = self
            .transformer_file
            .take()
            .ok_or_else(|| "the validated transformer descriptor is unavailable".to_owned())?;
        let transformer_geometry = self
            .transformer_geometry
            .clone()
            .ok_or_else(|| "the validated transformer geometry is unavailable".to_owned())?;
        match Flux2KleinTransformer::load_with_geometry_and_performance_attribution(
            runtime,
            transformer_file,
            transformer_geometry,
            &retained_block_indices,
            performance_attribution,
        ) {
            Ok(transformer) => {
                self.transformer = Some(transformer);
                Ok(())
            }
            Err(error) => {
                let limits = MlxMemoryLimits::new(
                    self.effective_mlx_memory_ceiling_bytes,
                    self.allocator_cache_memory_limit_bytes,
                )
                .map_err(|limit_error| {
                    format!("{error}; MLX cleanup runtime recovery failed: {limit_error}")
                })?;
                self.runtime = Some(MlxRuntime::initialize(limits).map_err(|runtime_error| {
                    format!("{error}; MLX cleanup runtime recovery failed: {runtime_error}")
                })?);
                Err(error.to_string())
            }
        }
    }

    fn recover_runtime_from_transformer(&mut self) {
        if self.runtime.is_none() {
            self.runtime = self
                .transformer
                .take()
                .map(Flux2KleinTransformer::into_runtime);
        }
    }

    fn reset_request_arrays(&mut self) {
        self.forward_state = None;
        self.forward_step_index = None;
        self.denoising_step_started_at = None;
        self.decoded_rgb = None;
        self.vae_decode_state = None;
        self.vae_decoder = None;
        self.latents = None;
        self.text_conditioning_state = None;
        self.conditioning = None;
        self.image_position_ids = None;
        self.text_position_ids = None;
        self.transformer_file = None;
        self.vae_file = None;
        self.latent_layout = None;
        self.dimensions = None;
        self.request_id = None;
        self.request_seed = None;
        self.request_start_memory = None;
    }

    fn replenish_validated_descriptors(&mut self) -> Result<(), String> {
        if self.validated_artifact.is_some() {
            return Ok(());
        }
        self.validated_artifact = Some(
            Flux2KleinArtifactValidator::new()
                .validate(&self.model_directory, self.provenance.clone())
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    }
}

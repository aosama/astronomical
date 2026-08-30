//! Native BF16 component owner with sequential text, transformer, and VAE residency.

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
    MlxMemoryLimitAdjustment, MlxMemoryTelemetry, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionLog, PerformanceAttributionOutcome,
    PerformanceOperation, ValidatedWeightsFile,
};

use super::super::transformer::{Flux2KleinForwardAdvance, Flux2KleinForwardState};
use super::super::vae::{Flux2KleinVaeDecodeAdvance, Flux2KleinVaeDecodeState};
use super::super::{
    Flux2KleinArtifactProvenance, Flux2KleinArtifactValidator, Flux2KleinFlowSchedule,
    Flux2KleinFlowScheduler, Flux2KleinFlowStep, Flux2KleinImageDimensions,
    Flux2KleinMemoryAdmission, Flux2KleinPackedLatentLayout, Flux2KleinPngEncoder,
    Flux2KleinResidencyPlan, Flux2KleinTransformer, Flux2KleinTransformerInputs,
    Flux2KleinVaeDecodeMode, Flux2KleinVaeDecoder, ValidatedFlux2KleinArtifact,
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

impl Flux2KleinEngineComponents for Flux2KleinMlxComponents {
    fn load(&mut self) -> Result<Flux2KleinComponentLoad, String> {
        self.load_inner()
    }

    fn start_request(
        &mut self,
        request_id: RequestId,
        dimensions: Flux2KleinImageDimensions,
        seed: u64,
    ) -> Result<Flux2KleinFlowSchedule, String> {
        if self.request_id.is_some() {
            return Err("a FLUX.2 Klein request is already active".to_owned());
        }
        self.replenish_validated_descriptors()?;
        self.request_attribution = Some(self.new_attribution());
        self.post_cleanup_memory_telemetry = None;
        let mut request_attribution = self.request_attribution()?;
        let schedule_result = request_attribution.measure_operation(
            PerformanceOperation::ImageScheduleConstruction,
            |_| {
                Flux2KleinFlowScheduler::schedule(
                    dimensions.width_pixels(),
                    dimensions.height_pixels(),
                )
                .map_err(|error| error.to_string())
            },
        );
        self.restore_request_attribution(request_attribution);
        let schedule = schedule_result?;
        self.request_id = Some(request_id);
        self.request_seed = Some(seed);
        self.request_start_memory = if self.performance_attribution_enabled {
            self.runtime()?.memory_snapshot().ok().map(|snapshot| {
                (
                    snapshot.active_memory_bytes() as u64,
                    snapshot.allocator_cache_memory_bytes() as u64,
                    snapshot.peak_memory_bytes() as u64,
                )
            })
        } else {
            None
        };
        self.dimensions = Some(dimensions);
        self.latent_layout = Some(
            Flux2KleinPackedLatentLayout::for_image_dimensions(1, &dimensions)
                .map_err(|error| error.to_string())?,
        );
        Ok(schedule)
    }

    fn condition_prompt(&mut self, prompt: &str) -> Result<bool, String> {
        let mut attribution = self.request_attribution()?;
        let operation_result = (|| {
            if self.text_conditioning_state.is_none() {
                let artifact = self.validated_artifact.take().ok_or_else(|| {
                    "the validated FLUX.2 Klein artifact is unavailable".to_owned()
                })?;
                let retained_files = artifact
                    .into_retained_files()
                    .map_err(|error| error.to_string())?;
                let tokenizer = Flux2KleinTokenizer::from_retained_sidecars(
                    retained_files.tokenizer_sidecars(),
                )
                .map_err(|error| error.to_string())?;
                let (text_shards, transformer_file, vae_file) = retained_files.into_weight_files();
                let text_encoder_mode = self
                    .residency_plan
                    .as_ref()
                    .map(Flux2KleinResidencyPlan::text_encoder_mode)
                    .ok_or_else(|| "the FLUX.2 Klein residency plan is unavailable".to_owned())?;
                let conditioner = Flux2KleinTextConditioner::load(
                    self.runtime()?,
                    tokenizer,
                    text_shards,
                    text_encoder_mode,
                    &mut attribution,
                )
                .map_err(|error| error.to_string())?;
                let conditioning_state = conditioner
                    .start(self.runtime()?, &[prompt.to_owned()], &mut attribution)
                    .map_err(|error| error.to_string())?;
                self.text_conditioning_state = Some(conditioning_state);
                self.transformer_file = Some(transformer_file);
                self.vae_file = Some(vae_file);
            }
            let conditioning_state = self
                .text_conditioning_state
                .take()
                .ok_or_else(|| "text conditioning state is unavailable".to_owned())?;
            match conditioning_state
                .advance_layer_group(
                    self.runtime()?,
                    TEXT_ENCODER_LAYERS_PER_CANCELLATION_GROUP,
                    &mut attribution,
                )
                .map_err(|error| error.to_string())?
            {
                Flux2KleinTextConditioningAdvance::LayerGroupCompleted(conditioning_state) => {
                    self.text_conditioning_state = Some(conditioning_state);
                    Ok(false)
                }
                Flux2KleinTextConditioningAdvance::ConditioningCompleted(conditioning) => {
                    if conditioning.batch_size() != 1
                        || conditioning.sequence_length()
                            != FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH
                        || conditioning.attention_mask().shape()[0] != 1
                    {
                        return Err(
                            "text conditioning produced incompatible request geometry".to_owned()
                        );
                    }
                    self.conditioning = Some(conditioning);
                    Ok(true)
                }
            }
        })();
        let cleanup_result = if operation_result
            .as_ref()
            .is_ok_and(|is_complete| *is_complete)
        {
            attribution.measure_operation(PerformanceOperation::MlxAllocatorCacheCleanup, |_| {
                self.runtime()?
                    .synchronize_gpu_stream_and_clear_allocator_cache()
                    .map_err(|error| error.to_string())
            })
        } else {
            Ok(())
        };
        self.restore_request_attribution(attribution);
        let is_complete = operation_result?;
        cleanup_result?;
        Ok(is_complete)
    }

    fn initialize_keyed_noise(&mut self, seed: u64, initial_sigma: f64) -> Result<(), String> {
        let mut attribution = self.request_attribution()?;
        let operation_result =
            self.initialize_keyed_noise_inner(seed, initial_sigma, &mut attribution);
        self.restore_request_attribution(attribution);
        operation_result
    }

    fn denoise_euler(
        &mut self,
        step_index: usize,
        flow_step: Flux2KleinFlowStep,
    ) -> Result<bool, String> {
        let mut attribution = self.request_attribution()?;
        let denoising_started_at = self
            .denoising_step_started_at
            .take()
            .or_else(|| attribution.begin_operation_span());
        let operation_result = self.denoise_euler_inner(step_index, flow_step, &mut attribution);
        if operation_result
            .as_ref()
            .is_ok_and(|did_complete_step| *did_complete_step)
            || operation_result.is_err()
        {
            attribution.complete_operation_span(
                PerformanceOperation::ImageDenoisingStepSpan,
                denoising_started_at,
            );
        } else {
            self.denoising_step_started_at = denoising_started_at;
        }
        self.restore_request_attribution(attribution);
        operation_result
    }

    fn decode_latents(&mut self) -> Result<bool, String> {
        let mut attribution = self.request_attribution()?;
        let operation_result = self.decode_latents_inner(&mut attribution);
        self.restore_request_attribution(attribution);
        operation_result
    }

    fn encode_png(&mut self) -> Result<Vec<u8>, String> {
        let mut attribution = self.request_attribution()?;
        let operation_result = (|| {
            let dimensions = self
                .dimensions
                .ok_or_else(|| "image dimensions are unavailable".to_owned())?;
            Flux2KleinPngEncoder::encode_decoded_mlx_rgb_with_performance_attribution(
                self.runtime()?,
                self.decoded_rgb
                    .as_ref()
                    .ok_or_else(|| "decoded RGB state is unavailable".to_owned())?,
                dimensions.width_pixels(),
                dimensions.height_pixels(),
                &mut attribution,
            )
            .map_err(|error| error.to_string())
        })();
        self.restore_request_attribution(attribution);
        operation_result
    }

    fn finalize_request(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        encoded_bytes: Option<u64>,
        failure_description: Option<&str>,
    ) -> Result<(), String> {
        self.finalize_request_inner(outcome, encoded_bytes, failure_description)
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.post_cleanup_memory_telemetry.take()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.runtime
            .as_ref()
            .and_then(|runtime| runtime.memory_snapshot().ok())
            .map(|snapshot| {
                MlxMemoryTelemetry::new(
                    snapshot.active_memory_bytes() as u64,
                    snapshot.allocator_cache_memory_bytes() as u64,
                    snapshot.peak_memory_bytes() as u64,
                    crate::MlxActiveMemoryBreakdown::default(),
                )
            })
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, String> {
        if self.request_id.is_some() {
            return Err("memory limits cannot change during image generation".to_owned());
        }
        let requested_ceiling = usize::try_from(requested_mlx_memory_ceiling_bytes)
            .map_err(|_| "the requested MLX memory ceiling exceeds this platform".to_owned())?;
        self.replenish_validated_descriptors()?;
        let geometry = memory_geometry(
            self.validated_artifact
                .as_ref()
                .ok_or_else(|| "the validated FLUX.2 Klein artifact is unavailable".to_owned())?,
        )?;
        let residency_plan =
            Flux2KleinMemoryAdmission::plan(requested_mlx_memory_ceiling_bytes, &geometry)
                .map_err(|error| error.to_string())?;
        let allocator_cache_memory_limit_bytes = allocator_cache_limit_for_ceiling(
            self.original_allocator_cache_memory_limit_bytes,
            requested_ceiling,
        );
        let memory_limits =
            MlxMemoryLimits::new(requested_ceiling, allocator_cache_memory_limit_bytes)
                .map_err(|error| error.to_string())?;
        self.runtime
            .as_mut()
            .ok_or_else(|| "the MLX runtime is unavailable".to_owned())?
            .update_memory_limits(memory_limits)
            .map_err(|error| error.to_string())?;
        self.effective_mlx_memory_ceiling_bytes = requested_ceiling;
        self.allocator_cache_memory_limit_bytes = allocator_cache_memory_limit_bytes;
        let minimum_mlx_memory_ceiling_bytes = residency_plan.minimum_mlx_memory_ceiling_bytes();
        self.residency_plan = Some(residency_plan);
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            allocator_cache_memory_limit_bytes as u64,
            minimum_mlx_memory_ceiling_bytes,
            ExpertMemoryMode::Resident,
            self.collect_mlx_memory_telemetry(),
        ))
    }
}

fn allocator_cache_limit_for_ceiling(
    original_allocator_cache_memory_limit_bytes: usize,
    requested_mlx_memory_ceiling_bytes: usize,
) -> usize {
    original_allocator_cache_memory_limit_bytes.min(requested_mlx_memory_ceiling_bytes)
}

//! The `Flux2KleinEngineComponents` trait implementation for the MLX
//! component set: request execution, telemetry, and live memory-ceiling
//! adjustment. Split from the construction code in the parent so each file
//! stays inside the source-size budget; as a child module it can reach the
//! component set's private fields and the parent's imports directly.

use ipc_protocol::ExpertMemoryMode;
use runtime_integration::MlxMemorySnapshot;

use super::*;

impl ImageEngineComponents for Flux2KleinMlxComponents {
    fn load(&mut self) -> Result<Flux2KleinComponentLoad, String> {
        self.load_inner()
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
            .map(|snapshot| self.memory_telemetry_for_snapshot(&snapshot))
    }

    fn adopt_floor_raise_limit(&mut self, floor_raise_limit_bytes: u64) {
        self.floor_raise_limit_bytes = floor_raise_limit_bytes;
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
        let mlx_buffer_cache_limit_bytes = mlx_buffer_cache_limit_for_ceiling(
            self.original_mlx_buffer_cache_limit_bytes,
            requested_ceiling,
        );
        let memory_limits = MlxMemoryLimits::new(requested_ceiling, mlx_buffer_cache_limit_bytes)
            .map_err(|error| error.to_string())?;
        self.runtime
            .as_mut()
            .ok_or_else(|| "the MLX runtime is unavailable".to_owned())?
            .update_memory_limits(memory_limits)
            .map_err(|error| error.to_string())?;
        self.effective_mlx_memory_ceiling_bytes = requested_ceiling;
        self.mlx_buffer_cache_limit_bytes = mlx_buffer_cache_limit_bytes;
        let minimum_mlx_memory_ceiling_bytes = residency_plan.minimum_mlx_memory_ceiling_bytes();
        self.residency_plan = Some(residency_plan);
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_limit_bytes as u64,
            minimum_mlx_memory_ceiling_bytes,
            ExpertMemoryMode::Resident,
            self.collect_mlx_memory_telemetry(),
        ))
    }
}

impl Flux2KleinEngineComponents for Flux2KleinMlxComponents {
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
                    snapshot.mlx_buffer_cache_bytes() as u64,
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
        let operation_result = self.advance_text_conditioning(prompt, &mut attribution);
        let cleanup_result = if operation_result
            .as_ref()
            .is_ok_and(|is_complete| *is_complete)
        {
            self.synchronize_and_clear_mlx_buffer_cache(&mut attribution)
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
}

impl Flux2KleinMlxComponents {
    /// Reads one MLX memory snapshot as telemetry, with the image-lane split of unused headroom
    /// when the memory geometry is known.
    pub(super) fn memory_telemetry_for_snapshot(
        &self,
        snapshot: &MlxMemorySnapshot,
    ) -> MlxMemoryTelemetry {
        let mut telemetry = MlxMemoryTelemetry::new(
            snapshot.active_memory_bytes() as u64,
            snapshot.mlx_buffer_cache_bytes() as u64,
            snapshot.peak_memory_bytes() as u64,
            memory_policy::telemetry::MlxActiveMemoryBreakdown::default(),
        );
        if let Some(memory_ceiling_utilization) =
            self.memory_ceiling_utilization(snapshot.active_memory_bytes() as u64)
        {
            telemetry = telemetry.with_memory_ceiling_utilization(memory_ceiling_utilization);
        }
        telemetry
    }

    fn synchronize_and_clear_mlx_buffer_cache(
        &self,
        attribution: &mut PerformanceAttribution,
    ) -> Result<(), String> {
        attribution.measure_operation(PerformanceOperation::MlxBufferCacheCleanup, |_| {
            self.runtime()?
                .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
                .map_err(|error| error.to_string())
        })
    }

    /// Runs one text-conditioning boundary and reports whether conditioning is now complete.
    /// The first boundary of a request also loads the text conditioner and starts the prompt.
    fn advance_text_conditioning(
        &mut self,
        prompt: &str,
        attribution: &mut PerformanceAttribution,
    ) -> Result<bool, String> {
        if self.text_conditioning_state.is_none() {
            self.start_text_conditioning(prompt, attribution)?;
        }
        let conditioning_state = self
            .text_conditioning_state
            .take()
            .ok_or_else(|| "text conditioning state is unavailable".to_owned())?;
        match conditioning_state
            .advance_layer_group(
                self.runtime()?,
                TEXT_ENCODER_LAYERS_PER_CANCELLATION_GROUP,
                attribution,
            )
            .map_err(|error| error.to_string())?
        {
            Flux2KleinTextConditioningAdvance::LayerGroupCompleted(conditioning_state) => {
                self.text_conditioning_state = Some(conditioning_state);
                Ok(false)
            }
            Flux2KleinTextConditioningAdvance::ConditioningCompleted(conditioning) => {
                self.store_completed_conditioning(conditioning)
            }
        }
    }

    /// Turns the retained artifact into the text conditioner, starts the prompt on it, and
    /// hands the transformer and VAE weight files on to the later phases.
    fn start_text_conditioning(
        &mut self,
        prompt: &str,
        attribution: &mut PerformanceAttribution,
    ) -> Result<(), String> {
        let artifact = self
            .validated_artifact
            .take()
            .ok_or_else(|| "the validated FLUX.2 Klein artifact is unavailable".to_owned())?;
        let retained_files = artifact
            .into_retained_files()
            .map_err(|error| error.to_string())?;
        let tokenizer =
            Flux2KleinTokenizer::from_retained_sidecars(retained_files.tokenizer_sidecars())
                .map_err(|error| error.to_string())?;
        let (text_encoder_weight_files, transformer_weight_files, vae_weight_files) =
            retained_files.into_weight_files();
        let text_encoder_mode = self
            .residency_plan
            .as_ref()
            .map(Flux2KleinResidencyPlan::text_encoder_mode)
            .ok_or_else(|| "the FLUX.2 Klein residency plan is unavailable".to_owned())?;
        let conditioner = Flux2KleinTextConditioner::load(
            self.runtime()?,
            tokenizer,
            text_encoder_weight_files,
            text_encoder_mode,
            attribution,
        )
        .map_err(|error| error.to_string())?;
        let conditioning_state = conditioner
            .start(self.runtime()?, &[prompt.to_owned()], attribution)
            .map_err(|error| error.to_string())?;
        self.text_conditioning_state = Some(conditioning_state);
        self.transformer_weight_files = Some(transformer_weight_files);
        self.vae_weight_files = Some(vae_weight_files);
        Ok(())
    }

    fn store_completed_conditioning(
        &mut self,
        conditioning: Flux2KleinTextConditioning,
    ) -> Result<bool, String> {
        if !conditioning_describes_one_request(
            conditioning.batch_size(),
            conditioning.sequence_length(),
            &conditioning.attention_mask().shape(),
        ) {
            return Err("text conditioning produced incompatible request geometry".to_owned());
        }
        self.conditioning = Some(conditioning);
        Ok(true)
    }
}

/// Whether finished text conditioning covers exactly one request: a batch of one, the fixed
/// conditioning sequence length, and an attention mask whose leading dimension is one. The mask
/// arrives as its shape, and a zero-dimensional mask has no leading dimension, so it is rejected
/// like any other wrong geometry instead of being indexed.
fn conditioning_describes_one_request(
    batch_size: usize,
    sequence_length: usize,
    attention_mask_shape: &[i32],
) -> bool {
    batch_size == 1
        && sequence_length == FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH
        && attention_mask_shape.first() == Some(&1)
}

fn mlx_buffer_cache_limit_for_ceiling(
    original_mlx_buffer_cache_limit_bytes: usize,
    requested_mlx_memory_ceiling_bytes: usize,
) -> usize {
    original_mlx_buffer_cache_limit_bytes.min(requested_mlx_memory_ceiling_bytes)
}

use image_generation::image_engine_shell::ImageEngineComponents;
use memory_policy::telemetry::MlxMemoryLimitAdjustment;

#[cfg(test)]
#[path = "engine_components_tests.rs"]
mod engine_components_tests;

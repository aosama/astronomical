//! The `Flux2KleinEngineComponents` trait implementation for the MLX
//! component set: request execution, telemetry, and live memory-ceiling
//! adjustment. Split from the construction code in the parent so each file
//! stays inside the source-size budget; as a child module it can reach the
//! component set's private fields and the parent's imports directly.

use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::MlxMemoryLimitAdjustment;

use super::*;

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
                let mut telemetry = MlxMemoryTelemetry::new(
                    snapshot.active_memory_bytes() as u64,
                    snapshot.allocator_cache_memory_bytes() as u64,
                    snapshot.peak_memory_bytes() as u64,
                    crate::MlxActiveMemoryBreakdown::default(),
                );
                if let Some(memory_ceiling_utilization) =
                    self.memory_ceiling_utilization(snapshot.active_memory_bytes() as u64)
                {
                    telemetry =
                        telemetry.with_memory_ceiling_utilization(memory_ceiling_utilization);
                }
                telemetry
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

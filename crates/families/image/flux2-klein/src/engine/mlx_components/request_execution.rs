//! Request-only MLX graph construction, synchronization, release, and report completion.

use model_primitives::performance_attribution::PerformanceAttributionReport;
use runtime_integration::MlxMemorySnapshot;

use super::*;

mod denoising_inputs;
mod finished_request;

use denoising_inputs::DenoisingInputs;
use finished_request::{
    FinishedRequest, bounded_failure_description, memory_bytes_from_snapshot, memory_readings,
};

impl Flux2KleinMlxComponents {
    pub(super) fn initialize_keyed_noise_inner(
        &mut self,
        seed: u64,
        initial_sigma: f64,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), String> {
        let layout = self
            .latent_layout
            .ok_or_else(|| "latent layout is unavailable".to_owned())?;
        let shape = signed_shape(&layout.packed_shape())?;
        let runtime = self.runtime()?;
        let latents = performance_attribution.measure_operation(
            PerformanceOperation::SeededNoiseGraphConstruction,
            |_| {
                build_keyed_initial_latents(runtime, seed, &shape, initial_sigma)
                    .map_err(|error| error.to_string())
            },
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::SeededNoiseSynchronizationWait,
            |_| {
                runtime
                    .evaluate_arrays(&[&latents])
                    .map_err(|error| error.to_string())
            },
        )?;
        let (image_position_ids, text_position_ids) = performance_attribution
            .measure_operation(PerformanceOperation::ImagePositionGraphConstruction, |_| {
                build_position_ids(runtime, layout)
            })?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImagePositionSynchronizationWait,
            |_| {
                runtime
                    .evaluate_arrays(&[&image_position_ids, &text_position_ids])
                    .map_err(|error| error.to_string())
            },
        )?;
        self.latents = Some(latents);
        self.image_position_ids = Some(image_position_ids);
        self.text_position_ids = Some(text_position_ids);
        Ok(())
    }

    pub(super) fn denoise_euler_inner(
        &mut self,
        step_index: usize,
        flow_step: Flux2KleinFlowStep,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<bool, String> {
        self.load_transformer_if_needed(performance_attribution)?;
        if self
            .forward_step_index
            .is_some_and(|active_step_index| active_step_index != step_index)
        {
            return Err(
                "transformer forward state belongs to a different denoising step".to_owned(),
            );
        }
        let inputs = DenoisingInputs::bind(
            self.transformer.as_ref(),
            self.latents.as_ref(),
            self.conditioning.as_ref(),
            self.image_position_ids.as_ref(),
            self.text_position_ids.as_ref(),
        )?;
        let forward_state = match self.forward_state.take() {
            Some(forward_state) => forward_state,
            None => {
                inputs.start_forward_state(&flow_step, step_index, &mut self.forward_step_index)?
            }
        };
        let output = match inputs.advance_block_group(forward_state, performance_attribution)? {
            Flux2KleinForwardAdvance::BlockGroupCompleted(forward_state) => {
                self.forward_state = Some(forward_state);
                return Ok(false);
            }
            Flux2KleinForwardAdvance::ForwardCompleted(output) => output,
        };
        self.forward_step_index = None;
        let updated_latents =
            inputs.euler_updated_latents(&output, &flow_step, performance_attribution)?;
        self.latents = Some(updated_latents);
        Ok(true)
    }

    pub(super) fn decode_latents_inner(
        &mut self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<bool, String> {
        if self.vae_decoder.is_none() {
            performance_attribution
                .measure_operation(PerformanceOperation::ImageTransformerRelease, |_| {
                    self.recover_runtime_from_transformer()
                });
            self.conditioning = None;
            self.image_position_ids = None;
            self.text_position_ids = None;
            let vae_weight_files = self
                .vae_weight_files
                .take()
                .ok_or_else(|| "the validated VAE descriptor is unavailable".to_owned())?;
            performance_attribution.measure_operation(
                PerformanceOperation::MlxBufferCacheCleanup,
                |_| {
                    self.runtime()?
                        .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
                        .map_err(|error| error.to_string())
                },
            )?;
            let decoder = Flux2KleinVaeDecoder::load_with_performance_attribution(
                self.runtime()?,
                vae_weight_files,
                performance_attribution,
            )
            .map_err(|error| error.to_string())?;
            let layout = self
                .latent_layout
                .ok_or_else(|| "latent layout is unavailable".to_owned())?;
            let decode_state = decoder
                .start_decode_packed_latents_with_performance_attribution(
                    self.runtime()?,
                    self.latents
                        .as_ref()
                        .ok_or_else(|| "latent state is unavailable".to_owned())?,
                    layout,
                    Flux2KleinVaeDecodeMode::Complete,
                    performance_attribution,
                )
                .map_err(|error| error.to_string())?;
            self.vae_decoder = Some(decoder);
            self.vae_decode_state = Some(decode_state);
            return Ok(false);
        }

        let decode_state = self
            .vae_decode_state
            .take()
            .ok_or_else(|| "VAE decode state is unavailable".to_owned())?;
        let decoder = self
            .vae_decoder
            .as_ref()
            .ok_or_else(|| "VAE decoder is unavailable".to_owned())?;
        match decoder
            .advance_decode_with_performance_attribution(
                self.runtime()?,
                decode_state,
                performance_attribution,
            )
            .map_err(|error| error.to_string())?
        {
            Flux2KleinVaeDecodeAdvance::Decoding(next_state) => {
                self.vae_decode_state = Some(next_state);
                Ok(false)
            }
            Flux2KleinVaeDecodeAdvance::PixelsReady(decoded_rgb) => {
                self.latents = None;
                self.vae_decoder = None;
                performance_attribution.measure_operation(
                    PerformanceOperation::MlxBufferCacheCleanup,
                    |_| {
                        self.runtime()?
                            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
                            .map_err(|error| error.to_string())
                    },
                )?;
                self.decoded_rgb = Some(decoded_rgb);
                Ok(true)
            }
        }
    }

    pub(super) fn finalize_request_inner(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        encoded_bytes: Option<u64>,
        failure_description: Option<&str>,
    ) -> Result<(), String> {
        self.post_cleanup_memory_telemetry = None;
        let mut attribution = self
            .request_attribution
            .take()
            .unwrap_or_else(PerformanceAttribution::disabled);
        let mut finished_request = self.release_request_state(&mut attribution);
        let memory_snapshot = self.synchronize_and_sample_memory(&mut attribution, outcome)?;
        if let Some(snapshot) = memory_snapshot.as_ref() {
            self.post_cleanup_memory_telemetry = Some(self.memory_telemetry_for_snapshot(snapshot));
        }
        finished_request.final_memory = memory_snapshot.as_ref().map(memory_bytes_from_snapshot);
        let report = self.finish_image_report(
            attribution,
            outcome,
            encoded_bytes,
            failure_description,
            &finished_request,
        );
        if let Some(report) = report
            && let Some(attribution_log) = self.performance_attribution_log.as_mut()
        {
            attribution_log
                .record(&report)
                .map_err(|error| error.to_string())?;
        }
        self.replenish_validated_descriptors()
    }

    /// Drops every request owner and returns the facts the report still needs from them.
    fn release_request_state(
        &mut self,
        attribution: &mut PerformanceAttribution,
    ) -> FinishedRequest {
        // Dropping every request owner first ensures cancellation retires no graph that can
        // later publish partial text, denoising state, or decoded pixels.
        self.forward_state = None;
        self.forward_step_index = None;
        self.denoising_step_started_at = None;
        if self.transformer.is_some() {
            attribution.measure_operation(PerformanceOperation::ImageTransformerRelease, |_| {
                self.recover_runtime_from_transformer()
            });
        }
        let finished_request = FinishedRequest {
            request_id: self.request_id.map(RequestId::value).unwrap_or(0),
            seed: self.request_seed.unwrap_or(0),
            dimensions: self.dimensions,
            start_memory: self.request_start_memory,
            final_memory: None,
        };
        attribution.measure_operation(PerformanceOperation::ImageComponentRelease, |_| {
            self.reset_request_arrays()
        });
        finished_request
    }

    /// Waits for the GPU stream, clears the `mlx_buffer_cache`, and reads MLX memory afterwards.
    /// A cancelled request is measured under its own operation name.
    fn synchronize_and_sample_memory(
        &self,
        attribution: &mut PerformanceAttribution,
        outcome: PerformanceAttributionOutcome,
    ) -> Result<Option<MlxMemorySnapshot>, String> {
        let runtime = self.runtime()?;
        let cleanup_operation = if outcome == PerformanceAttributionOutcome::Cancelled {
            PerformanceOperation::ImageCancellationSynchronization
        } else {
            PerformanceOperation::ImageFinalCleanup
        };
        attribution
            .measure_operation(cleanup_operation, |_| {
                runtime.synchronize_gpu_stream()?;
                runtime.clear_mlx_buffer_cache()
            })
            .map_err(|error| error.to_string())?;
        Ok(runtime.memory_snapshot().ok())
    }

    /// Finishes the image-generation report, or returns `None` when attribution is off.
    fn finish_image_report(
        &self,
        attribution: PerformanceAttribution,
        outcome: PerformanceAttributionOutcome,
        encoded_bytes: Option<u64>,
        failure_description: Option<&str>,
        finished_request: &FinishedRequest,
    ) -> Option<PerformanceAttributionReport> {
        if !attribution.is_enabled() {
            return None;
        }
        let (width_pixels, height_pixels) = finished_request.image_size_pixels();
        attribution.finish_image_generation(
            outcome,
            finished_request.request_id,
            self.serving_model_id.clone(),
            self.provenance.revision().to_owned(),
            width_pixels,
            height_pixels,
            4,
            1_000,
            finished_request.seed,
            encoded_bytes,
            memory_readings(finished_request.start_memory),
            memory_readings(finished_request.final_memory),
            bounded_failure_description(failure_description),
        )
    }
}

fn build_keyed_initial_latents(
    runtime: &MlxRuntime,
    seed: u64,
    shape: &[i32],
    initial_sigma: f64,
) -> Result<MlxArray, runtime_integration::MlxRuntimeError> {
    let key = runtime.random_key(seed)?;
    let noise = runtime.random_normal(shape, MlxDtype::BFloat16, 0.0, 1.0, &key)?;
    Ok(runtime.multiply_scalar(&noise, initial_sigma as f32)?)
}

/// Exposes production shape, schedule, and keyed-noise construction to external acceptance.
pub fn flux2_klein_initial_latents_for_tests(
    runtime: &MlxRuntime,
    seed: u64,
    dimensions: &Flux2KleinImageDimensions,
) -> Result<MlxArray, String> {
    let layout = Flux2KleinPackedLatentLayout::for_image_dimensions(1, dimensions)
        .map_err(|error| error.to_string())?;
    let shape = signed_shape(&layout.packed_shape())?;
    let schedule =
        Flux2KleinFlowScheduler::schedule(dimensions.width_pixels(), dimensions.height_pixels())
            .map_err(|error| error.to_string())?;
    let latents = build_keyed_initial_latents(runtime, seed, &shape, schedule.initial_sigma())
        .map_err(|error| error.to_string())?;
    runtime
        .evaluate_arrays(&[&latents])
        .map_err(|error| error.to_string())?;
    Ok(latents)
}

/// Direct-MLX oracle for request-keyed BF16 noise followed by one Euler update.
pub fn flux2_klein_keyed_noise_and_euler_for_tests(
    runtime: &MlxRuntime,
    seed: u64,
    shape: &[i32],
    delta_sigma: f32,
) -> Result<MlxArray, runtime_integration::MlxRuntimeError> {
    let noise = build_keyed_initial_latents(runtime, seed, shape, 1.0)?;
    let unit_velocity = runtime.full(shape, 1.0, MlxDtype::BFloat16)?;
    let updated = flux2_klein_euler_update(runtime, &noise, &unit_velocity, delta_sigma)?;
    runtime.evaluate_arrays(&[&updated])?;
    Ok(updated)
}

/// Direct-MLX scalar seam for proving pinned scheduler accumulation.
pub fn flux2_klein_euler_update_for_tests(
    runtime: &MlxRuntime,
    sample: &MlxArray,
    model_output: &MlxArray,
    delta_sigma: f32,
) -> Result<MlxArray, runtime_integration::MlxRuntimeError> {
    flux2_klein_euler_update(runtime, sample, model_output, delta_sigma)
}

fn flux2_klein_euler_update(
    runtime: &MlxRuntime,
    sample: &MlxArray,
    model_output: &MlxArray,
    delta_sigma: f32,
) -> Result<MlxArray, runtime_integration::MlxRuntimeError> {
    let float32_sample = runtime.astype(sample, MlxDtype::Float32)?;
    let float32_model_output = runtime.astype(model_output, MlxDtype::Float32)?;
    let scaled_model_output = runtime.multiply_scalar(&float32_model_output, delta_sigma)?;
    let float32_updated_sample = runtime.add(&float32_sample, &scaled_model_output)?;
    Ok(runtime.astype(&float32_updated_sample, MlxDtype::BFloat16)?)
}

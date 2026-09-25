//! Native MLX component owner for the Qwen-Image-2.1 engine.
//!
//! One owner, one process-global `MlxRuntime`, and one render session at a time. The render is
//! the family's verified pipeline: each request constructs `QwenImage21Pipeline` fresh — the
//! render releases components in the reference's offload order, so a pipeline is single-render
//! by design and weights re-map per request — then the session advances one boundary per
//! `advance_render` call.
//!
//! Load is deliberately cheap: the artifact is validated and the memory floor advertised, but
//! no weights map until the first request's preparation boundary, so a swapped-in image model
//! does not pin ten gigabytes of wired memory while idle. Request finalization follows the
//! image-lane contract: drop request owners, synchronize the GPU stream, clear reclaimable
//! allocator storage, then sample final memory before the completion publishes.

use std::path::PathBuf;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::qwen_image_21::artifact::{QwenImage21ArtifactProvenance, QwenImage21ArtifactValidator};
use crate::qwen_image_21::official_profile::{
    QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES, QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
    qwen_image_21_image_generation_capabilities, qwen_image_21_official_model_id,
};
use crate::qwen_image_21::pipeline::QwenImage21Pipeline;
use crate::qwen_image_21::render_contract::{QwenImage21RenderAdvance, QwenImage21RenderRequest};
use crate::qwen_image_21::render_session::QwenImage21RenderSession;
use crate::{
    MlxMemoryLimitAdjustment, MlxMemoryTelemetry, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionLog, PerformanceAttributionOutcome,
    PerformanceOperation,
};

use super::components::{
    QwenImage21ComponentLoad, QwenImage21EngineComponents, QwenImage21EngineRequest,
};

/// Bound on the public failure descriptions that can carry component detail.
const MAXIMUM_FAILURE_DESCRIPTION_CHARACTERS: usize = 512;

/// The concrete component owner behind the Qwen-Image-2.1 lifecycle.
pub struct QwenImage21MlxComponents {
    model_directory: PathBuf,
    provenance: QwenImage21ArtifactProvenance,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    minimum_mlx_memory_ceiling_bytes: Option<u64>,
    performance_attribution_enabled: bool,
    performance_attribution_log_path: PathBuf,
    performance_attribution_log: Option<PerformanceAttributionLog>,
    runtime: Option<MlxRuntime>,
    request_id: Option<RequestId>,
    request: Option<RequestControls>,
    session: Option<QwenImage21RenderSession>,
    request_attribution: Option<PerformanceAttribution>,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
}

/// The request facts the finalization report and each boundary need, in the components'
/// own types.
#[derive(Clone)]
struct RequestControls {
    prompt: String,
    width_pixels: u32,
    height_pixels: u32,
    steps: u16,
    seed: u64,
}

impl QwenImage21MlxComponents {
    /// Real-artifact seam for the worker's model-family factory.
    #[must_use]
    pub fn new(
        model_directory: impl Into<PathBuf>,
        provenance: QwenImage21ArtifactProvenance,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        performance_attribution_enabled: bool,
        performance_attribution_log_path: PathBuf,
    ) -> Self {
        Self {
            model_directory: model_directory.into(),
            provenance,
            effective_mlx_memory_ceiling_bytes,
            allocator_cache_memory_limit_bytes,
            minimum_mlx_memory_ceiling_bytes: None,
            performance_attribution_enabled,
            performance_attribution_log_path,
            performance_attribution_log: None,
            runtime: None,
            request_id: None,
            request: None,
            session: None,
            request_attribution: None,
            post_cleanup_memory_telemetry: None,
        }
    }

    fn new_attribution(&self) -> PerformanceAttribution {
        if self.performance_attribution_enabled {
            PerformanceAttribution::enabled()
        } else {
            PerformanceAttribution::disabled()
        }
    }

    fn initialize_runtime(&mut self) -> Result<(), String> {
        if self.runtime.is_some() {
            return Ok(());
        }
        let memory_limits = MlxMemoryLimits::new(
            self.effective_mlx_memory_ceiling_bytes,
            self.allocator_cache_memory_limit_bytes,
        )
        .map_err(|error| error.to_string())?;
        self.runtime =
            Some(MlxRuntime::initialize(memory_limits).map_err(|error| error.to_string())?);
        Ok(())
    }

    fn memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        let snapshot = self.runtime.as_ref()?.memory_snapshot().ok()?;
        Some(MlxMemoryTelemetry::new(
            snapshot.active_memory_bytes() as u64,
            snapshot.allocator_cache_memory_bytes() as u64,
            snapshot.peak_memory_bytes() as u64,
            crate::MlxActiveMemoryBreakdown::default(),
        ))
    }

    /// Runs `operation` under request attribution, with the runtime temporarily handed out so
    /// the closure can borrow it while `&mut self` drives the boundary.
    fn measure_boundary<T>(
        &mut self,
        boundary: PerformanceOperation,
        runtime: &MlxRuntime,
        operation: impl FnOnce(&mut Self, &MlxRuntime) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut attribution = self
            .request_attribution
            .take()
            .unwrap_or_else(PerformanceAttribution::disabled);
        let operation_result =
            attribution.measure_operation(boundary, |_| operation(self, runtime));
        self.request_attribution = Some(attribution);
        operation_result
    }
}

impl QwenImage21EngineComponents for QwenImage21MlxComponents {
    fn load(&mut self) -> Result<QwenImage21ComponentLoad, String> {
        let mut load_attribution = self.new_attribution();
        let mut attribution_log = PerformanceAttributionLog::open(
            &self.performance_attribution_log_path,
            self.performance_attribution_enabled,
        )
        .map_err(|error| error.to_string())?;
        let load_result = QwenImage21ArtifactValidator::new()
            .validate_with_performance_attribution(
                &self.model_directory,
                self.provenance.clone(),
                &mut load_attribution,
            )
            .map_err(|error| error.to_string());
        let validated_artifact = match load_result {
            Ok(validated_artifact) => validated_artifact,
            Err(error) => {
                let bounded_failure = error
                    .chars()
                    .take(MAXIMUM_FAILURE_DESCRIPTION_CHARACTERS)
                    .collect();
                if let Some(report) = load_attribution.finish_model_loading(
                    ModelLoadingPerformanceAttributionMetadata {
                        outcome: PerformanceAttributionOutcome::Failed,
                        model_id: Some(qwen_image_21_official_model_id().to_owned()),
                        model_revision: Some(self.provenance.revision().to_owned()),
                        prefill_transient_observation_completed: false,
                        prefill_observed_transient_high_water_bytes: 0,
                        total_artifact_payload_bytes: None,
                        resident_model_payload_bytes: Some(0),
                        model_shard_count: None,
                        mlx_active_memory_bytes: None,
                        mlx_allocator_cache_memory_bytes: None,
                        mlx_peak_memory_bytes: None,
                        failure_description: Some(bounded_failure),
                    },
                ) {
                    attribution_log
                        .record(&report)
                        .map_err(|error| error.to_string())?;
                }
                self.performance_attribution_log = Some(attribution_log);
                return Err(error);
            }
        };
        // The memory floor is the artifact's own weight bytes plus the measured activation
        // headroom; the worker may raise the live ceiling from here but never below it.
        let minimum_mlx_memory_ceiling_bytes = validated_artifact
            .text_encoder_inventory()
            .payload_bytes()
            .checked_add(validated_artifact.transformer_inventory().payload_bytes())
            .and_then(|payload_bytes| {
                payload_bytes.checked_add(validated_artifact.vae_inventory().payload_bytes())
            })
            .and_then(|payload_bytes| {
                payload_bytes.checked_add(QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES as u64)
            })
            .ok_or_else(|| "Qwen-Image-2.1 artifact payload accounting overflowed".to_owned())?;
        let total_artifact_payload_bytes = minimum_mlx_memory_ceiling_bytes
            .checked_sub(QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES as u64);
        // The reviewed artifact stores each component as one weight file, so the shard count
        // is the component count.
        let model_shard_count = Some(3_usize);
        if usize::try_from(minimum_mlx_memory_ceiling_bytes).unwrap_or(usize::MAX)
            > self.effective_mlx_memory_ceiling_bytes
        {
            // The runtime initializes lazily on the first request; refusing here keeps the load
            // boundary honest about a ceiling that cannot serve the artifact.
            return Err(format!(
                "the configured MLX memory ceiling is below the artifact floor of {minimum_mlx_memory_ceiling_bytes} bytes"
            ));
        }
        if let Some(report) =
            load_attribution.finish_model_loading(ModelLoadingPerformanceAttributionMetadata {
                outcome: PerformanceAttributionOutcome::Success,
                model_id: Some(qwen_image_21_official_model_id().to_owned()),
                model_revision: Some(self.provenance.revision().to_owned()),
                prefill_transient_observation_completed: false,
                prefill_observed_transient_high_water_bytes: 0,
                total_artifact_payload_bytes,
                resident_model_payload_bytes: Some(0),
                model_shard_count,
                mlx_active_memory_bytes: None,
                mlx_allocator_cache_memory_bytes: None,
                mlx_peak_memory_bytes: None,
                failure_description: None,
            })
        {
            attribution_log
                .record(&report)
                .map_err(|error| error.to_string())?;
        }
        self.performance_attribution_log = Some(attribution_log);
        self.minimum_mlx_memory_ceiling_bytes = Some(minimum_mlx_memory_ceiling_bytes);
        Ok(QwenImage21ComponentLoad::new(
            qwen_image_21_official_model_id(),
            self.provenance.revision().to_owned(),
            qwen_image_21_image_generation_capabilities(),
            minimum_mlx_memory_ceiling_bytes,
        ))
    }

    fn start_request(
        &mut self,
        request_id: RequestId,
        request: QwenImage21EngineRequest,
    ) -> Result<(), String> {
        if self.request_id.is_some() {
            return Err("a Qwen-Image-2.1 request is already active".to_owned());
        }
        self.request_attribution = Some(self.new_attribution());
        self.post_cleanup_memory_telemetry = None;
        self.request = Some(RequestControls {
            prompt: request.prompt,
            width_pixels: request.width_pixels,
            height_pixels: request.height_pixels,
            steps: request.steps,
            seed: request.seed,
        });
        self.request_id = Some(request_id);
        Ok(())
    }

    fn advance_render(&mut self) -> Result<QwenImage21RenderAdvance, String> {
        let request = self
            .request
            .clone()
            .ok_or_else(|| "no Qwen-Image-2.1 request is active".to_owned())?;
        if self.session.is_none() {
            // The preparation boundary: map every component's weights and check the request in.
            self.initialize_runtime()?;
            let render_request = QwenImage21RenderRequest {
                prompt: request.prompt,
                width: request.width_pixels as usize,
                height: request.height_pixels as usize,
                num_inference_steps: request.steps as usize,
                seed: request.seed,
            };
            let model_directory = self.model_directory.clone();
            // The runtime is handed out for the boundary and restored even on failure, because
            // the process-global MLX state cannot be re-initialized with different limits.
            let runtime = self
                .runtime
                .take()
                .ok_or_else(|| "the MLX runtime is unavailable".to_owned())?;
            let session_result = self.measure_boundary(
                PerformanceOperation::ImagePipelineConstruction,
                &runtime,
                |_components, runtime| {
                    let pipeline = QwenImage21Pipeline::load(runtime, &model_directory)
                        .map_err(|error| error.to_string())?;
                    pipeline
                        .into_render_session(&render_request)
                        .map_err(|error| error.to_string())
                },
            );
            self.runtime = Some(runtime);
            let session = session_result?;
            self.session = Some(session);
            return Ok(QwenImage21RenderAdvance::Preparing);
        }
        let runtime = self
            .runtime
            .take()
            .ok_or_else(|| "the MLX runtime is unavailable".to_owned())?;
        let advance_result = self.measure_boundary(
            PerformanceOperation::ImageRenderBoundary,
            &runtime,
            |components, runtime| {
                components
                    .session
                    .as_mut()
                    .ok_or_else(|| "the Qwen-Image-2.1 render session is unavailable".to_owned())?
                    .advance(runtime)
                    .map_err(|error| error.to_string())
            },
        );
        self.runtime = Some(runtime);
        advance_result
    }

    fn finalize_request(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        encoded_bytes: Option<u64>,
        failure_description: Option<&str>,
    ) -> Result<(), String> {
        let mut attribution = self
            .request_attribution
            .take()
            .unwrap_or_else(PerformanceAttribution::disabled);
        // Dropping every request owner first ensures cancellation retires no graph that can
        // later publish partial denoising state or decoded pixels.
        let request_controls = self.request.take();
        self.session = None;
        let request_id = self.request_id.take().map(RequestId::value).unwrap_or(0);
        let memory_snapshot = if let Some(runtime) = self.runtime.as_ref() {
            let cleanup_operation = if outcome == PerformanceAttributionOutcome::Cancelled {
                PerformanceOperation::ImageCancellationSynchronization
            } else {
                PerformanceOperation::ImageFinalCleanup
            };
            attribution
                .measure_operation(cleanup_operation, |_| {
                    runtime.synchronize_gpu_stream_and_clear_allocator_cache()
                })
                .map_err(|error| error.to_string())?;
            runtime.memory_snapshot().ok()
        } else {
            None
        };
        self.post_cleanup_memory_telemetry = memory_snapshot.as_ref().map(|snapshot| {
            MlxMemoryTelemetry::new(
                snapshot.active_memory_bytes() as u64,
                snapshot.allocator_cache_memory_bytes() as u64,
                snapshot.peak_memory_bytes() as u64,
                crate::MlxActiveMemoryBreakdown::default(),
            )
        });
        let (width_pixels, height_pixels, steps, seed) = request_controls
            .map(|controls| {
                (
                    controls.width_pixels,
                    controls.height_pixels,
                    controls.steps,
                    controls.seed,
                )
            })
            .unwrap_or((0, 0, 0, 0));
        let report = if attribution.is_enabled() {
            let bounded_failure_description = failure_description
                .map(|description| description.chars().take(512).collect::<String>());
            attribution.finish_image_generation(
                outcome,
                request_id,
                qwen_image_21_official_model_id().to_owned(),
                self.provenance.revision().to_owned(),
                width_pixels,
                height_pixels,
                steps,
                QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
                seed,
                encoded_bytes,
                (None, None, None),
                memory_snapshot.map_or((None, None, None), |snapshot| {
                    (
                        Some(snapshot.active_memory_bytes() as u64),
                        Some(snapshot.allocator_cache_memory_bytes() as u64),
                        Some(snapshot.peak_memory_bytes() as u64),
                    )
                }),
                bounded_failure_description,
            )
        } else {
            None
        };
        if let Some(report) = report
            && let Some(attribution_log) = self.performance_attribution_log.as_mut()
        {
            attribution_log
                .record(&report)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.post_cleanup_memory_telemetry.take()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.memory_telemetry()
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
        let minimum_mlx_memory_ceiling_bytes = self.minimum_mlx_memory_ceiling_bytes.unwrap_or(0);
        if requested_mlx_memory_ceiling_bytes < minimum_mlx_memory_ceiling_bytes {
            return Err(format!(
                "the requested MLX memory ceiling is below the artifact floor of {minimum_mlx_memory_ceiling_bytes} bytes"
            ));
        }
        let memory_limits =
            MlxMemoryLimits::new(requested_ceiling, self.allocator_cache_memory_limit_bytes)
                .map_err(|error| error.to_string())?;
        if let Some(runtime) = self.runtime.as_mut() {
            runtime
                .update_memory_limits(memory_limits)
                .map_err(|error| error.to_string())?;
        }
        // Without a runtime the next request initializes at the requested ceiling.
        self.effective_mlx_memory_ceiling_bytes = requested_ceiling;
        Ok(MlxMemoryLimitAdjustment::new(
            requested_mlx_memory_ceiling_bytes,
            self.allocator_cache_memory_limit_bytes as u64,
            minimum_mlx_memory_ceiling_bytes,
            ExpertMemoryMode::Resident,
            self.collect_mlx_memory_telemetry(),
        ))
    }
}

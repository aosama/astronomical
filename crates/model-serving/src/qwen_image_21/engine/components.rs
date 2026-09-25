//! Architecture-neutral component contract behind the Qwen-Image-2.1 lifecycle owner.
//!
//! The seam keeps the engine's user-journey tests hermetic (a fake components owner drives the
//! full lifecycle without MLX) while production retains one concrete owner. It deliberately has
//! no `Send` bound because MLX component state is runtime-thread-affine.

use astronomical_ipc_protocol::{ImageGenerationCapabilities, RequestId};

use crate::{
    MlxMemoryLimitAdjustment, MlxMemoryTelemetry, PerformanceAttributionOutcome,
    qwen_image_21::render_contract::QwenImage21RenderAdvance,
};

/// Loaded identity supplied by either the native components or a hermetic lifecycle fake.
pub struct QwenImage21ComponentLoad {
    pub(super) model_id: String,
    pub(super) revision: String,
    pub(super) capabilities: ImageGenerationCapabilities,
    pub(super) minimum_mlx_memory_ceiling_bytes: u64,
}

impl QwenImage21ComponentLoad {
    /// Records the identity, advertisement, and memory floor one load established.
    #[must_use]
    pub fn new(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        capabilities: ImageGenerationCapabilities,
        minimum_mlx_memory_ceiling_bytes: u64,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            revision: revision.into(),
            capabilities,
            minimum_mlx_memory_ceiling_bytes,
        }
    }
}

/// One request's controls as the engine validated them, in the components' own types.
// The MLX component owner reads these fields when it builds the render session; without the
// feature the struct only travels through the engine untouched.
#[cfg_attr(not(feature = "direct-mlx"), allow(dead_code))]
pub struct QwenImage21EngineRequest {
    pub prompt: String,
    pub width_pixels: u32,
    pub height_pixels: u32,
    pub steps: u16,
    pub seed: u64,
}

impl QwenImage21EngineRequest {
    /// The request fields the render session consumes.
    #[must_use]
    pub fn new(
        prompt: impl Into<String>,
        width_pixels: u32,
        height_pixels: u32,
        steps: u16,
        seed: u64,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            width_pixels,
            height_pixels,
            steps,
            seed,
        }
    }
}

/// The component seam the Qwen-Image-2.1 lifecycle drives.
pub trait QwenImage21EngineComponents {
    /// Validates the artifact and reports the serving identity, advertisement, and memory floor.
    fn load(&mut self) -> Result<QwenImage21ComponentLoad, String>;

    /// Records one request's controls; the first `advance_render` constructs its pipeline.
    fn start_request(
        &mut self,
        request_id: RequestId,
        request: QwenImage21EngineRequest,
    ) -> Result<(), String>;

    /// Executes exactly one render boundary; `Rendered` carries the completed image.
    fn advance_render(&mut self) -> Result<QwenImage21RenderAdvance, String>;

    /// Closes the request: releases the render session, writes attribution, and captures the
    /// post-cleanup memory observation.
    fn finalize_request(
        &mut self,
        outcome: PerformanceAttributionOutcome,
        encoded_bytes: Option<u64>,
        failure_description: Option<&str>,
    ) -> Result<(), String>;

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        None
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        None
    }

    fn update_mlx_memory_limit(
        &mut self,
        _requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, String> {
        Err("live MLX memory limits are unavailable".to_owned())
    }
}

//! Architecture-neutral component contract behind the Qwen-Image-2.1 lifecycle owner.
//!
//! The seam keeps the engine's user-journey tests scripted (a fake components owner drives the
//! full lifecycle without MLX) while production retains one concrete owner. It deliberately has
//! no `Send` bound because MLX component state is runtime-thread-affine.

use ipc_protocol::RequestId;

use crate::image_engine_shell::{ImageComponentLoad, ImageEngineComponents};
use crate::qwen_image_21::render_contract::QwenImage21RenderAdvance;

/// Loaded identity supplied by either the native components or a scripted lifecycle fake.
pub type QwenImage21ComponentLoad = ImageComponentLoad;

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

/// The render steps the Qwen-Image-2.1 lifecycle drives, beyond the shared
/// [`ImageEngineComponents`].
pub trait QwenImage21EngineComponents: ImageEngineComponents {
    /// Records one request's controls; the first `advance_render` constructs its pipeline.
    fn start_request(
        &mut self,
        request_id: RequestId,
        request: QwenImage21EngineRequest,
    ) -> Result<(), String>;

    /// Executes exactly one render boundary; `Rendered` carries the completed image.
    fn advance_render(&mut self) -> Result<QwenImage21RenderAdvance, String>;
}

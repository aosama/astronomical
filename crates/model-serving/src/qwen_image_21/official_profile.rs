//! The serving profile every Qwen-Image-2.1 engine boundary agrees on.
//!
//! One place states the reviewed envelope — the serving identity, the image bounds, the
//! diffusion controls, and the memory floor — so the capability advertisement, the request
//! validation, and the worker's memory admission can never disagree. The bounds are the
//! verified envelope: dimensions up to the reviewed 1024 output resolution and steps up to the
//! reference's own default of 40, both exercised end to end by the family's acceptance
//! journeys. Widening them is a one-constant change plus a journey that proves the new bound.

use astronomical_ipc_protocol::ImageGenerationCapabilities;

use crate::qwen_image_21::QWEN_IMAGE_21_OFFICIAL_MODEL_ID;

/// Smallest square-free side the pipeline renders without degenerate latent grids
/// (16×16 latent tokens).
pub const QWEN_IMAGE_21_MINIMUM_IMAGE_DIMENSION_PIXELS: u32 = 256;
/// Largest side the reviewed memory envelope covers: 1024 is the artifact's default output
/// resolution and the size the decode-peak journeys measured.
pub const QWEN_IMAGE_21_MAXIMUM_IMAGE_DIMENSION_PIXELS: u32 = 1_024;
/// The VAE's spatial multiple: every generation side must be a multiple of 32.
pub const QWEN_IMAGE_21_IMAGE_DIMENSION_MULTIPLE_PIXELS: u32 =
    crate::qwen_image_21::VAE_SPATIAL_MULTIPLE as u32;
/// Lower step bound: at least one denoising step, matching the pipeline's own requirement.
pub const QWEN_IMAGE_21_MINIMUM_IMAGE_GENERATION_STEPS: u16 = 1;
/// Upper step bound: the reference pipeline's default inference step count, which is also
/// the schedule every served request runs (mirrored as the discovery-time
/// `default_steps` in the config crate because the supervisor cannot reach this crate).
pub const QWEN_IMAGE_21_MAXIMUM_IMAGE_GENERATION_STEPS: u16 = 40;
/// Qwen-Image-2.1 samples without classifier-free guidance; 1.0 is the neutral scale the
/// reference documents for exactly that case, and the supervisor always sends it.
pub const QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS: u32 = 1_000;
/// Activation headroom above the three components' weight bytes. The render releases the
/// encoder and transformer before the VAE decode, so the peak is the decode's staged f32
/// activations; two gigabytes covered it on every measured revision.
pub const QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// The capabilities a loaded Qwen-Image-2.1 engine advertises to the supervisor.
#[must_use]
pub fn qwen_image_21_image_generation_capabilities() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        minimum_width_pixels: QWEN_IMAGE_21_MINIMUM_IMAGE_DIMENSION_PIXELS,
        maximum_width_pixels: QWEN_IMAGE_21_MAXIMUM_IMAGE_DIMENSION_PIXELS,
        minimum_height_pixels: QWEN_IMAGE_21_MINIMUM_IMAGE_DIMENSION_PIXELS,
        maximum_height_pixels: QWEN_IMAGE_21_MAXIMUM_IMAGE_DIMENSION_PIXELS,
        dimension_multiple_pixels: QWEN_IMAGE_21_IMAGE_DIMENSION_MULTIPLE_PIXELS,
        maximum_steps: QWEN_IMAGE_21_MAXIMUM_IMAGE_GENERATION_STEPS,
        maximum_guidance_thousandths: QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
        output_mime_types: vec!["image/png".to_owned()],
    }
}

/// The canonical serving identity for the reviewed artifact.
#[must_use]
pub const fn qwen_image_21_official_model_id() -> &'static str {
    QWEN_IMAGE_21_OFFICIAL_MODEL_ID
}

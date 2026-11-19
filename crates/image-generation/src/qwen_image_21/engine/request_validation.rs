//! Request validation for the official Qwen-Image-2.1 serving envelope.
//!
//! The IPC protocol validates family-neutral bounds (dimensions, step count, guidance range);
//! this module adds the envelope the reviewed profile actually serves: 32-pixel alignment, the
//! verified 256..1024 range, the reference's step default as the ceiling, and the neutral
//! guidance scale — Qwen-Image-2.1 samples without classifier-free guidance, so any other
//! value would be silently ignored and then misreported in completion metadata.

use astronomical_ipc_protocol::{ImageGenerationCommand, ImageGenerationFailureReason};

use crate::qwen_image_21::official_profile::{
    QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS, QWEN_IMAGE_21_IMAGE_DIMENSION_MULTIPLE_PIXELS,
    QWEN_IMAGE_21_MAXIMUM_IMAGE_DIMENSION_PIXELS, QWEN_IMAGE_21_MAXIMUM_IMAGE_GENERATION_STEPS,
    QWEN_IMAGE_21_MINIMUM_IMAGE_DIMENSION_PIXELS, QWEN_IMAGE_21_MINIMUM_IMAGE_GENERATION_STEPS,
    qwen_image_21_official_model_id,
};

/// Rejects requests outside the official envelope before any component state is touched.
pub fn validate_official_request(
    serving_model_id: &str,
    command: &ImageGenerationCommand,
) -> Result<(), ImageGenerationFailureReason> {
    if command.model != serving_model_id {
        return Err(ImageGenerationFailureReason::invalid_request(format!(
            "model must be the loaded Qwen-Image-2.1 identity {}",
            qwen_image_21_official_model_id()
        )));
    }
    validate_dimension("width", command.settings.width_pixels)?;
    validate_dimension("height", command.settings.height_pixels)?;
    if !(QWEN_IMAGE_21_MINIMUM_IMAGE_GENERATION_STEPS
        ..=QWEN_IMAGE_21_MAXIMUM_IMAGE_GENERATION_STEPS)
        .contains(&command.settings.steps)
    {
        return Err(ImageGenerationFailureReason::invalid_request(format!(
            "steps must be within {}..={}",
            QWEN_IMAGE_21_MINIMUM_IMAGE_GENERATION_STEPS,
            QWEN_IMAGE_21_MAXIMUM_IMAGE_GENERATION_STEPS
        )));
    }
    if command.settings.guidance_thousandths != QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS {
        return Err(ImageGenerationFailureReason::invalid_request(
            "Qwen-Image-2.1 samples without classifier-free guidance; guidance must be 1.0"
                .to_owned(),
        ));
    }
    Ok(())
}

fn validate_dimension(
    parameter_name: &str,
    dimension_pixels: u32,
) -> Result<(), ImageGenerationFailureReason> {
    if !(QWEN_IMAGE_21_MINIMUM_IMAGE_DIMENSION_PIXELS
        ..=QWEN_IMAGE_21_MAXIMUM_IMAGE_DIMENSION_PIXELS)
        .contains(&dimension_pixels)
    {
        return Err(ImageGenerationFailureReason::invalid_request(format!(
            "{parameter_name} must be within {}..={} pixels",
            QWEN_IMAGE_21_MINIMUM_IMAGE_DIMENSION_PIXELS,
            QWEN_IMAGE_21_MAXIMUM_IMAGE_DIMENSION_PIXELS
        )));
    }
    if !dimension_pixels.is_multiple_of(QWEN_IMAGE_21_IMAGE_DIMENSION_MULTIPLE_PIXELS) {
        return Err(ImageGenerationFailureReason::invalid_request(format!(
            "{parameter_name} must be a multiple of {QWEN_IMAGE_21_IMAGE_DIMENSION_MULTIPLE_PIXELS} pixels"
        )));
    }
    Ok(())
}

//! Canonical serving-identity and official generation-profile validation.

use astronomical_ipc_protocol::{ImageGenerationCommand, ImageGenerationFailureReason};

use super::super::Flux2KleinOfficialProfile;

pub(super) fn validate_official_request(
    serving_model_id: &str,
    generation_command: &ImageGenerationCommand,
) -> Result<(), ImageGenerationFailureReason> {
    generation_command
        .validate()
        .map_err(|error| ImageGenerationFailureReason::invalid_request(error.to_string()))?;
    if generation_command.model != serving_model_id {
        return Err(ImageGenerationFailureReason::invalid_request(
            "request model does not match the loaded FLUX.2 Klein artifact",
        ));
    }
    if generation_command.prompt.trim().is_empty() {
        return Err(ImageGenerationFailureReason::invalid_request(
            "image prompt must contain text",
        ));
    }
    if usize::from(generation_command.settings.steps)
        != Flux2KleinOfficialProfile::inference_step_count()
    {
        return Err(ImageGenerationFailureReason::invalid_request(
            "FLUX.2 Klein requires exactly four denoising steps",
        ));
    }
    if generation_command.settings.guidance_thousandths
        != Flux2KleinOfficialProfile::guidance_thousandths()
    {
        return Err(ImageGenerationFailureReason::invalid_request(
            "FLUX.2 Klein requires guidance 1.0",
        ));
    }
    Ok(())
}

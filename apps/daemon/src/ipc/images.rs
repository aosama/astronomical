//! Image generation routing for the ephemeral daemon IPC service: resolves
//! the target image model against its capability envelope, dispatches one
//! image command to the worker executor, and answers with a single terminal
//! frame carrying the PNG.
//!
//! This is the CLI's door to image generation, so it never touches the REST
//! surface. It applies the same rules the REST endpoint applies (the model
//! must support text-to-image, the size must sit inside the model's envelope
//! before queue admission, the diffusion step count is the family's own) so a
//! request that would fail late fails here, early and with a named reason.

use astronomical_config::{
    DiscoveredModel, ImageGenerationCapabilities, ModelCapabilities, near_model_matches,
    resolve_model_id,
};
use astronomical_ipc_protocol::{
    DaemonResponse, DaemonTransportError, ImageGenerateRequest, ImageGenerationCommand,
    ImageGenerationFailureReason, ImageGenerationSettings, RequestId, StreamingResponseWriter,
};

use crate::{
    DaemonIpcGenerationContext, ImageGenerationExecutionError, ImageGenerationOutput,
    application::allocate_chat_request_id,
    ipc::{
        models::live_discovered_models,
        service::{
            generation_start_rejection_reason, send_terminal_streaming_response,
            unknown_model_rejection_reason,
        },
    },
    serving::image_seed::generated_seed,
};

/// Classifier-free guidance, in thousandths, applied to every served image.
/// The REST surface pins the same value; it is a family constant, not a knob.
const SERVED_GUIDANCE_THOUSANDTHS: u32 = 1_000;

/// The model a request resolved to and the capability facts it needs.
struct ImageTargetModel {
    resolved_model_id: String,
    image_capabilities: ImageGenerationCapabilities,
}

/// Runs one IPC image request and answers with exactly one terminal frame.
pub(crate) async fn run_image_generation(
    generation_context: &DaemonIpcGenerationContext,
    image_request: ImageGenerateRequest,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let terminal_response =
        resolve_image_terminal_response(generation_context, image_request).await;
    send_terminal_streaming_response(streaming_response_writer, &terminal_response).await
}

/// Produces the single terminal frame: a rejection before admission, or the
/// worker's completed or failed outcome after.
async fn resolve_image_terminal_response(
    generation_context: &DaemonIpcGenerationContext,
    image_request: ImageGenerateRequest,
) -> DaemonResponse {
    let target_model = match resolve_image_target_model(generation_context, &image_request) {
        Ok(target_model) => target_model,
        Err(rejection_reason) => return rejected(rejection_reason),
    };
    let Some(request_id_raw) = allocate_chat_request_id(&generation_context.next_chat_request_id)
    else {
        return rejected("the local request identifier space is exhausted".to_owned());
    };
    let image_command = build_image_command(request_id_raw, &target_model, image_request);
    if let Err(validation_error) = image_command.validate() {
        return rejected(validation_error.to_string());
    }
    let mut image_result_receiver = match generation_context
        .executor
        .start_image_generation(image_command)
        .await
    {
        Ok(image_result_receiver) => image_result_receiver,
        Err(start_error) => return rejected(generation_start_rejection_reason(start_error)),
    };
    match image_result_receiver.recv().await {
        Some(image_outcome) => image_outcome_to_response(image_outcome, target_model),
        None => DaemonResponse::ImageGenerationFailed {
            reason: ImageGenerationFailureReason::FatalExecution {
                reason: "the worker stream ended before the image completed".to_owned(),
            },
        },
    }
}

fn rejected(reason: String) -> DaemonResponse {
    DaemonResponse::GenerationRejected { reason }
}

/// Resolves the requested model and checks, before any queue admission, that
/// it can draw and that the size is one it can render. A doomed request must
/// not win the active-generation permit or trigger a model swap.
fn resolve_image_target_model(
    generation_context: &DaemonIpcGenerationContext,
    image_request: &ImageGenerateRequest,
) -> Result<ImageTargetModel, String> {
    if !generation_context
        .executor
        .worker_health_snapshot()
        .status
        .is_ready()
    {
        return Err("the worker is not ready yet".to_owned());
    }
    let discovered_models = live_discovered_models(generation_context);
    let discovered_model = find_discovered_model(&discovered_models, &image_request.model)?;
    let ModelCapabilities::ImageGeneration(image_capabilities) = &discovered_model.capabilities
    else {
        return Err(not_an_image_model(&discovered_model.model_id));
    };
    if !image_capabilities.supports_text_to_image {
        return Err(not_an_image_model(&discovered_model.model_id));
    }
    if let Some((_parameter_name, violation_message)) = image_capabilities
        .image_dimension_violation(
            &discovered_model.model_id,
            image_request.width_pixels,
            image_request.height_pixels,
        )
    {
        return Err(violation_message);
    }
    Ok(ImageTargetModel {
        resolved_model_id: discovered_model.model_id.clone(),
        image_capabilities: *image_capabilities,
    })
}

fn not_an_image_model(model_id: &str) -> String {
    format!("the model {model_id} does not support text-to-image generation")
}

/// Matches the requested id (full or leaf, any case the alias table accepts)
/// to one discovered model, or names the near matches.
fn find_discovered_model<'models>(
    discovered_models: &'models [DiscoveredModel],
    requested_model_id: &str,
) -> Result<&'models DiscoveredModel, String> {
    let mut known_model_ids = Vec::with_capacity(discovered_models.len());
    for discovered_model in discovered_models {
        known_model_ids.push(discovered_model.model_id.as_str());
    }
    let resolved_model_id = resolve_model_id(requested_model_id, &known_model_ids);
    for discovered_model in discovered_models {
        if discovered_model.model_id == resolved_model_id {
            return Ok(discovered_model);
        }
    }
    Err(unknown_model_rejection_reason(
        requested_model_id,
        &near_model_matches(requested_model_id, &known_model_ids),
        "run `astronomical models list` to see what is installed",
    ))
}

/// The family's own step count and guidance apply; the caller chose only the
/// prompt, the size, and optionally the seed.
fn build_image_command(
    request_id_raw: u64,
    target_model: &ImageTargetModel,
    image_request: ImageGenerateRequest,
) -> ImageGenerationCommand {
    let effective_seed = match image_request.seed {
        Some(requested_seed) => requested_seed,
        None => generated_seed(request_id_raw),
    };
    ImageGenerationCommand {
        request_id: RequestId::new(request_id_raw),
        model: target_model.resolved_model_id.clone(),
        prompt: image_request.prompt,
        settings: ImageGenerationSettings {
            width_pixels: image_request.width_pixels,
            height_pixels: image_request.height_pixels,
            steps: target_model.image_capabilities.default_steps,
            guidance_thousandths: SERVED_GUIDANCE_THOUSANDTHS,
            seed: effective_seed,
        },
    }
}

fn image_outcome_to_response(
    image_outcome: Result<ImageGenerationOutput, ImageGenerationExecutionError>,
    target_model: ImageTargetModel,
) -> DaemonResponse {
    match image_outcome {
        Ok(image_output) => DaemonResponse::ImageGenerationCompleted {
            model: target_model.resolved_model_id,
            image: image_output.generated_image,
            metadata: image_output.result_metadata,
        },
        Err(ImageGenerationExecutionError::WorkerFailure(failure_reason)) => {
            DaemonResponse::ImageGenerationFailed {
                reason: failure_reason,
            }
        }
        Err(ImageGenerationExecutionError::DeadlineExceeded) => {
            DaemonResponse::ImageGenerationFailed {
                reason: ImageGenerationFailureReason::FatalExecution {
                    reason: "the image did not finish before the daemon's deadline".to_owned(),
                },
            }
        }
        Err(ImageGenerationExecutionError::WorkerUnavailable) => {
            rejected("the worker is unavailable".to_owned())
        }
    }
}

#[cfg(test)]
#[path = "images_tests.rs"]
mod images_tests;

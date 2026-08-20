//! OpenAI-compatible image generation with complete validation before shared queue admission.

use std::time::{SystemTime, UNIX_EPOCH};

use astronomical_config::ModelCapabilities;
use astronomical_ipc_protocol::{
    ImageGenerationCommand, ImageGenerationFailureReason, ImageGenerationSettings, ProtocolError,
    RequestId, WorkerCommand, encode_command,
};
use astronomical_rest_contract::{
    OpenAiErrorResponse, OpenAiGeneratedImageParts, OpenAiImageGenerationRequest,
    OpenAiImageGenerationResponse, OpenAiImageGenerationValidationError,
};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};

use crate::{
    GenerationStartError, ImageGenerationExecutionError,
    application::{ApplicationState, allocate_chat_request_id},
};

pub(crate) async fn create_image_generation(
    State(application_state): State<ApplicationState>,
    request_body_bytes: Bytes,
) -> Response {
    let image_request =
        match serde_json::from_slice::<OpenAiImageGenerationRequest>(request_body_bytes.as_ref()) {
            Ok(image_request) => image_request,
            Err(json_error) => {
                return invalid_request_response(
                    format!("request body is not valid JSON: {json_error}"),
                    None,
                    "invalid_json",
                );
            }
        };
    let request_parts = match image_request.into_parts() {
        Ok(request_parts) => request_parts,
        Err(validation_error) => {
            return invalid_request_response(
                validation_error.to_string(),
                Some(image_validation_parameter(&validation_error)),
                "invalid_request",
            );
        }
    };

    // This lock keeps validation, immutable queue admission, and configuration replacement on one
    // side of the same generation, while releasing before a queued request waits for execution.
    let configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    if !worker_health_snapshot.status.is_ready() {
        return worker_unavailable_response();
    }
    let discovered_models = application_state.discovered_models_snapshot();
    let known_model_ids = discovered_models
        .iter()
        .map(|discovered_model| discovered_model.model_id.as_str())
        .collect::<Vec<_>>();
    let resolved_model_id =
        astronomical_config::resolve_model_id(&request_parts.model, &known_model_ids);
    let Some(discovered_model) = discovered_models
        .iter()
        .find(|discovered_model| discovered_model.model_id == resolved_model_id)
    else {
        return invalid_request_response(
            "model is not available to the local worker",
            Some("model"),
            "model_not_found",
        );
    };
    if !matches!(
        &discovered_model.capabilities,
        ModelCapabilities::ImageGeneration(capabilities) if capabilities.supports_text_to_image
    ) {
        return invalid_request_response(
            "the requested model does not support text-to-image generation",
            Some("model"),
            "model_capability_mismatch",
        );
    }

    let request_id = match allocate_chat_request_id(&application_state.next_chat_request_id) {
        Some(request_id) => request_id,
        None => return request_identifier_exhausted_response(),
    };
    let effective_seed = request_parts
        .seed
        .unwrap_or_else(|| generated_seed(request_id));
    let image_generation_command = ImageGenerationCommand {
        request_id: RequestId::new(request_id),
        model: resolved_model_id.to_owned(),
        prompt: request_parts.prompt,
        settings: ImageGenerationSettings {
            width_pixels: request_parts.width,
            height_pixels: request_parts.height,
            // REST validation has already established the exact supported values.
            steps: u16::try_from(request_parts.steps).unwrap_or(4),
            guidance_thousandths: 1_000,
            seed: effective_seed,
        },
    };
    if let Err(transport_error) = encode_command(&WorkerCommand::GenerateImage(
        image_generation_command.clone(),
    )) {
        return transport_validation_error_response(transport_error);
    }

    let model_revision = discovered_model.revision.clone();
    let (admission_sender, mut admission_receiver) = tokio::sync::oneshot::channel();
    let mut generation_start_future = application_state
        .generation_executor
        .start_image_generation_with_admission_signal(image_generation_command, admission_sender);
    let generation_start_result = tokio::select! {
        admission_result = &mut admission_receiver => {
            drop(configuration_transition_guard);
            match admission_result {
                Ok(()) => generation_start_future.await,
                Err(_) => Err(GenerationStartError::WorkerUnavailable),
            }
        }
        generation_start_result = &mut generation_start_future => {
            drop(configuration_transition_guard);
            generation_start_result
        }
    };
    let mut image_result_receiver = match generation_start_result {
        Ok(image_result_receiver) => image_result_receiver,
        Err(start_error) => return generation_start_error_response(start_error),
    };
    let image_output = match image_result_receiver.recv().await {
        Some(Ok(image_output)) => image_output,
        Some(Err(image_error)) => return image_execution_error_response(request_id, image_error),
        None => return worker_unavailable_response(),
    };
    let created_at_unix_seconds = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration_since_epoch) => duration_since_epoch.as_secs(),
        Err(system_time_error) => {
            tracing::error!(error = %system_time_error, "system clock predates the Unix epoch");
            return worker_unavailable_response();
        }
    };
    Json(OpenAiImageGenerationResponse::new(
        created_at_unix_seconds,
        OpenAiGeneratedImageParts {
            b64_json: STANDARD.encode(image_output.generated_image.encoded_bytes),
            mime_type: image_output.generated_image.mime_type,
            model_revision,
            effective_seed: image_output.result_metadata.seed,
            width: image_output.result_metadata.width_pixels,
            height: image_output.result_metadata.height_pixels,
        },
    ))
    .into_response()
}

fn image_validation_parameter(error: &OpenAiImageGenerationValidationError) -> &'static str {
    match error {
        OpenAiImageGenerationValidationError::UnknownField { .. } => "request",
        OpenAiImageGenerationValidationError::BlankModel => "model",
        OpenAiImageGenerationValidationError::BlankPrompt => "prompt",
        OpenAiImageGenerationValidationError::UnsupportedDimension { parameter_name, .. } => {
            parameter_name
        }
        OpenAiImageGenerationValidationError::UnsupportedStepCount { .. } => "steps",
        OpenAiImageGenerationValidationError::UnsupportedGuidance { .. } => "guidance",
        OpenAiImageGenerationValidationError::UnsupportedResponseFormat { .. } => "response_format",
        OpenAiImageGenerationValidationError::UnsupportedImageCount { .. } => "n",
    }
}

fn generation_start_error_response(start_error: GenerationStartError) -> Response {
    match start_error {
        GenerationStartError::CapacityUnavailable => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(OpenAiErrorResponse::capacity_unavailable(
                "the generation queue is full",
            )),
        )
            .into_response(),
        GenerationStartError::ModelLoadFailed {
            model_load_failure_reason,
        } => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(OpenAiErrorResponse::model_load_failed(
                model_load_failure_reason,
            )),
        )
            .into_response(),
        GenerationStartError::RequestTooLarge { .. } => request_too_large_response(),
        GenerationStartError::WorkerUnavailable => worker_unavailable_response(),
    }
}

fn image_execution_error_response(
    request_id: u64,
    image_error: ImageGenerationExecutionError,
) -> Response {
    match image_error {
        ImageGenerationExecutionError::WorkerFailure(
            ImageGenerationFailureReason::InvalidRequest { reason },
        ) => {
            tracing::warn!(request_id, reason = %reason, "image worker rejected a validated request");
            invalid_request_response(
                "the image request was rejected by the local worker",
                None,
                "invalid_request",
            )
        }
        ImageGenerationExecutionError::WorkerFailure(
            ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration,
        ) => invalid_request_response(
            "the requested model does not support image generation",
            Some("model"),
            "model_capability_mismatch",
        ),
        ImageGenerationExecutionError::WorkerFailure(ImageGenerationFailureReason::EngineBusy) => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(OpenAiErrorResponse::capacity_unavailable(
                "the image engine is busy",
            )),
        )
            .into_response(),
        ImageGenerationExecutionError::WorkerFailure(
            ImageGenerationFailureReason::EncodingFailed { reason },
        ) => {
            tracing::error!(request_id, reason = %reason, "image worker could not encode its output");
            image_worker_failure_response()
        }
        ImageGenerationExecutionError::WorkerFailure(
            ImageGenerationFailureReason::FatalExecution { reason },
        ) => {
            tracing::error!(request_id, reason = %reason, "image worker execution failed");
            image_worker_failure_response()
        }
        ImageGenerationExecutionError::WorkerFailure(ImageGenerationFailureReason::Cancelled)
        | ImageGenerationExecutionError::WorkerUnavailable => worker_unavailable_response(),
        ImageGenerationExecutionError::DeadlineExceeded => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(OpenAiErrorResponse::service_unavailable(
                "image generation exceeded its bounded execution deadline",
                Some("image_generation_timeout"),
            )),
        )
            .into_response(),
    }
}

fn image_worker_failure_response() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(OpenAiErrorResponse::service_unavailable(
            "image generation failed in the local worker",
            Some("image_generation_failed"),
        )),
    )
        .into_response()
}

fn transport_validation_error_response(transport_error: ProtocolError) -> Response {
    match transport_error {
        ProtocolError::OutgoingMessageTooLarge { .. } => request_too_large_response(),
        serialization_error => {
            tracing::error!(error = %serialization_error, "could not validate image IPC transport");
            worker_unavailable_response()
        }
    }
}

fn request_too_large_response() -> Response {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(OpenAiErrorResponse::invalid_request(
            "the image request exceeds the local worker transport limit",
            None,
            Some("request_too_large"),
        )),
    )
        .into_response()
}

fn invalid_request_response(
    message: impl Into<String>,
    parameter: Option<&str>,
    code: &str,
) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(OpenAiErrorResponse::invalid_request(
            message,
            parameter,
            Some(code),
        )),
    )
        .into_response()
}

fn worker_unavailable_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(OpenAiErrorResponse::service_unavailable(
            "the local worker is unavailable",
            Some("worker_unavailable"),
        )),
    )
        .into_response()
}

fn request_identifier_exhausted_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(OpenAiErrorResponse::service_unavailable(
            "the local request identifier space is exhausted",
            Some("request_id_exhausted"),
        )),
    )
        .into_response()
}

fn generated_seed(request_id: u64) -> u64 {
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration_since_epoch| {
            duration_since_epoch.as_secs().rotate_left(32)
                ^ u64::from(duration_since_epoch.subsec_nanos())
        })
        .unwrap_or_default();
    time_seed ^ request_id.rotate_left(17)
}

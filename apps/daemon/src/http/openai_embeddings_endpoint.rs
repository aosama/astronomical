//! OpenAI-compatible native embeddings with complete validation before shared queue admission.

use astronomical_config::ModelCapabilities;
use astronomical_ipc_protocol::{
    EmbeddingEncodingFormat, EmbeddingsCommand, EmbeddingsFailureReason, RequestId,
};
use astronomical_rest_contract::{
    OpenAiEmbedding, OpenAiEmbeddingEncodingFormat as RestEncodingFormat, OpenAiEmbeddingVector,
    OpenAiEmbeddingsRequest, OpenAiEmbeddingsResponse, OpenAiEmbeddingsValidationError,
};
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};

use crate::{
    EmbeddingsExecutionError, EmbeddingsOutput, GenerationStartError,
    application::{ApplicationState, allocate_chat_request_id},
};

pub(crate) async fn create_embeddings(
    State(application_state): State<ApplicationState>,
    request_body_bytes: axum::body::Bytes,
) -> Response {
    let embeddings_request =
        match serde_json::from_slice::<OpenAiEmbeddingsRequest>(request_body_bytes.as_ref()) {
            Ok(embeddings_request) => embeddings_request,
            Err(json_error) => {
                return invalid_request_response(
                    format!("request body is not valid JSON: {json_error}"),
                    None,
                    "invalid_json",
                );
            }
        };
    let request_parts = match embeddings_request.into_parts() {
        Ok(request_parts) => request_parts,
        Err(validation_error) => {
            return invalid_request_response(
                validation_error.to_string(),
                Some(embeddings_validation_parameter(&validation_error)),
                "invalid_request",
            );
        }
    };

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
        ModelCapabilities::Embeddings(_)
    ) {
        return invalid_request_response(
            "the requested model does not support embeddings",
            Some("model"),
            "model_capability_mismatch",
        );
    }

    let request_identifier = match allocate_chat_request_id(&application_state.next_chat_request_id)
    {
        Some(request_identifier) => request_identifier,
        None => return request_identifier_exhausted_response(),
    };
    let embeddings_command = EmbeddingsCommand {
        request_id: RequestId::new(request_identifier),
        model: resolved_model_id.to_owned(),
        inputs: request_parts.inputs,
        encoding_format: match request_parts.encoding_format {
            RestEncodingFormat::Float => EmbeddingEncodingFormat::Float,
            RestEncodingFormat::Base64 => EmbeddingEncodingFormat::Base64,
        },
        dimensions: request_parts.dimensions,
    };
    let (admission_sender, mut admission_receiver) = tokio::sync::oneshot::channel();
    let mut embeddings_start_future = application_state
        .generation_executor
        .start_embeddings_generation_with_admission_signal(embeddings_command, admission_sender);
    let embeddings_start_result = tokio::select! {
        admission_result = &mut admission_receiver => {
            drop(configuration_transition_guard);
            match admission_result {
                Ok(()) => embeddings_start_future.await,
                Err(_) => Err(GenerationStartError::WorkerUnavailable),
            }
        }
        embeddings_start_result = &mut embeddings_start_future => {
            drop(configuration_transition_guard);
            embeddings_start_result
        }
    };
    let mut embeddings_result_receiver = match embeddings_start_result {
        Ok(embeddings_result_receiver) => embeddings_result_receiver,
        Err(start_error) => return generation_start_error_response(start_error),
    };
    let embeddings_output: EmbeddingsOutput = match embeddings_result_receiver.recv().await {
        Some(Ok(embeddings_output)) => embeddings_output,
        Some(Err(embeddings_error)) => {
            return embeddings_execution_error_response(request_identifier, embeddings_error);
        }
        None => return worker_unavailable_response(),
    };
    let aggregate_input_token_count = embeddings_output
        .input_token_counts
        .iter()
        .fold(0u32, |total, count| total.saturating_add(*count));
    let Some(usage) =
        astronomical_rest_contract::OpenAiTokenUsage::new(aggregate_input_token_count, 0)
    else {
        return worker_unavailable_response();
    };
    let embedding_rows = embeddings_output
        .embeddings
        .iter()
        .enumerate()
        .map(|(row_index, components)| {
            OpenAiEmbedding::new(
                u32::try_from(row_index).unwrap_or(u32::MAX),
                match request_parts.encoding_format {
                    RestEncodingFormat::Float => OpenAiEmbeddingVector::Float(components.clone()),
                    RestEncodingFormat::Base64 => OpenAiEmbeddingVector::Base64(
                        STANDARD.encode(float32_little_endian_bytes(components)),
                    ),
                },
            )
        })
        .collect();
    Json(OpenAiEmbeddingsResponse::new(
        embedding_rows,
        resolved_model_id.to_owned(),
        usage,
    ))
    .into_response()
}

fn float32_little_endian_bytes(components: &[f32]) -> Vec<u8> {
    components
        .iter()
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

fn embeddings_validation_parameter(error: &OpenAiEmbeddingsValidationError) -> &'static str {
    match error {
        OpenAiEmbeddingsValidationError::UnknownField { .. } => "request",
        OpenAiEmbeddingsValidationError::EmptyModel => "model",
        OpenAiEmbeddingsValidationError::EmptyInput
        | OpenAiEmbeddingsValidationError::InputCountExceeded { .. }
        | OpenAiEmbeddingsValidationError::InputTextTooLarge { .. }
        | OpenAiEmbeddingsValidationError::TotalInputBytesExceeded { .. } => "input",
        OpenAiEmbeddingsValidationError::UnsupportedEncodingFormat { .. } => "encoding_format",
        OpenAiEmbeddingsValidationError::InvalidDimensions => "dimensions",
    }
}

fn invalid_request_response(
    message: impl Into<String>,
    parameter: Option<&'static str>,
    code: &'static str,
) -> Response {
    (
        axum::http::StatusCode::BAD_REQUEST,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::invalid_request(
                message,
                parameter,
                Some(code),
            ),
        ),
    )
        .into_response()
}

fn worker_unavailable_response() -> Response {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::service_unavailable(
                "the local worker is unavailable",
                Some("worker_unavailable"),
            ),
        ),
    )
        .into_response()
}

fn request_identifier_exhausted_response() -> Response {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::service_unavailable(
                "the local request identifier space is exhausted",
                Some("request_id_exhausted"),
            ),
        ),
    )
        .into_response()
}

fn generation_start_error_response(start_error: GenerationStartError) -> Response {
    match start_error {
        GenerationStartError::CapacityUnavailable => (
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            Json(
                astronomical_rest_contract::OpenAiErrorResponse::capacity_unavailable(
                    "the generation queue is full",
                ),
            ),
        )
            .into_response(),
        GenerationStartError::ModelLoadFailed {
            model_load_failure_reason,
        } => (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(
                astronomical_rest_contract::OpenAiErrorResponse::model_load_failed(
                    model_load_failure_reason,
                ),
            ),
        )
            .into_response(),
        GenerationStartError::RequestTooLarge { .. } => request_too_large_response(),
        GenerationStartError::WorkerUnavailable => worker_unavailable_response(),
    }
}

fn request_too_large_response() -> Response {
    (
        axum::http::StatusCode::PAYLOAD_TOO_LARGE,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::invalid_request(
                "the request exceeded the local IPC transport limit",
                None,
                Some("request_too_large"),
            ),
        ),
    )
        .into_response()
}

fn embeddings_execution_error_response(
    request_identifier: u64,
    embeddings_error: EmbeddingsExecutionError,
) -> Response {
    match embeddings_error {
        EmbeddingsExecutionError::WorkerFailure(EmbeddingsFailureReason::InvalidRequest {
            reason,
        }) => {
            tracing::warn!(
                request_identifier,
                reason = %reason,
                "embeddings worker rejected a validated request"
            );
            invalid_request_response(
                "the embeddings request was rejected by the local worker",
                None,
                "invalid_request",
            )
        }
        EmbeddingsExecutionError::WorkerFailure(
            EmbeddingsFailureReason::ContextLengthExceeded {
                actual_total_context_tokens,
                maximum_context_tokens,
            },
        ) => invalid_request_response(
            format!(
                "embedding input has {actual_total_context_tokens} tokens, exceeding the {maximum_context_tokens}-token encoder context"
            ),
            Some("input"),
            "context_length_exceeded",
        ),
        EmbeddingsExecutionError::WorkerFailure(EmbeddingsFailureReason::EngineBusy) => (
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            Json(
                astronomical_rest_contract::OpenAiErrorResponse::capacity_unavailable(
                    "the embedding engine is busy",
                ),
            ),
        )
            .into_response(),
        EmbeddingsExecutionError::WorkerFailure(EmbeddingsFailureReason::FatalExecution {
            reason,
        }) => {
            tracing::error!(
                request_identifier,
                reason = %reason,
                "embeddings worker execution failed"
            );
            embeddings_worker_failure_response()
        }
        EmbeddingsExecutionError::WorkerFailure(EmbeddingsFailureReason::MalformedModelOutput) => {
            tracing::error!(
                request_identifier,
                "embeddings worker produced malformed output"
            );
            embeddings_worker_failure_response()
        }
        EmbeddingsExecutionError::WorkerUnavailable => worker_unavailable_response(),
    }
}

fn embeddings_worker_failure_response() -> Response {
    (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::service_unavailable(
                "embedding generation failed in the local worker",
                Some("embeddings_failed"),
            ),
        ),
    )
        .into_response()
}

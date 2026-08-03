use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use astronomical_ipc_protocol::RequestId;
use astronomical_rest_contract::{
    OpenAiErrorResponse, OpenAiResponseRequestConfiguration, OpenAiResponseStreamEvent,
    OpenAiResponsesRequest,
};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Event, sse::Sse},
};
use futures_util::stream;

use crate::application::{ApplicationState, allocate_chat_request_id};
use crate::openai_responses_translation::translate_openai_responses_request_parts;
use crate::{
    ChatGenerationStreamEvent, GenerationStartError, OpenAiResponsesCollector,
    OpenAiResponsesStreamEncoder,
};

pub(crate) async fn create_response(
    State(application_state): State<ApplicationState>,
    request_body_bytes: Bytes,
) -> Response {
    let responses_request =
        match serde_json::from_slice::<OpenAiResponsesRequest>(request_body_bytes.as_ref()) {
            Ok(responses_request) => responses_request,
            Err(json_error) => {
                return invalid_request_response(
                    format!("request body is not valid JSON: {json_error}"),
                    None,
                    "invalid_json",
                );
            }
        };
    let request_parts = match responses_request.into_parts() {
        Ok(request_parts) => request_parts,
        Err(validation_error) => {
            return invalid_request_response(validation_error.to_string(), None, "invalid_request");
        }
    };
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    if !worker_health_snapshot.status.is_ready() {
        return worker_unavailable_response();
    }
    let requested_model = &request_parts.model;
    let Some(resolved_model_id) = application_state.resolve_available_generation_model_id(
        requested_model,
        worker_health_snapshot.ready_model_id.as_deref(),
    ) else {
        return invalid_request_response(
            "model is not loaded by the local worker",
            Some("model"),
            "model_not_found",
        );
    };
    let should_stream_response = request_parts.stream;
    let response_instructions = request_parts.instructions.clone();
    let response_request_configuration = request_parts.response_configuration();
    let request_id = match allocate_chat_request_id(&application_state.next_chat_request_id) {
        Some(request_id) => request_id,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(OpenAiErrorResponse::service_unavailable(
                    "the local request identifier space is exhausted",
                    Some("request_id_exhausted"),
                )),
            )
                .into_response();
        }
    };
    let mut chat_generation_command =
        match translate_openai_responses_request_parts(RequestId::new(request_id), request_parts) {
            Ok(chat_generation_command) => chat_generation_command,
            Err(translation_error) => {
                return invalid_request_response(
                    translation_error.to_string(),
                    None,
                    "invalid_request",
                );
            }
        };
    chat_generation_command.model = resolved_model_id;
    chat_generation_command.settings.max_output_tokens =
        crate::generation_output_ceiling::cap_generation_output_tokens(
            application_state.reloadable_config.as_ref(),
            chat_generation_command.settings.max_output_tokens,
        );
    tracing::info!(
        request_id,
        model = %chat_generation_command.model,
        message_count = chat_generation_command.messages.len(),
        tool_count = chat_generation_command.tools.len(),
        stream = should_stream_response,
        request_body_bytes = request_body_bytes.len(),
        "accepted REST Responses request"
    );
    let model_id = chat_generation_command.model.clone();
    let stream_event_receiver = match application_state
        .generation_executor
        .start_chat_generation(chat_generation_command)
        .await
    {
        Ok(stream_event_receiver) => stream_event_receiver,
        Err(GenerationStartError::CapacityUnavailable) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(OpenAiErrorResponse::capacity_unavailable(
                    "the generation queue is full",
                )),
            )
                .into_response();
        }
        Err(GenerationStartError::ModelLoadFailed {
            model_load_failure_reason,
        }) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(OpenAiErrorResponse::model_load_failed(
                    model_load_failure_reason,
                )),
            )
                .into_response();
        }
        Err(GenerationStartError::WorkerUnavailable) => return worker_unavailable_response(),
    };
    let created_at_unix_seconds = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration_since_epoch) => duration_since_epoch.as_secs(),
        Err(system_time_error) => {
            tracing::error!(error = %system_time_error, "system clock predates the Unix epoch");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAiErrorResponse::service_unavailable(
                    "the local server could not timestamp the response",
                    Some("response_timestamp_failed"),
                )),
            )
                .into_response();
        }
    };
    let response_id = format!(
        "resp_{}-{request_id}",
        application_state.completion_id_namespace
    );
    if !should_stream_response {
        return create_non_streaming_response(
            stream_event_receiver,
            response_id,
            created_at_unix_seconds,
            model_id,
            response_instructions,
            response_request_configuration,
        )
        .await;
    }
    create_streaming_response(
        stream_event_receiver,
        OpenAiResponsesStreamEncoder::new(
            response_id,
            created_at_unix_seconds,
            model_id,
            response_instructions,
            response_request_configuration,
        ),
    )
}

async fn create_non_streaming_response(
    mut stream_event_receiver: tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
    response_id: String,
    created_at_unix_seconds: u64,
    model_id: String,
    instructions: Option<String>,
    request_configuration: OpenAiResponseRequestConfiguration,
) -> Response {
    let mut response_collector = OpenAiResponsesCollector::new(
        response_id,
        created_at_unix_seconds,
        model_id,
        instructions,
        request_configuration,
    );
    while let Some(stream_event) = stream_event_receiver.recv().await {
        if let ChatGenerationStreamEvent::Failed {
            reason:
                astronomical_ipc_protocol::ChatGenerationFailureReason::ContextLengthExceeded {
                    actual_total_context_tokens,
                    maximum_context_tokens,
                },
        } = &stream_event
        {
            return invalid_request_response(
                format!(
                    "requested context uses {actual_total_context_tokens} tokens, exceeding the {maximum_context_tokens}-token model context window"
                ),
                Some("input"),
                "context_length_exceeded",
            );
        }
        if let ChatGenerationStreamEvent::Completed {
            prompt_token_count,
            generated_token_count,
            reasoning_token_count,
            cached_token_count,
            reason,
        } = stream_event
        {
            return match response_collector.into_response(
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                reasoning_token_count,
                reason,
            ) {
                Ok(response) => Json(response).into_response(),
                Err(assembly_error) => assembly_failure_response(assembly_error.to_string()),
            };
        }
        if let Err(assembly_error) = response_collector.ingest_event(stream_event) {
            return assembly_failure_response(assembly_error.to_string());
        }
    }
    worker_unavailable_response()
}

fn create_streaming_response(
    stream_event_receiver: tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
    mut stream_encoder: OpenAiResponsesStreamEncoder,
) -> Response {
    let initial_events = stream_encoder.initial_events();
    let response_event_stream = stream::unfold(
        (stream_event_receiver, stream_encoder, initial_events),
        |(mut stream_event_receiver, mut stream_encoder, mut pending_events)| async move {
            loop {
                if let Some(response_stream_event) = pending_events.pop_front() {
                    let serialized_event = serialize_stream_event(response_stream_event);
                    return Some((
                        serialized_event,
                        (stream_event_receiver, stream_encoder, pending_events),
                    ));
                }
                if stream_encoder.is_terminal() {
                    return None;
                }
                let stream_event = match stream_event_receiver.recv().await {
                    Some(stream_event) => stream_event,
                    None => ChatGenerationStreamEvent::Error(
                        crate::ChatGenerationStreamErrorCode::WorkerUnavailable,
                    ),
                };
                match stream_encoder.encode(stream_event) {
                    Ok(encoded_events) => pending_events.extend(encoded_events),
                    Err(encoding_error) => {
                        return Some((
                            Err(axum::Error::new(encoding_error)),
                            (stream_event_receiver, stream_encoder, VecDeque::new()),
                        ));
                    }
                }
            }
        },
    );
    Sse::new(response_event_stream).into_response()
}

fn serialize_stream_event(
    response_stream_event: OpenAiResponseStreamEvent,
) -> Result<Event, axum::Error> {
    let event_type = response_stream_event.event_type();
    let serialized_payload =
        serde_json::to_string(&response_stream_event).map_err(axum::Error::new)?;
    Ok(Event::default().event(event_type).data(serialized_payload))
}

fn invalid_request_response(
    message: impl Into<String>,
    parameter: Option<&str>,
    code: &'static str,
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

fn assembly_failure_response(message: String) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(OpenAiErrorResponse::service_unavailable(
            message,
            Some("response_generation_failed"),
        )),
    )
        .into_response()
}

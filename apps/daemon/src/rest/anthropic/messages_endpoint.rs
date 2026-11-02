//! `POST /v1/messages`: the Anthropic-compatible Messages endpoint. It shares
//! model resolution, queue admission, and the worker generation path with
//! the OpenAI surfaces and differs only in the wire protocol at both ends.

use std::collections::VecDeque;

use astronomical_config::ModelCapabilities;
use astronomical_ipc_protocol::{ChatGenerationCommand, RequestId};
use astronomical_rest_contract::{
    AnthropicError, AnthropicMessageStreamEvent, AnthropicMessagesRequest,
    AnthropicMessagesRequestParts, AnthropicMessagesValidationError,
};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Event, sse::Sse},
};
use futures_util::stream;
use tokio::sync::mpsc::Receiver;

use super::messages_assembly::{AnthropicMessageCollector, WorkerTokenCounts, failure_error};
use super::messages_stream::AnthropicMessagesStreamEncoder;
use super::messages_translation::translate_anthropic_messages_request_parts;
use crate::application;
use crate::application::ApplicationState;
use crate::config::request_generation_defaults::{
    RequestGenerationSettingsPresence, apply_generation_defaults,
};
use crate::{ChatGenerationStreamErrorCode, ChatGenerationStreamEvent, GenerationStartError};

pub(crate) async fn create_message(
    State(application_state): State<ApplicationState>,
    request_body_bytes: Bytes,
) -> Response {
    let request_parts = match parse_messages_request(request_body_bytes.as_ref()) {
        Ok(request_parts) => request_parts,
        Err(rejection_response) => return rejection_response,
    };
    let should_stream_response = request_parts.stream;
    let admitted_generation =
        match admit_message_generation(&application_state, request_parts, request_body_bytes.len())
            .await
        {
            Ok(admitted_generation) => admitted_generation,
            Err(rejection_response) => return rejection_response,
        };
    let AdmittedMessageGeneration {
        message_id,
        model_id,
        stream_event_receiver,
    } = admitted_generation;
    if should_stream_response {
        return create_streaming_response(
            stream_event_receiver,
            AnthropicMessagesStreamEncoder::new(message_id, model_id),
        );
    }
    create_non_streaming_response(
        stream_event_receiver,
        AnthropicMessageCollector::new(message_id, model_id),
    )
    .await
}

/// A request the queue admitted: the identity its answer is stamped with and
/// the worker event stream that answer is built from.
struct AdmittedMessageGeneration {
    message_id: String,
    model_id: String,
    stream_event_receiver: Receiver<ChatGenerationStreamEvent>,
}

/// Parses and validates the body, rejecting before any daemon state is touched.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn parse_messages_request(
    request_body_bytes: &[u8],
) -> Result<AnthropicMessagesRequestParts, Response> {
    let messages_request = serde_json::from_slice::<AnthropicMessagesRequest>(request_body_bytes)
        .map_err(json_rejection_response)?;
    messages_request
        .into_parts()
        .map_err(validation_rejection_response)
}

/// Resolves the model, translates the request, and starts generation, holding
/// the configuration-transition lock until the queue admits the request so
/// resolution and admission observe one side of a worker replacement.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
async fn admit_message_generation(
    application_state: &ApplicationState,
    request_parts: AnthropicMessagesRequestParts,
    request_body_bytes: usize,
) -> Result<AdmittedMessageGeneration, Response> {
    let configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    let resolved_model_id = resolve_messages_target_model(application_state, &request_parts.model)?;
    let request_id = allocate_messages_request_identifier(application_state)?;
    let should_stream_response = request_parts.stream;
    let chat_generation_command = prepare_generation_command(
        application_state,
        request_parts,
        resolved_model_id,
        request_id,
    )?;
    log_accepted_request(
        &chat_generation_command,
        should_stream_response,
        request_body_bytes,
    );
    let model_id = chat_generation_command.model.clone();
    let stream_event_receiver = start_generation(
        application_state,
        chat_generation_command,
        configuration_transition_guard,
    )
    .await?;
    let message_id = format!(
        "msg_{}-{request_id}",
        application_state.completion_id_namespace
    );
    Ok(AdmittedMessageGeneration {
        message_id,
        model_id,
        stream_event_receiver,
    })
}

/// Translates the request onto the worker command, pins the resolved model,
/// and fills every setting the request left unset from the reloadable
/// generation defaults.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn prepare_generation_command(
    application_state: &ApplicationState,
    request_parts: AnthropicMessagesRequestParts,
    resolved_model_id: String,
    request_id: u64,
) -> Result<ChatGenerationCommand, Response> {
    let settings_presence = messages_request_generation_settings_presence(&request_parts);
    let mut chat_generation_command =
        translate_anthropic_messages_request_parts(RequestId::new(request_id), request_parts)
            .map_err(translation_rejection_response)?;
    chat_generation_command.model = resolved_model_id;
    apply_generation_defaults(
        application_state.reloadable_config.as_ref(),
        &chat_generation_command.model,
        settings_presence,
        &mut chat_generation_command.settings,
    );
    Ok(chat_generation_command)
}

// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
async fn start_generation(
    application_state: &ApplicationState,
    chat_generation_command: ChatGenerationCommand,
    configuration_transition_guard: tokio::sync::MutexGuard<'_, ()>,
) -> Result<Receiver<ChatGenerationStreamEvent>, Response> {
    let (admission_sender, mut admission_receiver) = tokio::sync::oneshot::channel();
    let mut generation_start_future = application_state
        .generation_executor
        .start_chat_generation_with_admission_signal(chat_generation_command, admission_sender);
    let generation_start_outcome = tokio::select! {
        admission_result = &mut admission_receiver => {
            drop(configuration_transition_guard);
            match admission_result {
                Ok(()) => generation_start_future.await,
                Err(_) => Err(GenerationStartError::WorkerUnavailable),
            }
        }
        generation_start_outcome = &mut generation_start_future => {
            drop(configuration_transition_guard);
            generation_start_outcome
        }
    };
    generation_start_outcome.map_err(generation_start_rejection_response)
}

/// Resolves the target model against worker readiness, the alias table, and
/// the model's capability set.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn resolve_messages_target_model(
    application_state: &ApplicationState,
    requested_model: &str,
) -> Result<String, Response> {
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    if !worker_health_snapshot.status.is_ready() {
        return Err(worker_unavailable_response());
    }
    let Some(resolved_model_id) = application_state.resolve_available_generation_model_id(
        requested_model,
        worker_health_snapshot.ready_model_id.as_deref(),
    ) else {
        return Err(anthropic_error_response(
            StatusCode::NOT_FOUND,
            AnthropicError::not_found(format!("model: {requested_model}")),
        ));
    };
    for discovered_model in application_state.discovered_models_snapshot().iter() {
        if discovered_model.model_id == resolved_model_id
            && !matches!(discovered_model.capabilities, ModelCapabilities::Chat(_))
        {
            return Err(anthropic_error_response(
                StatusCode::BAD_REQUEST,
                AnthropicError::invalid_request(
                    "the requested model does not support Messages generation",
                ),
            ));
        }
    }
    Ok(resolved_model_id)
}

/// `max_tokens` is required by the protocol, so the reloadable default never
/// replaces it; repetition penalty has no Messages spelling.
fn messages_request_generation_settings_presence(
    request_parts: &AnthropicMessagesRequestParts,
) -> RequestGenerationSettingsPresence {
    RequestGenerationSettingsPresence {
        maximum_output_tokens: true,
        temperature: request_parts.temperature.is_some(),
        top_p: request_parts.top_p.is_some(),
        repetition_penalty: false,
    }
}

/// The Messages surface shares the chat request identifier space.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn allocate_messages_request_identifier(
    application_state: &ApplicationState,
) -> Result<u64, Response> {
    application::allocate_chat_request_id(&application_state.next_chat_request_id)
        .ok_or_else(request_identifier_exhausted_response)
}

fn log_accepted_request(
    chat_generation_command: &ChatGenerationCommand,
    should_stream_response: bool,
    request_body_bytes: usize,
) {
    tracing::info!(
        request_id = chat_generation_command.request_id.value(),
        model = %chat_generation_command.model,
        message_count = chat_generation_command.messages.len(),
        tool_count = chat_generation_command.tools.len(),
        stream = should_stream_response,
        request_body_bytes,
        thinking_budget = ?chat_generation_command.settings.thinking_budget,
        "accepted REST Anthropic Messages request"
    );
}

async fn create_non_streaming_response(
    mut stream_event_receiver: Receiver<ChatGenerationStreamEvent>,
    mut message_collector: AnthropicMessageCollector,
) -> Response {
    while let Some(stream_event) = stream_event_receiver.recv().await {
        match stream_event {
            ChatGenerationStreamEvent::Completed {
                prompt_token_count,
                generated_token_count,
                cached_token_count,
                reason,
                ..
            } => {
                let worker_token_counts = WorkerTokenCounts {
                    prompt_token_count,
                    cached_token_count,
                    generated_token_count,
                };
                return match message_collector.into_message(worker_token_counts, reason) {
                    Ok(message) => Json(message).into_response(),
                    Err(assembly_error) => api_error_response(assembly_error.to_string()),
                };
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                return failure_response(&failure_error(&reason));
            }
            ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable) => {
                return worker_unavailable_response();
            }
            non_terminal_event => {
                if let Err(assembly_error) = message_collector.ingest_event(non_terminal_event) {
                    return api_error_response(assembly_error.to_string());
                }
            }
        }
    }
    worker_unavailable_response()
}

fn create_streaming_response(
    stream_event_receiver: Receiver<ChatGenerationStreamEvent>,
    stream_encoder: AnthropicMessagesStreamEncoder,
) -> Response {
    let initial_events = stream_encoder.initial_events();
    let message_event_stream = stream::unfold(
        (stream_event_receiver, stream_encoder, initial_events),
        next_serialized_stream_event,
    );
    Sse::new(message_event_stream).into_response()
}

type MessageStreamState = (
    Receiver<ChatGenerationStreamEvent>,
    AnthropicMessagesStreamEncoder,
    VecDeque<AnthropicMessageStreamEvent>,
);

/// Drains pending encoded events before reading the next worker event, and
/// ends the stream once the encoder has emitted its terminal event. A closed
/// worker channel becomes a worker-unavailable error event so a client never
/// sees a stream that stops without `message_stop` or `error`.
async fn next_serialized_stream_event(
    (mut stream_event_receiver, mut stream_encoder, mut pending_events): MessageStreamState,
) -> Option<(Result<Event, axum::Error>, MessageStreamState)> {
    loop {
        if let Some(message_stream_event) = pending_events.pop_front() {
            let serialized_event = serialize_stream_event(&message_stream_event);
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
            None => {
                ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
            }
        };
        pending_events.extend(stream_encoder.encode(stream_event));
    }
}

fn serialize_stream_event(
    message_stream_event: &AnthropicMessageStreamEvent,
) -> Result<Event, axum::Error> {
    let serialized_payload =
        serde_json::to_string(message_stream_event).map_err(axum::Error::new)?;
    Ok(Event::default()
        .event(message_stream_event.event_type())
        .data(serialized_payload))
}

fn anthropic_error_response(status_code: StatusCode, error: AnthropicError) -> Response {
    (status_code, Json(error.into_response_body())).into_response()
}

fn json_rejection_response(json_error: serde_json::Error) -> Response {
    anthropic_error_response(
        StatusCode::BAD_REQUEST,
        AnthropicError::invalid_request(format!(
            "request body is not a valid Messages request: {json_error}"
        )),
    )
}

fn validation_rejection_response(validation_error: AnthropicMessagesValidationError) -> Response {
    if validation_error.is_request_too_large() {
        return anthropic_error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            AnthropicError::request_too_large(validation_error.to_string()),
        );
    }
    anthropic_error_response(
        StatusCode::BAD_REQUEST,
        AnthropicError::invalid_request(validation_error.to_string()),
    )
}

fn translation_rejection_response(
    translation_error: super::messages_translation::AnthropicMessagesTranslationError,
) -> Response {
    anthropic_error_response(
        StatusCode::BAD_REQUEST,
        AnthropicError::invalid_request(translation_error.to_string()),
    )
}

fn generation_start_rejection_response(generation_start_error: GenerationStartError) -> Response {
    match generation_start_error {
        GenerationStartError::CapacityUnavailable => anthropic_error_response(
            StatusCode::TOO_MANY_REQUESTS,
            AnthropicError::rate_limit("the generation queue is full"),
        ),
        GenerationStartError::ModelLoadFailed {
            model_load_failure_reason,
        } => anthropic_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            AnthropicError::api_error(format!(
                "the model failed to load: {model_load_failure_reason}"
            )),
        ),
        GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes,
            maximum_ipc_message_bytes,
        } => anthropic_error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            AnthropicError::request_too_large(format!(
                "the request expands to {actual_ipc_message_bytes} bytes for local processing, exceeding the {maximum_ipc_message_bytes}-byte limit; reduce image sizes or conversation history"
            )),
        ),
        GenerationStartError::WorkerUnavailable => worker_unavailable_response(),
    }
}

fn failure_response(error: &AnthropicError) -> Response {
    let status_code = match error.error_type {
        "invalid_request_error" => StatusCode::BAD_REQUEST,
        "overloaded_error" => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    anthropic_error_response(status_code, error.clone())
}

fn worker_unavailable_response() -> Response {
    anthropic_error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        AnthropicError::api_error("the local worker is unavailable"),
    )
}

fn api_error_response(message: String) -> Response {
    anthropic_error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        AnthropicError::api_error(message),
    )
}

fn request_identifier_exhausted_response() -> Response {
    anthropic_error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        AnthropicError::api_error("the local request identifier space is exhausted"),
    )
}

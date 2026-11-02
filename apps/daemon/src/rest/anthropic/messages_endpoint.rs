//! `POST /v1/messages`: the Anthropic-compatible Messages endpoint. It shares
//! model resolution, queue admission, and the worker generation path with
//! the OpenAI surfaces and differs only in the wire protocol at both ends.

use astronomical_config::ModelCapabilities;
use astronomical_ipc_protocol::{ChatGenerationCommand, RequestId};
use astronomical_rest_contract::{
    AnthropicError, AnthropicMessagesRequest, AnthropicMessagesRequestParts,
};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    response::{IntoResponse, Response},
};
use tokio::sync::mpsc::Receiver;

use super::messages_assembly::{AnthropicMessageCollector, WorkerTokenCounts};
use super::messages_errors::{
    anthropic_error_response, failure_error, generation_start_error, json_rejection_error,
    model_capability_mismatch_error, request_identifier_exhausted_error,
    translation_rejection_error, unknown_model_error, validation_rejection_error,
    worker_unavailable_error,
};
use super::messages_sse_response::create_streaming_response;
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
    let is_thinking_display_omitted = request_parts.is_thinking_display_omitted;
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
            AnthropicMessagesStreamEncoder::new(message_id, model_id, is_thinking_display_omitted),
        );
    }
    create_non_streaming_response(
        stream_event_receiver,
        AnthropicMessageCollector::new(message_id, model_id, is_thinking_display_omitted),
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
    let messages_request =
        match serde_json::from_slice::<AnthropicMessagesRequest>(request_body_bytes) {
            Ok(messages_request) => messages_request,
            Err(json_error) => {
                return Err(anthropic_error_response(json_rejection_error(&json_error)));
            }
        };
    match messages_request.into_parts() {
        Ok(request_parts) => Ok(request_parts),
        Err(validation_error) => Err(anthropic_error_response(validation_rejection_error(
            &validation_error,
        ))),
    }
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
        match translate_anthropic_messages_request_parts(RequestId::new(request_id), request_parts)
        {
            Ok(chat_generation_command) => chat_generation_command,
            Err(translation_error) => {
                return Err(anthropic_error_response(translation_rejection_error(
                    &translation_error,
                )));
            }
        };
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
    match generation_start_outcome {
        Ok(stream_event_receiver) => Ok(stream_event_receiver),
        Err(start_error) => Err(anthropic_error_response(generation_start_error(
            start_error,
        ))),
    }
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
        return Err(anthropic_error_response(worker_unavailable_error()));
    }
    let Some(resolved_model_id) = application_state.resolve_available_generation_model_id(
        requested_model,
        worker_health_snapshot.ready_model_id.as_deref(),
    ) else {
        return Err(anthropic_error_response(unknown_model_error(
            requested_model,
        )));
    };
    for discovered_model in application_state.discovered_models_snapshot().iter() {
        if discovered_model.model_id == resolved_model_id
            && !matches!(discovered_model.capabilities, ModelCapabilities::Chat(_))
        {
            return Err(anthropic_error_response(model_capability_mismatch_error()));
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
    match application::allocate_chat_request_id(&application_state.next_chat_request_id) {
        Some(request_id) => Ok(request_id),
        None => Err(anthropic_error_response(
            request_identifier_exhausted_error(),
        )),
    }
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
                    Err(assembly_error) => assembly_failure_response(&assembly_error),
                };
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                return anthropic_error_response(failure_error(&reason));
            }
            ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable) => {
                return anthropic_error_response(worker_unavailable_error());
            }
            non_terminal_event => {
                if let Err(assembly_error) = message_collector.ingest_event(non_terminal_event) {
                    return assembly_failure_response(&assembly_error);
                }
            }
        }
    }
    anthropic_error_response(worker_unavailable_error())
}

fn assembly_failure_response(
    assembly_error: &super::messages_assembly::AnthropicMessagesAssemblyError,
) -> Response {
    anthropic_error_response(AnthropicError::api_error(assembly_error.to_string()))
}

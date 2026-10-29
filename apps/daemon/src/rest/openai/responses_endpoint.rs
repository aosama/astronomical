use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use astronomical_config::ModelCapabilities;
use astronomical_ipc_protocol::RequestId;
use astronomical_rest_contract::{
    OpenAiErrorResponse, OpenAiResponseRequestConfiguration, OpenAiResponseStreamEvent,
    OpenAiResponsesRequest, OpenAiResponsesRequestParts, OpenAiResponsesValidationError,
};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Event, sse::Sse},
};
use futures_util::stream;

use crate::application;
use crate::application::ApplicationState;
use crate::{
    ChatGenerationStreamEvent, GenerationStartError, OpenAiResponsesCollector,
    OpenAiResponsesStreamEncoder,
};

pub(crate) async fn create_response(
    State(application_state): State<ApplicationState>,
    request_body_bytes: Bytes,
) -> Response {
    let responses_request = match parse_responses_request(request_body_bytes.as_ref()) {
        Ok(responses_request) => responses_request,
        Err(rejection_response) => return rejection_response,
    };
    let request_parts = match responses_request.into_parts() {
        Ok(request_parts) => request_parts,
        Err(validation_error) => {
            return responses_validation_rejection_response(validation_error);
        }
    };
    // Policy resolution and queue admission must observe one side of a worker replacement.
    let configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    let resolved_model_id =
        match resolve_responses_target_model(&application_state, &request_parts.model) {
            Ok(resolved_model_id) => resolved_model_id,
            Err(rejection_response) => return rejection_response,
        };
    let should_stream_response = request_parts.stream;
    let reasoning_excluded = request_parts.reasoning_excluded;
    let structured_output = request_parts.structured_output.clone();
    let response_instructions = request_parts.instructions.clone();
    let response_request_configuration = request_parts.response_configuration();
    let settings_presence = responses_request_generation_settings_presence(&request_parts);
    let request_id = match allocate_responses_request_identifier(&application_state) {
        Ok(request_id) => request_id,
        Err(rejection_response) => return rejection_response,
    };
    let mut chat_generation_command = match translate_responses_request(request_id, request_parts) {
        Ok(chat_generation_command) => chat_generation_command,
        Err(rejection_response) => return rejection_response,
    };
    chat_generation_command = finalize_responses_generation_command(
        &application_state,
        chat_generation_command,
        resolved_model_id,
        request_id,
        should_stream_response,
        request_body_bytes.len(),
        settings_presence,
    );
    let model_id = chat_generation_command.model.clone();
    let (admission_sender, mut admission_receiver) = tokio::sync::oneshot::channel();
    let mut generation_start_future = application_state
        .generation_executor
        .start_chat_generation_with_admission_signal(chat_generation_command, admission_sender);
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
    let stream_event_receiver = match generation_start_result {
        Ok(stream_event_receiver) => stream_event_receiver,
        Err(generation_start_error) => {
            return super::chat_endpoint::generation_start_rejection_response(
                generation_start_error,
            );
        }
    };
    let created_at_unix_seconds = match responses_stream_timestamp() {
        Ok(created_at_unix_seconds) => created_at_unix_seconds,
        Err(rejection_response) => return rejection_response,
    };
    let response_presentation = OpenAiResponsePresentation {
        response_id: format!(
            "resp_{}-{request_id}",
            application_state.completion_id_namespace
        ),
        created_at_unix_seconds,
        model_id,
        instructions: response_instructions,
        request_configuration: response_request_configuration,
    };
    respond_with_responses_completion(
        stream_event_receiver,
        response_presentation,
        should_stream_response,
        reasoning_excluded,
        &structured_output,
    )
    .await
}

/// Parses the request body, rejecting malformed JSON before any state is
/// touched.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn parse_responses_request(request_body_bytes: &[u8]) -> Result<OpenAiResponsesRequest, Response> {
    serde_json::from_slice::<OpenAiResponsesRequest>(request_body_bytes).map_err(|json_error| {
        invalid_request_response(
            format!("request body is not valid JSON: {json_error}"),
            None,
            "invalid_json",
        )
    })
}

/// Resolves the Responses target model against worker readiness, the
/// available model alias table, and the model's capability set.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn resolve_responses_target_model(
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
        return Err(invalid_request_response(
            "model is not loaded by the local worker",
            Some("model"),
            "model_not_found",
        ));
    };
    if application_state
        .discovered_models_snapshot()
        .iter()
        .find(|discovered_model| discovered_model.model_id == resolved_model_id)
        .is_some_and(|discovered_model| {
            !matches!(discovered_model.capabilities, ModelCapabilities::Chat(_))
        })
    {
        return Err(invalid_request_response(
            "the requested model does not support Responses generation",
            Some("model"),
            "model_capability_mismatch",
        ));
    }
    Ok(resolved_model_id)
}

/// Pure: records which generation settings the request supplied explicitly,
/// so the reloadable defaults only fill the gaps. Repetition penalty is
/// never request-supplied on the Responses surface.
fn responses_request_generation_settings_presence(
    request_parts: &OpenAiResponsesRequestParts,
) -> crate::config::request_generation_defaults::RequestGenerationSettingsPresence {
    crate::config::request_generation_defaults::RequestGenerationSettingsPresence {
        maximum_output_tokens: request_parts.requested_maximum_output_tokens.is_some(),
        temperature: request_parts.temperature.is_some(),
        top_p: request_parts.top_p.is_some(),
        repetition_penalty: false,
    }
}

/// Allocates the next local chat request identifier; the Responses surface
/// shares the chat identifier space.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn allocate_responses_request_identifier(
    application_state: &ApplicationState,
) -> Result<u64, Response> {
    match application::allocate_chat_request_id(&application_state.next_chat_request_id) {
        Some(request_id) => Ok(request_id),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(OpenAiErrorResponse::service_unavailable(
                "the local request identifier space is exhausted",
                Some("request_id_exhausted"),
            )),
        )
            .into_response()),
    }
}

/// Translates the validated request parts into the IPC generation command.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn translate_responses_request(
    request_id: u64,
    request_parts: OpenAiResponsesRequestParts,
) -> Result<astronomical_ipc_protocol::ChatGenerationCommand, Response> {
    super::responses_translation::translate_openai_responses_request_parts(
        RequestId::new(request_id),
        request_parts,
    )
    .map_err(|translation_error| {
        invalid_request_response(translation_error.to_string(), None, "invalid_request")
    })
}

/// Pins the resolved model onto the command and applies the reloadable
/// generation defaults for every setting the request did not supply, then
/// logs the acceptance.
#[allow(clippy::too_many_arguments)]
fn finalize_responses_generation_command(
    application_state: &ApplicationState,
    mut chat_generation_command: astronomical_ipc_protocol::ChatGenerationCommand,
    resolved_model_id: String,
    request_id: u64,
    should_stream_response: bool,
    request_body_bytes: usize,
    settings_presence: crate::config::request_generation_defaults::RequestGenerationSettingsPresence,
) -> astronomical_ipc_protocol::ChatGenerationCommand {
    chat_generation_command.model = resolved_model_id;
    crate::config::request_generation_defaults::apply_generation_defaults(
        application_state.reloadable_config.as_ref(),
        &chat_generation_command.model,
        settings_presence,
        &mut chat_generation_command.settings,
    );
    tracing::info!(
        request_id,
        model = %chat_generation_command.model,
        message_count = chat_generation_command.messages.len(),
        tool_count = chat_generation_command.tools.len(),
        stream = should_stream_response,
        request_body_bytes,
        request_supplied_temperature = settings_presence.temperature,
        temperature_thousandths =
            chat_generation_command.settings.temperature_thousandths,
        "accepted REST Responses request"
    );
    chat_generation_command
}

/// Reads the wall clock for the response timestamp, rejecting when the local
/// clock predates the Unix epoch.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn responses_stream_timestamp() -> Result<u64, Response> {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration_since_epoch) => Ok(duration_since_epoch.as_secs()),
        Err(system_time_error) => {
            tracing::error!(error = %system_time_error, "system clock predates the Unix epoch");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAiErrorResponse::service_unavailable(
                    "the local server could not timestamp the response",
                    Some("response_timestamp_failed"),
                )),
            )
                .into_response())
        }
    }
}

/// Dispatches to the non-streaming collector or the server-sent-events
/// encoder, attaching the unenforced structured output warning either way.
async fn respond_with_responses_completion(
    stream_event_receiver: tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
    response_presentation: OpenAiResponsePresentation,
    should_stream_response: bool,
    reasoning_excluded: bool,
    structured_output: &Option<astronomical_rest_contract::OpenAiStructuredOutput>,
) -> Response {
    if !should_stream_response {
        return crate::serving::structured_output::attach_unenforced_structured_output_warning(
            create_non_streaming_response(
                stream_event_receiver,
                response_presentation,
                reasoning_excluded,
                structured_output.as_ref(),
            )
            .await,
            structured_output.as_ref(),
        );
    }
    let OpenAiResponsePresentation {
        response_id,
        created_at_unix_seconds,
        model_id,
        instructions,
        request_configuration,
    } = response_presentation;
    crate::serving::structured_output::attach_unenforced_structured_output_warning(
        create_streaming_response(
            stream_event_receiver,
            OpenAiResponsesStreamEncoder::new(
                response_id,
                created_at_unix_seconds,
                model_id,
                instructions,
                request_configuration,
                reasoning_excluded,
            ),
        ),
        structured_output.as_ref(),
    )
}
/// The response identity every `/v1/responses` answer is stamped with: the
/// non-streaming collector and the SSE encoder share these five fields.
struct OpenAiResponsePresentation {
    response_id: String,
    created_at_unix_seconds: u64,
    model_id: String,
    instructions: Option<String>,
    request_configuration: OpenAiResponseRequestConfiguration,
}

async fn create_non_streaming_response(
    mut stream_event_receiver: tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
    response_presentation: OpenAiResponsePresentation,
    reasoning_excluded: bool,
    structured_output: Option<&astronomical_rest_contract::OpenAiStructuredOutput>,
) -> Response {
    let OpenAiResponsePresentation {
        response_id,
        created_at_unix_seconds,
        model_id,
        instructions,
        request_configuration,
    } = response_presentation;
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
            if structured_output.is_some() {
                response_collector.replace_output_text_with_extracted_json();
            }
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
        if let ChatGenerationStreamEvent::ReasoningFragment(_) = &stream_event
            && reasoning_excluded
        {
            // The model still thought; the token count still lands in usage via
            // the Completed event. Withhold only the client-visible reasoning item.
            continue;
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

/// Renders a Responses contract rejection as HTTP 413 when it is a
/// portable-budget rejection, keeping the OpenAI-shaped error code on the same
/// `request_too_large` path the IPC frame overflow uses.
fn responses_validation_rejection_response(
    validation_error: OpenAiResponsesValidationError,
) -> Response {
    if validation_error.is_request_too_large() {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(OpenAiErrorResponse::invalid_request(
                validation_error.to_string(),
                None,
                Some("request_too_large"),
            )),
        )
            .into_response();
    }
    invalid_request_response(validation_error.to_string(), None, "invalid_request")
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

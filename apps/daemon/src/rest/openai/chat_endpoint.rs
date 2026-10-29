use std::time::{SystemTime, UNIX_EPOCH};

use astronomical_config::ModelCapabilities;
use astronomical_ipc_protocol::RequestId;
use astronomical_rest_contract::{
    OpenAiChatCompletionRequest, OpenAiChatCompletionRequestParts,
    OpenAiChatCompletionValidationError, OpenAiErrorResponse,
};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Sse},
};
use futures_util::stream;

use crate::{
    GenerationStartError, OpenAiChatRequestDiagnosticSnapshot,
    OpenAiChatRequestInfoDiagnosticSnapshot, build_openai_chat_request_diagnostic_snapshot,
    build_openai_chat_request_info_diagnostic_snapshot,
    rest::openai::chat_completion::create_non_streaming_chat_completion,
    rest::openai::chat_stream::OpenAiChatStreamEncoder,
};

use crate::application;
use crate::application::ApplicationState;

pub(crate) async fn create_chat_completion(
    State(application_state): State<ApplicationState>,
    request_body_bytes: Bytes,
) -> Response {
    let request_diagnostic_snapshot =
        build_openai_chat_request_diagnostic_snapshot(request_body_bytes.as_ref());
    let request_info_diagnostic_snapshot =
        build_openai_chat_request_info_diagnostic_snapshot(request_body_bytes.as_ref());
    let chat_completion_request =
        match parse_chat_completion_json(request_body_bytes.as_ref(), &request_diagnostic_snapshot)
        {
            Ok(chat_completion_request) => chat_completion_request,
            Err(rejection_response) => return rejection_response,
        };
    log_chat_request_received(&chat_completion_request);
    if let Err(rejection_response) = validate_chat_completion_request(
        &chat_completion_request,
        &request_diagnostic_snapshot,
        &request_info_diagnostic_snapshot,
    ) {
        return rejection_response;
    }
    // Policy resolution and queue admission must observe one side of a worker replacement.
    let configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    let requested_model = chat_completion_request.model();
    let resolved_model_id = match resolve_chat_target_model(&application_state, requested_model) {
        Ok(resolved_model_id) => resolved_model_id,
        Err(rejection_response) => return rejection_response,
    };
    let (should_stream_response, includes_usage) =
        chat_streaming_preferences(&chat_completion_request);
    let request_parts = match chat_completion_request.into_parts() {
        Ok(request_parts) => request_parts,
        Err(validation_error) => return chat_validation_rejection_response(validation_error),
    };
    let reasoning_excluded = request_parts.reasoning_excluded;
    let structured_output = request_parts.structured_output.clone();
    let settings_presence = chat_request_generation_settings_presence(&request_parts);
    let request_id = match allocate_chat_request_identifier(&application_state) {
        Ok(request_id) => request_id,
        Err(rejection_response) => return rejection_response,
    };
    let mut chat_generation_command = match translate_openai_chat_request(
        request_id,
        request_parts,
        &request_diagnostic_snapshot,
        &request_info_diagnostic_snapshot,
    ) {
        Ok(chat_generation_command) => chat_generation_command,
        Err(rejection_response) => return rejection_response,
    };
    chat_generation_command = finalize_chat_generation_command(
        &application_state,
        chat_generation_command,
        resolved_model_id,
        request_id,
        settings_presence,
        &request_info_diagnostic_snapshot,
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
    let chat_stream_event_receiver = match generation_start_result {
        Ok(chat_stream_event_receiver) => chat_stream_event_receiver,
        Err(generation_start_error) => {
            return generation_start_rejection_response(generation_start_error);
        }
    };
    let created_at_unix_seconds = match chat_stream_timestamp() {
        Ok(created_at_unix_seconds) => created_at_unix_seconds,
        Err(rejection_response) => return rejection_response,
    };
    respond_with_chat_completion(
        application_state.completion_id_namespace.as_ref(),
        request_id,
        chat_stream_event_receiver,
        should_stream_response,
        includes_usage,
        created_at_unix_seconds,
        model_id,
        reasoning_excluded,
        &structured_output,
    )
    .await
}

/// Reads the streaming preferences from the request.
fn chat_streaming_preferences(
    chat_completion_request: &OpenAiChatCompletionRequest,
) -> (bool, bool) {
    (
        chat_completion_request.stream(),
        chat_completion_request.includes_usage_in_stream(),
    )
}

/// Pure: records which generation settings the request supplied explicitly,
/// so the reloadable defaults only fill the gaps. Repetition penalty is
/// never request-supplied on the OpenAI surface.
fn chat_request_generation_settings_presence(
    request_parts: &OpenAiChatCompletionRequestParts,
) -> crate::config::request_generation_defaults::RequestGenerationSettingsPresence {
    crate::config::request_generation_defaults::RequestGenerationSettingsPresence {
        maximum_output_tokens: request_parts.requested_maximum_output_tokens.is_some(),
        temperature: request_parts.temperature.is_some(),
        top_p: request_parts.top_p.is_some(),
        repetition_penalty: false,
    }
}

/// Builds the completion identifier and dispatches to the non-streaming or
/// server-sent-events response body, attaching the unenforced structured
/// output warning either way.
#[allow(clippy::too_many_arguments)]
async fn respond_with_chat_completion(
    completion_id_namespace: &str,
    request_id: u64,
    chat_stream_event_receiver: tokio::sync::mpsc::Receiver<crate::ChatGenerationStreamEvent>,
    should_stream_response: bool,
    includes_usage: bool,
    created_at_unix_seconds: u64,
    model_id: String,
    reasoning_excluded: bool,
    structured_output: &Option<astronomical_rest_contract::OpenAiStructuredOutput>,
) -> Response {
    let completion_id = format!("chatcmpl-{completion_id_namespace}-{request_id}");
    if !should_stream_response {
        return crate::serving::structured_output::attach_unenforced_structured_output_warning(
            create_non_streaming_chat_completion(
                chat_stream_event_receiver,
                completion_id,
                created_at_unix_seconds,
                model_id,
                reasoning_excluded,
                structured_output.as_ref(),
            )
            .await,
            structured_output.as_ref(),
        );
    }
    crate::serving::structured_output::attach_unenforced_structured_output_warning(
        create_streaming_response(
            chat_stream_event_receiver,
            OpenAiChatStreamEncoder::new(
                request_id,
                completion_id,
                created_at_unix_seconds,
                model_id,
                includes_usage,
                reasoning_excluded,
            ),
        ),
        structured_output.as_ref(),
    )
}

/// Parses the request body, rejecting malformed JSON before any state is
/// touched.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn parse_chat_completion_json(
    request_body_bytes: &[u8],
    request_diagnostic_snapshot: &OpenAiChatRequestDiagnosticSnapshot,
) -> Result<OpenAiChatCompletionRequest, Response> {
    match serde_json::from_slice::<OpenAiChatCompletionRequest>(request_body_bytes) {
        Ok(chat_completion_request) => Ok(chat_completion_request),
        Err(json_error) => {
            tracing::warn!(
                error = %json_error,
                request_body_bytes = request_diagnostic_snapshot.request_body_bytes,
                request_body_sha256 = %request_diagnostic_snapshot.request_body_sha256,
                "rejected malformed REST chat completion JSON"
            );
            Err((
                StatusCode::BAD_REQUEST,
                Json(OpenAiErrorResponse::invalid_request(
                    format!("request body is not valid JSON: {json_error}"),
                    None,
                    Some("invalid_json"),
                )),
            )
                .into_response())
        }
    }
}

/// Logs the received request at debug level for request correlation.
fn log_chat_request_received(chat_completion_request: &OpenAiChatCompletionRequest) {
    tracing::debug!(
        model = chat_completion_request.model(),
        message_count = chat_completion_request.messages().len(),
        tool_count = chat_completion_request.tools().len(),
        maximum_output_tokens = chat_completion_request.maximum_output_tokens(),
        stream = chat_completion_request.stream(),
        "received REST chat completion request"
    );
}

/// Runs the portable chat-contract validation, rejecting invalid requests
/// with the OpenAI-shaped error body.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn validate_chat_completion_request(
    chat_completion_request: &OpenAiChatCompletionRequest,
    request_diagnostic_snapshot: &OpenAiChatRequestDiagnosticSnapshot,
    request_info_diagnostic_snapshot: &OpenAiChatRequestInfoDiagnosticSnapshot,
) -> Result<(), Response> {
    if let Err(validation_error) = chat_completion_request.validate() {
        tracing::warn!(
            error = %validation_error,
            request_body_bytes = request_diagnostic_snapshot.request_body_bytes,
            request_body_sha256 = %request_diagnostic_snapshot.request_body_sha256,
            message_count = ?request_info_diagnostic_snapshot.message_count,
            message_role_sequence_preview = ?request_info_diagnostic_snapshot.message_role_sequence_preview,
            "rejected invalid REST chat completion request"
        );
        return Err(chat_validation_rejection_response(validation_error));
    }
    Ok(())
}

/// Resolves the chat target model against worker readiness, the available
/// model alias table, and the model's capability set.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn resolve_chat_target_model(
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
        return Err((
            StatusCode::BAD_REQUEST,
            Json(OpenAiErrorResponse::invalid_request(
                "model is not loaded by the local worker",
                Some("model"),
                Some("model_not_found"),
            )),
        )
            .into_response());
    };
    if application_state
        .discovered_models_snapshot()
        .iter()
        .find(|discovered_model| discovered_model.model_id == resolved_model_id)
        .is_some_and(|discovered_model| {
            !matches!(discovered_model.capabilities, ModelCapabilities::Chat(_))
        })
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(OpenAiErrorResponse::invalid_request(
                "the requested model does not support chat generation",
                Some("model"),
                Some("model_capability_mismatch"),
            )),
        )
            .into_response());
    }
    Ok(resolved_model_id)
}

/// Allocates the next local chat request identifier.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn allocate_chat_request_identifier(application_state: &ApplicationState) -> Result<u64, Response> {
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

/// Translates the validated request parts into the IPC generation command,
/// rejecting translation failures with the request-correlation diagnostics.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn translate_openai_chat_request(
    request_id: u64,
    request_parts: OpenAiChatCompletionRequestParts,
    request_diagnostic_snapshot: &OpenAiChatRequestDiagnosticSnapshot,
    request_info_diagnostic_snapshot: &OpenAiChatRequestInfoDiagnosticSnapshot,
) -> Result<astronomical_ipc_protocol::ChatGenerationCommand, Response> {
    match crate::rest::openai::chat_translation::translate_openai_chat_completion_request_parts(
        RequestId::new(request_id),
        request_parts,
    ) {
        Ok(chat_generation_command) => Ok(chat_generation_command),
        Err(translation_error) => {
            tracing::warn!(
                request_id,
                error = %translation_error,
                request_body_bytes = request_diagnostic_snapshot.request_body_bytes,
                request_body_sha256 = %request_diagnostic_snapshot.request_body_sha256,
                message_count = ?request_info_diagnostic_snapshot.message_count,
                message_role_sequence_preview = ?request_info_diagnostic_snapshot.message_role_sequence_preview,
                "rejected REST chat completion during IPC translation"
            );
            Err((
                StatusCode::BAD_REQUEST,
                Json(OpenAiErrorResponse::invalid_request(
                    translation_error.to_string(),
                    None,
                    Some("invalid_request"),
                )),
            )
                .into_response())
        }
    }
}

/// Pins the resolved model onto the command, applies the reloadable
/// generation defaults for every setting the request did not supply, and
/// logs the acceptance with the request-correlation diagnostics.
fn finalize_chat_generation_command(
    application_state: &ApplicationState,
    mut chat_generation_command: astronomical_ipc_protocol::ChatGenerationCommand,
    resolved_model_id: String,
    request_id: u64,
    settings_presence: crate::config::request_generation_defaults::RequestGenerationSettingsPresence,
    request_info_diagnostic_snapshot: &OpenAiChatRequestInfoDiagnosticSnapshot,
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
        message_count = ?request_info_diagnostic_snapshot.message_count,
        last_user_message_character_count = ?request_info_diagnostic_snapshot.last_user_message_character_count,
        last_user_message_sha256 = ?request_info_diagnostic_snapshot.last_user_message_sha256,
        request_supplied_temperature = settings_presence.temperature,
        temperature_thousandths =
            chat_generation_command.settings.temperature_thousandths,
        "accepted REST chat completion request"
    );
    chat_generation_command
}

/// Maps a failed generation start to its OpenAI-shaped HTTP rejection.
fn generation_start_rejection_response(generation_start_error: GenerationStartError) -> Response {
    match generation_start_error {
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
        GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes,
            maximum_ipc_message_bytes,
        } => (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(OpenAiErrorResponse::invalid_request(
                format!(
                    "the request expands to {actual_ipc_message_bytes} bytes for local processing, exceeding the {maximum_ipc_message_bytes}-byte limit; reduce image sizes or conversation history"
                ),
                None,
                Some("request_too_large"),
            )),
        )
            .into_response(),
        GenerationStartError::WorkerUnavailable => worker_unavailable_response(),
    }
}

/// Reads the wall clock for the chat stream timestamp, rejecting when the
/// local clock predates the Unix epoch.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn chat_stream_timestamp() -> Result<u64, Response> {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration_since_epoch) => Ok(duration_since_epoch.as_secs()),
        Err(system_time_error) => {
            tracing::error!(error = %system_time_error, "system clock predates the Unix epoch");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAiErrorResponse::service_unavailable(
                    "the local server could not timestamp the chat stream",
                    Some("chat_stream_timestamp_failed"),
                )),
            )
                .into_response())
        }
    }
}

fn create_streaming_response(
    chat_stream_event_receiver: tokio::sync::mpsc::Receiver<crate::ChatGenerationStreamEvent>,
    stream_encoder: OpenAiChatStreamEncoder,
) -> Response {
    let initial_event = match stream_encoder.initial_event() {
        Ok(initial_event) => initial_event,
        Err(encoding_error) => {
            tracing::error!(error = %encoding_error, "failed to encode initial OpenAI chat chunk");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAiErrorResponse::service_unavailable(
                    "the local server could not start the chat stream",
                    Some("chat_stream_encoding_failed"),
                )),
            )
                .into_response();
        }
    };
    let sse_event_stream = stream::unfold(
        (
            chat_stream_event_receiver,
            stream_encoder,
            Some(initial_event),
        ),
        |(mut chat_stream_event_receiver, stream_encoder, mut pending_event)| async move {
            if let Some(event) = pending_event.take() {
                return Some((
                    Ok::<_, axum::Error>(event),
                    (chat_stream_event_receiver, stream_encoder, pending_event),
                ));
            }
            loop {
                let chat_stream_event = chat_stream_event_receiver.recv().await?;
                match stream_encoder.encode(chat_stream_event) {
                    Ok(mut encoded_events) => {
                        let Some(event) = encoded_events.pop_front() else {
                            continue;
                        };
                        return Some((
                            Ok(event),
                            (
                                chat_stream_event_receiver,
                                stream_encoder,
                                encoded_events.pop_front(),
                            ),
                        ));
                    }
                    Err(encoding_error) => {
                        return Some((
                            Err(encoding_error),
                            (chat_stream_event_receiver, stream_encoder, None),
                        ));
                    }
                }
            }
        },
    );

    Sse::new(sse_event_stream).into_response()
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

/// Renders a chat contract rejection as HTTP 413 when it is a portable-budget
/// rejection, keeping the OpenAI-shaped error code on the same
/// `request_too_large` path the IPC frame overflow uses.
fn chat_validation_rejection_response(
    validation_error: OpenAiChatCompletionValidationError,
) -> Response {
    let (status, error_code) = if validation_error.is_request_too_large() {
        (StatusCode::PAYLOAD_TOO_LARGE, "request_too_large")
    } else {
        (StatusCode::BAD_REQUEST, "invalid_request")
    };
    (
        status,
        Json(OpenAiErrorResponse::invalid_request(
            validation_error.to_string(),
            None,
            Some(error_code),
        )),
    )
        .into_response()
}

#[cfg(test)]
#[path = "chat_endpoint_tests.rs"]
mod chat_endpoint_tests;

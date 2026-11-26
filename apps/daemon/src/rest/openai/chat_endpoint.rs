use crate::rest::worker_replacement_wait::lock_configuration_transition_for_generation;

use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Sse},
};
use futures_util::stream;
use ipc_protocol::RequestId;
use rest_contract::{
    OpenAiChatCompletionRequest, OpenAiChatCompletionRequestParts,
    OpenAiChatCompletionValidationError, OpenAiErrorResponse,
};

use crate::{
    OpenAiChatRequestDiagnosticSnapshot, OpenAiChatRequestInfoDiagnosticSnapshot,
    build_openai_chat_request_diagnostic_snapshot,
    build_openai_chat_request_info_diagnostic_snapshot,
    rest::openai::chat_completion::create_non_streaming_chat_completion,
    rest::openai::chat_stream::OpenAiChatStreamEncoder,
};

use crate::application::ApplicationState;
use crate::rest::openai::chat_generation_admission::{
    allocate_chat_request_identifier, apply_configured_generation_settings,
    generation_start_rejection_response, generation_timestamp,
    request_generation_settings_presence, resolve_chat_target_model,
    start_chat_generation_after_admission,
};

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
        lock_configuration_transition_for_generation(&application_state).await;
    let requested_model = chat_completion_request.model();
    let resolved_model_id =
        match resolve_chat_target_model(&application_state, requested_model, "chat") {
            Ok(resolved_model_id) => resolved_model_id,
            Err(rejection_response) => return rejection_response,
        };
    let prepared_generation = match prepare_chat_generation(
        &application_state,
        chat_completion_request,
        resolved_model_id,
        &request_diagnostic_snapshot,
        &request_info_diagnostic_snapshot,
    ) {
        Ok(prepared_generation) => prepared_generation,
        Err(rejection_response) => return rejection_response,
    };
    start_chat_generation_and_respond(
        &application_state,
        prepared_generation,
        configuration_transition_guard,
    )
    .await
}

/// How the reply is delivered once the generation has started.
struct ChatReplyPlan {
    should_stream_response: bool,
    includes_usage: bool,
    reasoning_excluded: bool,
    structured_output: Option<rest_contract::OpenAiStructuredOutput>,
}

/// A chat request that has been validated, translated and given its
/// request identifier, ready for queue admission.
struct PreparedChatGeneration {
    chat_generation_command: ipc_protocol::ChatGenerationCommand,
    request_id: u64,
    reply_plan: ChatReplyPlan,
}

/// Splits a request into its parts and the reply plan. The streaming
/// preferences have to be read first because `into_parts` consumes the request.
fn split_chat_request(
    chat_completion_request: OpenAiChatCompletionRequest,
) -> Result<(OpenAiChatCompletionRequestParts, ChatReplyPlan), OpenAiChatCompletionValidationError>
{
    let (should_stream_response, includes_usage) =
        chat_streaming_preferences(&chat_completion_request);
    let request_parts = chat_completion_request.into_parts()?;
    let reply_plan = ChatReplyPlan {
        should_stream_response,
        includes_usage,
        reasoning_excluded: request_parts.reasoning_excluded,
        structured_output: request_parts.structured_output.clone(),
    };
    Ok((request_parts, reply_plan))
}

/// Turns the validated request into the IPC generation command, allocating
/// the request identifier only once the request parts are known to be valid.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn prepare_chat_generation(
    application_state: &ApplicationState,
    chat_completion_request: OpenAiChatCompletionRequest,
    resolved_model_id: String,
    request_diagnostic_snapshot: &OpenAiChatRequestDiagnosticSnapshot,
    request_info_diagnostic_snapshot: &OpenAiChatRequestInfoDiagnosticSnapshot,
) -> Result<PreparedChatGeneration, Response> {
    let (request_parts, reply_plan) = match split_chat_request(chat_completion_request) {
        Ok(split_request) => split_request,
        Err(validation_error) => return Err(chat_validation_rejection_response(validation_error)),
    };
    let settings_presence = request_generation_settings_presence(
        request_parts.requested_maximum_output_tokens,
        request_parts.temperature,
        request_parts.top_p,
    );
    let request_id = allocate_chat_request_identifier(application_state)?;
    let chat_generation_command = translate_openai_chat_request(
        request_id,
        request_parts,
        request_diagnostic_snapshot,
        request_info_diagnostic_snapshot,
    )?;
    let chat_generation_command = finalize_chat_generation_command(
        application_state,
        chat_generation_command,
        resolved_model_id,
        request_id,
        settings_presence,
        request_info_diagnostic_snapshot,
    );
    Ok(PreparedChatGeneration {
        chat_generation_command,
        request_id,
        reply_plan,
    })
}

/// Admits the prepared generation to the worker queue and answers with its
/// completion, or with the rejection that stopped it.
async fn start_chat_generation_and_respond(
    application_state: &ApplicationState,
    prepared_generation: PreparedChatGeneration,
    configuration_transition_guard: tokio::sync::MutexGuard<'_, ()>,
) -> Response {
    let PreparedChatGeneration {
        chat_generation_command,
        request_id,
        reply_plan,
    } = prepared_generation;
    let model_id = chat_generation_command.model.clone();
    let generation_start_result = start_chat_generation_after_admission(
        application_state,
        chat_generation_command,
        configuration_transition_guard,
    )
    .await;
    let chat_stream_event_receiver = match generation_start_result {
        Ok(chat_stream_event_receiver) => chat_stream_event_receiver,
        Err(generation_start_error) => {
            return generation_start_rejection_response(generation_start_error);
        }
    };
    let created_at_unix_seconds = match generation_timestamp(
        "the local server could not timestamp the chat stream",
        "chat_stream_timestamp_failed",
    ) {
        Ok(created_at_unix_seconds) => created_at_unix_seconds,
        Err(rejection_response) => return rejection_response,
    };
    respond_with_chat_completion(
        application_state.completion_id_namespace.as_ref(),
        request_id,
        chat_stream_event_receiver,
        created_at_unix_seconds,
        model_id,
        &reply_plan,
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

/// Builds the completion identifier and dispatches to the non-streaming or
/// server-sent-events response body, attaching the unenforced structured
/// output warning either way.
async fn respond_with_chat_completion(
    completion_id_namespace: &str,
    request_id: u64,
    chat_stream_event_receiver: tokio::sync::mpsc::Receiver<crate::ChatGenerationStreamEvent>,
    created_at_unix_seconds: u64,
    model_id: String,
    reply_plan: &ChatReplyPlan,
) -> Response {
    let completion_id = format!("chatcmpl-{completion_id_namespace}-{request_id}");
    let structured_output = reply_plan.structured_output.as_ref();
    if !reply_plan.should_stream_response {
        return crate::serving::structured_output::attach_unenforced_structured_output_warning(
            create_non_streaming_chat_completion(
                chat_stream_event_receiver,
                completion_id,
                created_at_unix_seconds,
                model_id,
                reply_plan.reasoning_excluded,
                structured_output,
            )
            .await,
            structured_output,
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
                reply_plan.includes_usage,
                reply_plan.reasoning_excluded,
            ),
        ),
        structured_output,
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
) -> Result<ipc_protocol::ChatGenerationCommand, Response> {
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
    mut chat_generation_command: ipc_protocol::ChatGenerationCommand,
    resolved_model_id: String,
    request_id: u64,
    settings_presence: crate::config::request_generation_defaults::RequestGenerationSettingsPresence,
    request_info_diagnostic_snapshot: &OpenAiChatRequestInfoDiagnosticSnapshot,
) -> ipc_protocol::ChatGenerationCommand {
    apply_configured_generation_settings(
        application_state,
        &mut chat_generation_command,
        resolved_model_id,
        settings_presence,
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

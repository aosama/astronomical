use std::time::{SystemTime, UNIX_EPOCH};

use astronomical_config::ModelCapabilities;
use astronomical_ipc_protocol::RequestId;
use astronomical_rest_contract::{OpenAiChatCompletionRequest, OpenAiErrorResponse};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Sse},
};
use futures_util::stream;

use crate::{
    GenerationStartError, build_openai_chat_request_diagnostic_snapshot,
    build_openai_chat_request_info_diagnostic_snapshot,
    openai_chat_completion::create_non_streaming_chat_completion,
    openai_chat_stream::OpenAiChatStreamEncoder,
};

use crate::application::{ApplicationState, allocate_chat_request_id};

pub(crate) async fn create_chat_completion(
    State(application_state): State<ApplicationState>,
    request_body_bytes: Bytes,
) -> Response {
    let request_diagnostic_snapshot =
        build_openai_chat_request_diagnostic_snapshot(request_body_bytes.as_ref());
    let request_info_diagnostic_snapshot =
        build_openai_chat_request_info_diagnostic_snapshot(request_body_bytes.as_ref());
    tracing::trace!(
        request_body_bytes = request_diagnostic_snapshot.request_body_bytes,
        request_body_sha256 = %request_diagnostic_snapshot.request_body_sha256,
        "captured REST chat completion request metadata for diagnostics"
    );
    let chat_completion_request =
        match serde_json::from_slice::<OpenAiChatCompletionRequest>(request_body_bytes.as_ref()) {
            Ok(chat_completion_request) => chat_completion_request,
            Err(json_error) => {
                tracing::warn!(
                    error = %json_error,
                    request_body_bytes = request_diagnostic_snapshot.request_body_bytes,
                    request_body_sha256 = %request_diagnostic_snapshot.request_body_sha256,
                    "rejected malformed REST chat completion JSON"
                );
                return (
                    StatusCode::BAD_REQUEST,
                    Json(OpenAiErrorResponse::invalid_request(
                        format!("request body is not valid JSON: {json_error}"),
                        None,
                        Some("invalid_json"),
                    )),
                )
                    .into_response();
            }
        };
    tracing::debug!(
        model = chat_completion_request.model(),
        message_count = chat_completion_request.messages().len(),
        tool_count = chat_completion_request.tools().len(),
        maximum_output_tokens = chat_completion_request.maximum_output_tokens(),
        stream = chat_completion_request.stream(),
        "received REST chat completion request"
    );
    if let Err(validation_error) = chat_completion_request.validate() {
        tracing::warn!(
            error = %validation_error,
            request_body_bytes = request_diagnostic_snapshot.request_body_bytes,
            request_body_sha256 = %request_diagnostic_snapshot.request_body_sha256,
            message_count = ?request_info_diagnostic_snapshot.message_count,
            message_role_sequence_preview = ?request_info_diagnostic_snapshot.message_role_sequence_preview,
            "rejected invalid REST chat completion request"
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(OpenAiErrorResponse::invalid_request(
                validation_error.to_string(),
                None,
                Some("invalid_request"),
            )),
        )
            .into_response();
    }
    // Policy resolution and queue admission must observe one side of a worker replacement.
    let configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    if !worker_health_snapshot.status.is_ready() {
        return worker_unavailable_response();
    }
    let requested_model = chat_completion_request.model();
    let Some(resolved_model_id) = application_state.resolve_available_generation_model_id(
        requested_model,
        worker_health_snapshot.ready_model_id.as_deref(),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(OpenAiErrorResponse::invalid_request(
                "model is not loaded by the local worker",
                Some("model"),
                Some("model_not_found"),
            )),
        )
            .into_response();
    };
    if application_state
        .discovered_models_snapshot()
        .iter()
        .find(|discovered_model| discovered_model.model_id == resolved_model_id)
        .is_some_and(|discovered_model| {
            !matches!(discovered_model.capabilities, ModelCapabilities::Chat(_))
        })
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(OpenAiErrorResponse::invalid_request(
                "the requested model does not support chat generation",
                Some("model"),
                Some("model_capability_mismatch"),
            )),
        )
            .into_response();
    }
    let should_stream_response = chat_completion_request.stream();
    let includes_usage = chat_completion_request.includes_usage_in_stream();
    let request_parts = match chat_completion_request.into_parts() {
        Ok(request_parts) => request_parts,
        Err(validation_error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(OpenAiErrorResponse::invalid_request(
                    validation_error.to_string(),
                    None,
                    Some("invalid_request"),
                )),
            )
                .into_response();
        }
    };
    let structured_output = request_parts.structured_output.clone();
    let settings_presence = crate::request_generation_defaults::RequestGenerationSettingsPresence {
        maximum_output_tokens: request_parts.requested_maximum_output_tokens.is_some(),
        temperature: request_parts.temperature.is_some(),
        top_p: request_parts.top_p.is_some(),
    };
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
        match crate::openai_chat_translation::translate_openai_chat_completion_request_parts(
            RequestId::new(request_id),
            request_parts,
        ) {
            Ok(chat_generation_command) => chat_generation_command,
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
                return (
                    StatusCode::BAD_REQUEST,
                    Json(OpenAiErrorResponse::invalid_request(
                        translation_error.to_string(),
                        None,
                        Some("invalid_request"),
                    )),
                )
                    .into_response();
            }
        };
    chat_generation_command.model = resolved_model_id;
    crate::request_generation_defaults::apply_generation_defaults(
        application_state.reloadable_config.as_ref(),
        &chat_generation_command.model,
        settings_presence,
        &mut chat_generation_command.settings,
    );
    chat_generation_command.qwen_thinking_channel_seed =
        crate::load_configured_qwen_thinking_channel_seed(
            &application_state,
            &chat_generation_command.model,
        )
        .await;
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
        Err(GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes,
            maximum_ipc_message_bytes,
        }) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(OpenAiErrorResponse::invalid_request(
                    format!(
                        "the request expands to {actual_ipc_message_bytes} bytes for local processing, exceeding the {maximum_ipc_message_bytes}-byte limit; reduce image sizes or conversation history"
                    ),
                    None,
                    Some("request_too_large"),
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
                    "the local server could not timestamp the chat stream",
                    Some("chat_stream_timestamp_failed"),
                )),
            )
                .into_response();
        }
    };
    let completion_id = format!(
        "chatcmpl-{}-{request_id}",
        application_state.completion_id_namespace
    );
    if !should_stream_response {
        return crate::structured_output::attach_unenforced_structured_output_warning(
            create_non_streaming_chat_completion(
                chat_stream_event_receiver,
                completion_id,
                created_at_unix_seconds,
                model_id,
                structured_output.as_ref(),
            )
            .await,
            structured_output.as_ref(),
        );
    }
    crate::structured_output::attach_unenforced_structured_output_warning(
        create_streaming_response(
            chat_stream_event_receiver,
            OpenAiChatStreamEncoder::new(
                request_id,
                completion_id,
                created_at_unix_seconds,
                model_id,
                includes_usage,
            ),
        ),
        structured_output.as_ref(),
    )
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

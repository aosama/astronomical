//! The steps every request that runs as a chat generation takes at the REST
//! edge, whichever OpenAI surface it arrived on: resolve the loaded chat
//! model, allocate the request identifier, fill the settings the request left
//! out from the configured defaults, start the generation while holding the
//! configuration transition until admission, and timestamp the result, with
//! the OpenAI-shaped rejection for each. Chat Completions and Responses each
//! wrote these steps before; each now keeps only its own parsing, translation,
//! logging and response shape.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use config::ModelCapabilities;
use rest_contract::OpenAiErrorResponse;
use tokio::sync::{MutexGuard as AsyncMutexGuard, mpsc};

use crate::application;
use crate::application::ApplicationState;
use crate::config::request_generation_defaults::{
    RequestGenerationSettingsPresence, apply_generation_defaults,
};
use crate::serving::skill_declaration::declare_astronomical_skill;
use crate::{ChatGenerationStreamEvent, GenerationStartError};

/// The loaded chat model a request may run on: the worker must be ready, the
/// requested name must resolve to a loaded model, and that model must chat.
/// `surface_name` names the generation in the capability refusal ("chat",
/// "Responses").
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
pub(crate) fn resolve_chat_target_model(
    application_state: &ApplicationState,
    requested_model: &str,
    surface_name: &str,
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
        return Err(model_rejection_response(
            "model is not loaded by the local worker".to_owned(),
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
        return Err(model_rejection_response(
            format!("the requested model does not support {surface_name} generation"),
            "model_capability_mismatch",
        ));
    }
    Ok(resolved_model_id)
}

fn model_rejection_response(message: String, code: &'static str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(OpenAiErrorResponse::invalid_request(
            message,
            Some("model"),
            Some(code),
        )),
    )
        .into_response()
}

/// Allocates the next local request identifier; every chat-generation
/// surface shares one identifier space.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
pub(crate) fn allocate_chat_request_identifier(
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

/// Which generation settings the request supplied, so the configured
/// defaults fill only the gaps. Repetition penalty is never request-supplied
/// on the OpenAI surfaces.
pub(crate) fn request_generation_settings_presence(
    requested_maximum_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> RequestGenerationSettingsPresence {
    RequestGenerationSettingsPresence {
        maximum_output_tokens: requested_maximum_output_tokens.is_some(),
        temperature: temperature.is_some(),
        top_p: top_p.is_some(),
        repetition_penalty: false,
    }
}

/// Pins the resolved model onto the command, fills every setting the request
/// did not supply from the configured defaults, and declares the
/// Astronomical skill when the configuration asks for it.
pub(crate) fn apply_configured_generation_settings(
    application_state: &ApplicationState,
    chat_generation_command: &mut ipc_protocol::ChatGenerationCommand,
    resolved_model_id: String,
    settings_presence: RequestGenerationSettingsPresence,
) {
    chat_generation_command.model = resolved_model_id;
    apply_generation_defaults(
        application_state.reloadable_config.as_ref(),
        &chat_generation_command.model,
        settings_presence,
        &mut chat_generation_command.settings,
    );
    declare_astronomical_skill(
        application_state.reloadable_config.as_ref(),
        &mut chat_generation_command.messages,
    );
}

/// Starts the generation and releases the configuration transition as soon
/// as the request is admitted to the queue, or as soon as its start fails, so
/// a worker replacement waits only for admission, never for the reply.
pub(crate) async fn start_chat_generation_after_admission(
    application_state: &ApplicationState,
    chat_generation_command: ipc_protocol::ChatGenerationCommand,
    configuration_transition_guard: AsyncMutexGuard<'_, ()>,
) -> Result<mpsc::Receiver<ChatGenerationStreamEvent>, GenerationStartError> {
    let (admission_sender, mut admission_receiver) = tokio::sync::oneshot::channel();
    let mut generation_start_future = application_state
        .generation_executor
        .start_chat_generation_with_admission_signal(chat_generation_command, admission_sender);
    tokio::select! {
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
    }
}

/// Reads the wall clock for the reply's timestamp, rejecting with
/// `failure_message` and `failure_code` when the local clock predates the
/// Unix epoch.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
pub(crate) fn generation_timestamp(
    failure_message: &'static str,
    failure_code: &'static str,
) -> Result<u64, Response> {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration_since_epoch) => Ok(duration_since_epoch.as_secs()),
        Err(system_time_error) => {
            tracing::error!(error = %system_time_error, "system clock predates the Unix epoch");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OpenAiErrorResponse::service_unavailable(
                    failure_message,
                    Some(failure_code),
                )),
            )
                .into_response())
        }
    }
}

/// Maps a failed generation start to its OpenAI-shaped HTTP rejection.
pub(crate) fn generation_start_rejection_response(
    generation_start_error: GenerationStartError,
) -> Response {
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
            Json(crate::rest::request_rejections::model_load_failed_body(
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

pub(crate) fn worker_unavailable_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(OpenAiErrorResponse::service_unavailable(
            "the local worker is unavailable",
            Some("worker_unavailable"),
        )),
    )
        .into_response()
}

#[cfg(test)]
#[path = "chat_generation_admission_tests.rs"]
mod chat_generation_admission_tests;

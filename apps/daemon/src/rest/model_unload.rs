//! `POST /v1/control/unload-model`: frees the loaded model's memory now,
//! instead of waiting for the idle unload. A tool that needs the Mac's memory
//! for something else, or a person who wants it back, can ask for it.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

use crate::ModelUnloadOutcome;
use crate::application::ApplicationState;

#[derive(Serialize)]
struct ModelUnloadResponse {
    /// `unloaded`, `nothing_loaded`, `busy`, `unavailable` or
    /// `unloaded_engine_restarting`.
    status: &'static str,
    /// The model that was unloaded, when one was.
    model_id: Option<String>,
    message: &'static str,
}

/// Unloads the loaded model when no request is running or waiting.
pub(crate) async fn unload_model(State(application_state): State<ApplicationState>) -> Response {
    let Some(worker_handle) = application_state.worker_control.as_ref() else {
        return (StatusCode::NOT_FOUND, "live worker control is unavailable").into_response();
    };
    match worker_handle.unload_model().await {
        Ok(outcome) => unload_response(outcome),
        Err(worker_control_error) => {
            tracing::error!(error = %worker_control_error, "model unload failed");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "the model could not be unloaded",
            )
                .into_response()
        }
    }
}

pub(crate) fn unload_response(outcome: ModelUnloadOutcome) -> Response {
    let (status_code, body) = match outcome {
        ModelUnloadOutcome::Unloaded { model_id } => (
            StatusCode::OK,
            ModelUnloadResponse {
                status: "unloaded",
                model_id: Some(model_id),
                message: "The model was unloaded and its memory freed.",
            },
        ),
        ModelUnloadOutcome::NothingLoaded => (
            StatusCode::OK,
            ModelUnloadResponse {
                status: "nothing_loaded",
                model_id: None,
                message: "No model was loaded.",
            },
        ),
        ModelUnloadOutcome::Busy => (
            StatusCode::CONFLICT,
            ModelUnloadResponse {
                status: "busy",
                model_id: None,
                message: "A request is running or waiting, so the model stays loaded. Try again when it finishes.",
            },
        ),
        ModelUnloadOutcome::Unavailable => (
            StatusCode::SERVICE_UNAVAILABLE,
            ModelUnloadResponse {
                status: "unavailable",
                model_id: None,
                message: "The engine is not running, so there is nothing to unload.",
            },
        ),
        ModelUnloadOutcome::UnloadedWithoutReadyWorker { model_id } => (
            StatusCode::SERVICE_UNAVAILABLE,
            ModelUnloadResponse {
                status: "unloaded_engine_restarting",
                model_id: Some(model_id),
                message: "The model was unloaded, but the engine is still restarting.",
            },
        ),
    };
    (status_code, Json(body)).into_response()
}

#[cfg(test)]
#[path = "model_unload_tests.rs"]
mod model_unload_tests;

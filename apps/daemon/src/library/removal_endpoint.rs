//! Removes a model that Download Models installed: `POST /v1/library/remove`.
//!
//! The page sends only the catalog identity. The daemon refuses requests from
//! any page but this instance's own Observatory, then asks the download
//! coordinator, which under its command lock refuses while a model from the
//! folder is loaded or loading and moves only its own provenance-marked folder
//! for the identity to the Trash.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, Serialize};

use crate::application::ApplicationState;
use crate::library::download::InstalledModelRemovalError;
use crate::library::model_folder_busy::{
    ModelFolderBusy, ModelFolderBusyCheck, WorkerModelFolderBusyCheck,
};
use crate::library::model_folder_trash::ModelFolderTrashError;
use crate::rest::loopback_browser_guard::refuse_foreign_browser_requests;

pub(crate) fn library_removal_routes() -> Router<ApplicationState> {
    // Moving gigabytes to the Trash must not be something any web page on this
    // Mac can trigger through DNS rebinding or a permissive CORS origin.
    Router::new()
        .route("/v1/library/remove", post(remove_library_model))
        // The body only names a catalog identity, as the download controls'
        // bodies do, so it gets their small cap.
        .layer(DefaultBodyLimit::max(MAXIMUM_REMOVAL_BODY_BYTES))
        .route_layer(axum::middleware::from_fn(refuse_foreign_browser_requests))
}

const MAXIMUM_REMOVAL_BODY_BYTES: usize = 4 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LibraryRemovalRequest {
    huggingface_id: String,
}

#[derive(Serialize)]
struct LibraryRemovalErrorBody {
    code: &'static str,
    message: &'static str,
}

#[derive(Serialize)]
struct LibraryRemovalErrorResponse {
    error: LibraryRemovalErrorBody,
}

async fn remove_library_model(
    State(application_state): State<ApplicationState>,
    Json(removal_request): Json<LibraryRemovalRequest>,
) -> Response {
    let Some(download_coordinator) = application_state.library_download_coordinator.clone() else {
        return removal_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "download_models_unavailable",
            "Model downloads are unavailable in this server mode.",
        );
    };
    let model_folder_busy_check: Arc<dyn ModelFolderBusyCheck> =
        Arc::new(WorkerModelFolderBusyCheck::new(application_state.clone()));
    let model_folder_trash = Arc::clone(&application_state.model_folder_trash);
    let huggingface_id = removal_request.huggingface_id;
    // Spawned so the removal runs to the end even when the page goes away
    // mid-request: dropping the handler between the move and forgetting the
    // publication would leave the catalog listing a model the Trash now holds.
    let removal_task = tokio::spawn(async move {
        download_coordinator
            .remove_installed_model(&huggingface_id, model_folder_busy_check, model_folder_trash)
            .await
    });
    match removal_task.await {
        Ok(Ok(())) => (
            StatusCode::OK,
            Json(serde_json::json!({ "status": "removed" })),
        )
            .into_response(),
        Ok(Err(removal_error)) => removal_failure_response(&removal_error),
        Err(join_error) => {
            tracing::warn!(%join_error, "the model removal task did not finish");
            removal_failure_response(&InstalledModelRemovalError::Trash(
                ModelFolderTrashError::Failed,
            ))
        }
    }
}

fn busy_failure(folder_busy: ModelFolderBusy) -> (StatusCode, &'static str, &'static str) {
    match folder_busy {
        ModelFolderBusy::Loaded => (
            StatusCode::CONFLICT,
            "model_loaded",
            "This model is loaded right now. It unloads on its own after a short idle time; remove it then.",
        ),
        ModelFolderBusy::Loading => (
            StatusCode::CONFLICT,
            "model_loading",
            "A model is loading right now. Try again once it has loaded.",
        ),
    }
}

fn removal_failure_response(removal_error: &InstalledModelRemovalError) -> Response {
    let (status, error_code, message) = match removal_error {
        InstalledModelRemovalError::CatalogEntryNotFound => (
            StatusCode::NOT_FOUND,
            "catalog_entry_not_found",
            "That model is not in this release's catalog.",
        ),
        InstalledModelRemovalError::NotOnThisMac => (
            StatusCode::NOT_FOUND,
            "not_on_this_mac",
            "This model is not in the Models download folder, so there is nothing to remove.",
        ),
        InstalledModelRemovalError::NotInstalledByDownloadModels => (
            StatusCode::CONFLICT,
            "not_installed_by_download_models",
            "This copy was not downloaded here, so Astronomical leaves it where it is.",
        ),
        InstalledModelRemovalError::UnsafeFolder => (
            StatusCode::CONFLICT,
            "unsafe_model_folder",
            "Astronomical could not safely reach this model's folder (a link, a file in its place, or no permission), so it leaves it where it is.",
        ),
        InstalledModelRemovalError::DownloadInProgress => (
            StatusCode::CONFLICT,
            "download_in_progress",
            "This model is still downloading. Cancel the download instead.",
        ),
        InstalledModelRemovalError::ModelFolderBusy(folder_busy) => busy_failure(*folder_busy),
        InstalledModelRemovalError::DownloadState(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "download_state_unreadable",
            "The download state could not be read. Try again.",
        ),
        InstalledModelRemovalError::Trash(trash_error) => trash_failure(*trash_error),
    };
    removal_error_response(status, error_code, message)
}

fn trash_failure(trash_error: ModelFolderTrashError) -> (StatusCode, &'static str, &'static str) {
    match trash_error {
        ModelFolderTrashError::Unavailable => (
            StatusCode::NOT_IMPLEMENTED,
            "trash_unavailable",
            "This version of macOS cannot move models to the Trash from Astronomical. Show the folder in Finder and move it to the Trash there.",
        ),
        ModelFolderTrashError::Failed => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "trash_failed",
            "The folder could not be moved to the Trash. Show it in Finder and move it there.",
        ),
    }
}

fn removal_error_response(
    status: StatusCode,
    error_code: &'static str,
    message: &'static str,
) -> Response {
    (
        status,
        Json(LibraryRemovalErrorResponse {
            error: LibraryRemovalErrorBody {
                code: error_code,
                message,
            },
        }),
    )
        .into_response()
}

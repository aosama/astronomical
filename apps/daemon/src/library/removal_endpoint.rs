//! Removes a model that Download Models installed: `POST /v1/library/remove`.
//!
//! The page sends only the catalog identity. The daemon refuses requests from
//! any page but this instance's own Observatory, refuses while a model in the
//! folder is loaded or loading, then asks the download coordinator, which moves
//! only its own provenance-marked folder for the identity to the Trash.

use std::path::Path;
use std::sync::Arc;

use astronomical_config::DiscoveredModel;
use astronomical_ipc_protocol::ModelLoadInFlight;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, Serialize};

use crate::application::ApplicationState;
use crate::library::download::InstalledModelRemovalError;
use crate::library::model_folder_trash::ModelFolderTrashError;
use crate::library::{LibraryDownloadCoordinator, requestable_model_id_from_huggingface_id};
use crate::rest::loopback_browser_guard::refuse_foreign_browser_requests;

pub(crate) fn library_removal_routes() -> Router<ApplicationState> {
    // Moving gigabytes to the Trash must not be something any web page on this
    // Mac can trigger through DNS rebinding or a permissive CORS origin.
    Router::new()
        .route("/v1/library/remove", post(remove_library_model))
        .route_layer(axum::middleware::from_fn(refuse_foreign_browser_requests))
}

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

/// Why a model's folder cannot move right now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModelFolderBusy {
    Loaded,
    Loading,
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
    if let Some(folder_busy) = folder_busy_now(
        &application_state,
        &download_coordinator,
        &removal_request.huggingface_id,
    ) {
        return busy_response(folder_busy);
    }
    let model_folder_trash = Arc::clone(&application_state.model_folder_trash);
    let huggingface_id = removal_request.huggingface_id;
    // Spawned so the removal runs to the end even when the page goes away
    // mid-request: dropping the handler between the move and forgetting the
    // publication would leave the catalog listing a model the Trash now holds.
    let removal_task = tokio::spawn(async move {
        download_coordinator
            .remove_installed_model(&huggingface_id, model_folder_trash)
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

/// Reads what the worker is serving or loading and checks it against the
/// models discovered inside this catalog model's folder.
fn folder_busy_now(
    application_state: &ApplicationState,
    download_coordinator: &LibraryDownloadCoordinator,
    huggingface_id: &str,
) -> Option<ModelFolderBusy> {
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    let mut model_ids_in_folder = model_ids_served_from_folder(
        &application_state.discovered_models_snapshot(),
        &download_coordinator.destination_directory(huggingface_id),
    );
    // A freshly published model is requestable under this id before the next
    // discovery names it, so it counts as served from the folder too.
    model_ids_in_folder.push(requestable_model_id_from_huggingface_id(huggingface_id));
    model_folder_busy_reason(
        &model_ids_in_folder,
        worker_health_snapshot.ready_model_id.as_deref(),
        worker_health_snapshot.model_load_in_flight.as_ref(),
    )
}

/// The model ids discovery serves from files inside `model_folder`. Matching by
/// folder rather than by catalog identity also catches a model whose files are
/// in this folder but whose served name differs from the catalog's.
pub(crate) fn model_ids_served_from_folder(
    discovered_models: &[DiscoveredModel],
    model_folder: &Path,
) -> Vec<String> {
    discovered_models
        .iter()
        .filter(|discovered_model| discovered_model.model_directory.starts_with(model_folder))
        .map(|discovered_model| discovered_model.model_id.clone())
        .collect()
}

/// A load that cannot name its model may be loading this one, so it refuses
/// too; the person retries a moment later instead of moving files mid-load.
pub(crate) fn model_folder_busy_reason(
    model_ids_in_folder: &[String],
    ready_model_id: Option<&str>,
    model_load_in_flight: Option<&ModelLoadInFlight>,
) -> Option<ModelFolderBusy> {
    let is_in_folder = |model_id: &str| {
        model_ids_in_folder
            .iter()
            .any(|folder_model_id| folder_model_id == model_id)
    };
    if let Some(load_in_flight) = model_load_in_flight {
        match load_in_flight.model_id.as_deref() {
            None => return Some(ModelFolderBusy::Loading),
            Some(loading_model_id) if is_in_folder(loading_model_id) => {
                return Some(ModelFolderBusy::Loading);
            }
            Some(_) => {}
        }
    }
    ready_model_id
        .filter(|model_id| is_in_folder(model_id))
        .map(|_| ModelFolderBusy::Loaded)
}

fn busy_response(folder_busy: ModelFolderBusy) -> Response {
    match folder_busy {
        ModelFolderBusy::Loaded => removal_error_response(
            StatusCode::CONFLICT,
            "model_loaded",
            "This model is loaded right now. It unloads on its own after a short idle time; remove it then.",
        ),
        ModelFolderBusy::Loading => removal_error_response(
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
            "This model is not in the Download Models folder, so there is nothing to remove.",
        ),
        InstalledModelRemovalError::NotInstalledByDownloadModels => (
            StatusCode::CONFLICT,
            "not_installed_by_download_models",
            "This copy was not downloaded here, so Astronomical leaves it where it is.",
        ),
        InstalledModelRemovalError::UnsafeFolder => (
            StatusCode::CONFLICT,
            "unsafe_model_folder",
            "This model's folder is reached through a link, so Astronomical leaves it where it is.",
        ),
        InstalledModelRemovalError::DownloadInProgress => (
            StatusCode::CONFLICT,
            "download_in_progress",
            "This model is still downloading. Cancel the download instead.",
        ),
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

#[cfg(test)]
#[path = "removal_endpoint_tests.rs"]
mod removal_endpoint_tests;

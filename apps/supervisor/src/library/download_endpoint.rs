//! Local REST controls and bounded public projection for one Library download.

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use crate::application::ApplicationState;

use super::{DownloadJob, DownloadJobPublicErrorCode, LibraryDownloadCoordinatorError};

const MAXIMUM_DOWNLOAD_CONTROL_BODY_BYTES: usize = 4_096;

pub(crate) fn library_download_routes() -> Router<ApplicationState> {
    Router::new()
        .route(
            "/v1/library/download",
            get(get_download).post(start_download),
        )
        .route("/v1/library/download/pause", post(pause_download))
        .route("/v1/library/download/resume", post(resume_download))
        .route("/v1/library/download/cancel", post(cancel_download))
        .layer(DefaultBodyLimit::max(MAXIMUM_DOWNLOAD_CONTROL_BODY_BYTES))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StartDownloadRequest {
    huggingface_id: String,
}

#[derive(Debug, Serialize)]
struct DownloadResponse {
    state: &'static str,
    huggingface_id: Option<String>,
    revision: Option<String>,
    bytes_completed: u64,
    bytes_total: u64,
    current_file_relative_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    destination_directory: Option<String>,
    error_code: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct DownloadErrorResponse {
    error: DownloadErrorBody,
}

#[derive(Debug, Serialize)]
struct DownloadErrorBody {
    code: &'static str,
    message: &'static str,
}

async fn get_download(State(application_state): State<ApplicationState>) -> Response {
    let Some(download_coordinator) = application_state.library_download_coordinator.as_ref() else {
        return Json(DownloadResponse::idle()).into_response();
    };
    match download_coordinator.current_job().await {
        Ok(Some(download_job)) => {
            let destination_directory = download_coordinator
                .destination_directory(download_job.huggingface_id())
                .display()
                .to_string();
            Json(DownloadResponse::from_job(
                &download_job,
                Some(destination_directory),
            ))
            .into_response()
        }
        Ok(None) => Json(DownloadResponse::idle()).into_response(),
        Err(download_error) => coordinator_error_response(download_error),
    }
}

async fn start_download(
    State(application_state): State<ApplicationState>,
    Json(start_request): Json<StartDownloadRequest>,
) -> Response {
    let Some(download_coordinator) = application_state.library_download_coordinator.as_ref() else {
        return unavailable_response();
    };
    match download_coordinator
        .start(&start_request.huggingface_id)
        .await
    {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(DownloadResponse::starting(&start_request.huggingface_id)),
        )
            .into_response(),
        Err(download_error) => coordinator_error_response(download_error),
    }
}

async fn pause_download(State(application_state): State<ApplicationState>) -> Response {
    let Some(download_coordinator) = application_state.library_download_coordinator.as_ref() else {
        return unavailable_response();
    };
    match download_coordinator.pause().await {
        Ok(download_job) => {
            let destination_directory = download_coordinator
                .destination_directory(download_job.huggingface_id())
                .display()
                .to_string();
            Json(DownloadResponse::from_job(
                &download_job,
                Some(destination_directory),
            ))
            .into_response()
        }
        Err(download_error) => coordinator_error_response(download_error),
    }
}

async fn resume_download(State(application_state): State<ApplicationState>) -> Response {
    let Some(download_coordinator) = application_state.library_download_coordinator.as_ref() else {
        return unavailable_response();
    };
    match download_coordinator.resume().await {
        Ok(()) => (StatusCode::ACCEPTED, Json(DownloadResponse::resuming())).into_response(),
        Err(download_error) => coordinator_error_response(download_error),
    }
}

async fn cancel_download(State(application_state): State<ApplicationState>) -> Response {
    let Some(download_coordinator) = application_state.library_download_coordinator.as_ref() else {
        return unavailable_response();
    };
    match download_coordinator.cancel().await {
        Ok(_) => Json(DownloadResponse::idle()).into_response(),
        Err(download_error) => coordinator_error_response(download_error),
    }
}

impl DownloadResponse {
    const fn idle() -> Self {
        Self {
            state: "idle",
            huggingface_id: None,
            revision: None,
            bytes_completed: 0,
            bytes_total: 0,
            current_file_relative_path: None,
            destination_directory: None,
            error_code: None,
        }
    }

    fn starting(huggingface_id: &str) -> Self {
        Self {
            state: "checking_disk",
            huggingface_id: Some(huggingface_id.to_owned()),
            ..Self::idle()
        }
    }

    fn resuming() -> Self {
        Self {
            state: "resuming",
            ..Self::idle()
        }
    }

    fn from_job(download_job: &DownloadJob, destination_directory: Option<String>) -> Self {
        Self {
            state: download_job.state().as_str(),
            huggingface_id: Some(download_job.huggingface_id().to_owned()),
            revision: Some(download_job.revision().to_owned()),
            bytes_completed: download_job.bytes_completed(),
            bytes_total: download_job.bytes_total(),
            current_file_relative_path: download_job
                .current_file_relative_path()
                .map(str::to_owned),
            destination_directory,
            error_code: download_job
                .error_code()
                .map(DownloadJobPublicErrorCode::as_str),
        }
    }
}

fn coordinator_error_response(download_error: LibraryDownloadCoordinatorError) -> Response {
    let (status, error_code, message) = match download_error {
        LibraryDownloadCoordinatorError::LibraryBusy => (
            StatusCode::CONFLICT,
            "library_busy",
            "Another model download is already active.",
        ),
        LibraryDownloadCoordinatorError::CatalogEntryNotFound => (
            StatusCode::NOT_FOUND,
            "catalog_entry_not_found",
            "That model is not available in this release catalog.",
        ),
        LibraryDownloadCoordinatorError::JobNotFound => (
            StatusCode::NOT_FOUND,
            "download_failed",
            "There is no model download to control.",
        ),
        LibraryDownloadCoordinatorError::JobStore(_)
        | LibraryDownloadCoordinatorError::BackgroundTask(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "download_failed",
            "The model download could not be updated. Retry from Library.",
        ),
    };
    (
        status,
        Json(DownloadErrorResponse {
            error: DownloadErrorBody {
                code: error_code,
                message,
            },
        }),
    )
        .into_response()
}

fn unavailable_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(DownloadErrorResponse {
            error: DownloadErrorBody {
                code: "download_failed",
                message: "Model downloads are unavailable in this server mode.",
            },
        }),
    )
        .into_response()
}

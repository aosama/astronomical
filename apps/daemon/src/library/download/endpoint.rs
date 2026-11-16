//! Local REST controls and bounded public projection for the Library
//! download and the list of models waiting after it.

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use crate::application::ApplicationState;

use crate::library::{
    DownloadEnqueueOutcome, DownloadJob, DownloadJobPublicErrorCode, LibraryDownloadCoordinator,
    LibraryDownloadCoordinatorError,
};

const BYTES_PER_GIGABYTE: f64 = 1_073_741_824.0;

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
        .route("/v1/library/download/queue/remove", post(remove_from_queue))
        .layer(DefaultBodyLimit::max(MAXIMUM_DOWNLOAD_CONTROL_BODY_BYTES))
        // Reading, starting, pausing or cancelling a multi-gigabyte download
        // answers only this instance's own page, the Thin Talk canvas (Astro
        // drives downloads from it) and tools that send no Origin.
        .route_layer(axum::middleware::from_fn(
            crate::rest::loopback_browser_guard::refuse_foreign_browser_requests_except_thin_talk,
        ))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
/// Names one catalog model, for asking for it or taking it off the list.
struct ModelRequest {
    huggingface_id: String,
}

/// The models waiting after the current download, and the listed ones that
/// could not finish.
#[derive(Debug, Default, Serialize)]
struct DownloadQueueResponse {
    waiting: Vec<WaitingModelResponse>,
    failed: Vec<FailedModelResponse>,
}

#[derive(Debug, Serialize)]
struct WaitingModelResponse {
    huggingface_id: String,
    approximate_size_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct FailedModelResponse {
    huggingface_id: String,
    error_code: &'static str,
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
    /// For a model just asked for: its 1-based place among the waiting
    /// models, when it did not start at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    queue_position: Option<usize>,
    /// The download list, on every status reading.
    #[serde(skip_serializing_if = "Option::is_none")]
    queue: Option<DownloadQueueResponse>,
}

#[derive(Debug, Serialize)]
struct DownloadErrorResponse {
    error: DownloadErrorBody,
}

#[derive(Debug, Serialize)]
struct DownloadErrorBody {
    code: &'static str,
    message: String,
}

async fn get_download(State(application_state): State<ApplicationState>) -> Response {
    let Some(download_coordinator) = application_state.library_download_coordinator.as_ref() else {
        return Json(DownloadResponse::idle()).into_response();
    };
    let current_download = match download_coordinator.current_job().await {
        Ok(Some(download_job)) => {
            let destination_directory = download_coordinator
                .destination_directory(download_job.huggingface_id())
                .display()
                .to_string();
            DownloadResponse::from_job(&download_job, Some(destination_directory))
        }
        Ok(None) => DownloadResponse::idle(),
        Err(download_error) => return coordinator_error_response(download_error),
    };
    Json(DownloadResponse {
        queue: Some(queue_response(download_coordinator)),
        ..current_download
    })
    .into_response()
}

async fn start_download(
    State(application_state): State<ApplicationState>,
    Json(start_request): Json<ModelRequest>,
) -> Response {
    let Some(download_coordinator) = application_state.library_download_coordinator.as_ref() else {
        return unavailable_response();
    };
    let huggingface_id = start_request.huggingface_id.as_str();
    match download_coordinator.enqueue(huggingface_id).await {
        Ok(enqueue_outcome) => enqueue_outcome_response(enqueue_outcome, huggingface_id),
        Err(download_error) => coordinator_error_response(download_error),
    }
}

/// The answer to asking for a model: starting now, already the current
/// download, or waiting at a place on the list.
fn enqueue_outcome_response(
    enqueue_outcome: DownloadEnqueueOutcome,
    huggingface_id: &str,
) -> Response {
    match enqueue_outcome {
        DownloadEnqueueOutcome::Started => (
            StatusCode::ACCEPTED,
            Json(DownloadResponse::starting(huggingface_id)),
        )
            .into_response(),
        DownloadEnqueueOutcome::AlreadyCurrent => {
            Json(DownloadResponse::named("current", huggingface_id, None)).into_response()
        }
        DownloadEnqueueOutcome::Queued { waiting_position } => (
            StatusCode::ACCEPTED,
            Json(DownloadResponse::named(
                "queued",
                huggingface_id,
                Some(waiting_position),
            )),
        )
            .into_response(),
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

async fn remove_from_queue(
    State(application_state): State<ApplicationState>,
    Json(remove_request): Json<ModelRequest>,
) -> Response {
    let Some(download_coordinator) = application_state.library_download_coordinator.as_ref() else {
        return unavailable_response();
    };
    match download_coordinator
        .remove_from_queue(&remove_request.huggingface_id)
        .await
    {
        Ok(true) => Json(DownloadResponse {
            queue: Some(queue_response(download_coordinator)),
            ..DownloadResponse::idle()
        })
        .into_response(),
        Ok(false) => error_response(
            StatusCode::NOT_FOUND,
            "not_in_queue",
            "That model is not on the download list.".to_owned(),
        ),
        Err(download_error) => coordinator_error_response(download_error),
    }
}

fn queue_response(download_coordinator: &LibraryDownloadCoordinator) -> DownloadQueueResponse {
    let download_queue = download_coordinator.queue_snapshot();
    DownloadQueueResponse {
        waiting: download_queue
            .waiting_huggingface_ids()
            .iter()
            .map(|waiting_id| WaitingModelResponse {
                huggingface_id: waiting_id.clone(),
                approximate_size_bytes: download_coordinator.catalog_size_bytes(waiting_id),
            })
            .collect(),
        failed: download_queue
            .failed_entries()
            .iter()
            .map(|failed_entry| FailedModelResponse {
                huggingface_id: failed_entry.huggingface_id.clone(),
                error_code: failed_entry.error_code.as_str(),
            })
            .collect(),
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
            queue_position: None,
            queue: None,
        }
    }

    fn named(state: &'static str, huggingface_id: &str, queue_position: Option<usize>) -> Self {
        Self {
            state,
            huggingface_id: Some(huggingface_id.to_owned()),
            queue_position,
            ..Self::idle()
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
            queue_position: None,
            queue: None,
        }
    }
}

fn coordinator_error_response(download_error: LibraryDownloadCoordinatorError) -> Response {
    let (status, error_code, message) = coordinator_error_parts(download_error);
    error_response(status, error_code, message)
}

/// The status, stable code and plain sentence for each coordinator failure.
fn coordinator_error_parts(
    download_error: LibraryDownloadCoordinatorError,
) -> (StatusCode, &'static str, String) {
    match download_error {
        LibraryDownloadCoordinatorError::LibraryBusy => (
            StatusCode::CONFLICT,
            "library_busy",
            "Another model download is already active.".to_owned(),
        ),
        LibraryDownloadCoordinatorError::CatalogEntryNotFound => (
            StatusCode::NOT_FOUND,
            "catalog_entry_not_found",
            "That model is not available in this release catalog.".to_owned(),
        ),
        LibraryDownloadCoordinatorError::JobNotFound => (
            StatusCode::NOT_FOUND,
            "download_failed",
            "There is no model download to control.".to_owned(),
        ),
        LibraryDownloadCoordinatorError::QueueDoesNotFit(queue_shortfall) => (
            StatusCode::CONFLICT,
            "insufficient_disk",
            insufficient_disk_message(
                queue_shortfall.required_bytes,
                queue_shortfall.available_bytes,
            ),
        ),
        LibraryDownloadCoordinatorError::JobStore(_)
        | LibraryDownloadCoordinatorError::BackgroundTask(_)
        | LibraryDownloadCoordinatorError::QueueStore(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "download_failed",
            "The model download could not be updated. Retry from the Models page.".to_owned(),
        ),
    }
}

/// Sizes in the gigabytes the Mac's own screens use (2^30 bytes), rounded
/// up for the need and down for the free space so the sentence never claims
/// more room than there is.
fn insufficient_disk_message(required_bytes: u64, available_bytes: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let (required_gigabytes, available_gigabytes) = (
        (required_bytes as f64 / BYTES_PER_GIGABYTE).ceil(),
        (available_bytes as f64 / BYTES_PER_GIGABYTE).floor(),
    );
    format!(
        "Not enough room on the disk: this model and the ones already listed need {required_gigabytes} GB, and {available_gigabytes} GB is free."
    )
}

fn error_response(status: StatusCode, error_code: &'static str, message: String) -> Response {
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
                message: "Model downloads are unavailable in this server mode.".to_owned(),
            },
        }),
    )
        .into_response()
}

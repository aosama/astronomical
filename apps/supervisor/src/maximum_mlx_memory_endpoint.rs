use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::application::ApplicationState;
use crate::{ChatGenerationExecutor, MlxMemoryLimitUpdateOutcome, WorkerHealthSnapshot};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MaximumMlxMemoryRequest {
    maximum_mlx_memory_gb: Option<u64>,
}

#[derive(Serialize)]
struct MaximumMlxMemoryResponse {
    configured_maximum_mlx_memory_gb: Option<u64>,
    effective_mlx_memory_ceiling_bytes: u64,
    machine_mlx_memory_ceiling_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
    pending_mlx_memory_ceiling_bytes: Option<u64>,
    message: String,
}

pub(crate) async fn update_maximum_mlx_memory(
    State(application_state): State<ApplicationState>,
    Json(maximum_mlx_memory_request): Json<MaximumMlxMemoryRequest>,
) -> Response {
    let Some(worker_handle) = application_state.worker_control.as_ref() else {
        return (StatusCode::NOT_FOUND, "live worker control is unavailable").into_response();
    };
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.as_ref() else {
        return (StatusCode::NOT_FOUND, "config persistence is unavailable").into_response();
    };
    let _config_mutation_guard = application_state.config_mutation_lock.lock().await;
    let worker_health_snapshot = worker_handle.worker_health_snapshot();
    let requested_mlx_memory_ceiling_bytes = match maximum_mlx_memory_request.maximum_mlx_memory_gb
    {
        Some(maximum_mlx_memory_gb) => {
            match astronomical_config::maximum_mlx_memory_gb_to_bytes(maximum_mlx_memory_gb) {
                Ok(requested_mlx_memory_ceiling_bytes) => requested_mlx_memory_ceiling_bytes,
                Err(configuration_error) => {
                    return invalid_request_response(
                        maximum_mlx_memory_request.maximum_mlx_memory_gb,
                        &worker_health_snapshot,
                        configuration_error.to_string(),
                    );
                }
            }
        }
        None => worker_health_snapshot.machine_mlx_memory_ceiling_bytes,
    };
    if worker_health_snapshot.machine_mlx_memory_ceiling_bytes == 0
        || requested_mlx_memory_ceiling_bytes
            > worker_health_snapshot.machine_mlx_memory_ceiling_bytes
        || requested_mlx_memory_ceiling_bytes
            < worker_health_snapshot.minimum_mlx_memory_ceiling_bytes
    {
        return invalid_request_response(
            maximum_mlx_memory_request.maximum_mlx_memory_gb,
            &worker_health_snapshot,
            "requested MLX memory ceiling is outside the worker's reported bounds".to_owned(),
        );
    }

    let prior_config_bytes = match astronomical_config::write_maximum_mlx_memory_gb(
        runtime_config_resolver.state_directory(),
        maximum_mlx_memory_request.maximum_mlx_memory_gb,
    ) {
        Ok(prior_config_bytes) => prior_config_bytes,
        Err(configuration_error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(response_document(
                    maximum_mlx_memory_request.maximum_mlx_memory_gb,
                    &worker_health_snapshot,
                    format!("could not persist the MLX memory setting: {configuration_error}"),
                )),
            )
                .into_response();
        }
    };
    match worker_handle
        .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
        .await
    {
        Ok(MlxMemoryLimitUpdateOutcome::Applied) | Ok(MlxMemoryLimitUpdateOutcome::Queued) => {
            if let Some(reloadable_config) = application_state.reloadable_config.as_ref()
                && let Ok(mut live_config) = reloadable_config.write()
            {
                live_config.maximum_mlx_memory_bytes = maximum_mlx_memory_request
                    .maximum_mlx_memory_gb
                    .map(|_| requested_mlx_memory_ceiling_bytes);
            }
            let updated_worker_health_snapshot = worker_handle.worker_health_snapshot();
            let is_queued = updated_worker_health_snapshot
                .pending_mlx_memory_ceiling_bytes
                .is_some();
            let status_code = if is_queued {
                StatusCode::ACCEPTED
            } else {
                StatusCode::OK
            };
            let message = if is_queued {
                "MLX memory setting persisted and queued until generation finalizes"
            } else {
                "MLX memory setting persisted and applied"
            };
            (
                status_code,
                Json(response_document(
                    maximum_mlx_memory_request.maximum_mlx_memory_gb,
                    &updated_worker_health_snapshot,
                    message.to_owned(),
                )),
            )
                .into_response()
        }
        Ok(MlxMemoryLimitUpdateOutcome::Rejected) => {
            if let Err(restore_error) = astronomical_config::restore_config_file(
                runtime_config_resolver.state_directory(),
                prior_config_bytes.as_deref(),
            ) {
                tracing::error!(error = %restore_error, "could not restore config after worker rejected MLX memory limit");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(response_document(
                        maximum_mlx_memory_request.maximum_mlx_memory_gb,
                        &worker_handle.worker_health_snapshot(),
                        format!("worker rejected the MLX memory limit and config restoration failed: {restore_error}"),
                    )),
                )
                    .into_response();
            }
            let updated_worker_health_snapshot = worker_handle.worker_health_snapshot();
            invalid_request_response(
                maximum_mlx_memory_request.maximum_mlx_memory_gb,
                &updated_worker_health_snapshot,
                updated_worker_health_snapshot
                    .mlx_memory_limit_error
                    .clone()
                    .unwrap_or_else(|| {
                        "worker rejected the requested MLX memory ceiling".to_owned()
                    }),
            )
        }
        Err(worker_control_error) => {
            if let Err(restore_error) = astronomical_config::restore_config_file(
                runtime_config_resolver.state_directory(),
                prior_config_bytes.as_deref(),
            ) {
                tracing::error!(error = %restore_error, "could not restore config after MLX memory worker-control failure");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(response_document(
                        maximum_mlx_memory_request.maximum_mlx_memory_gb,
                        &worker_handle.worker_health_snapshot(),
                        format!("worker control and config restoration failed: {restore_error}"),
                    )),
                )
                    .into_response();
            }
            tracing::error!(error = %worker_control_error, "live MLX memory update failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(response_document(
                    maximum_mlx_memory_request.maximum_mlx_memory_gb,
                    &worker_handle.worker_health_snapshot(),
                    "worker control failed".to_owned(),
                )),
            )
                .into_response()
        }
    }
}

fn invalid_request_response(
    configured_maximum_mlx_memory_gb: Option<u64>,
    worker_health_snapshot: &WorkerHealthSnapshot,
    message: String,
) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(response_document(
            configured_maximum_mlx_memory_gb,
            worker_health_snapshot,
            message,
        )),
    )
        .into_response()
}

fn response_document(
    configured_maximum_mlx_memory_gb: Option<u64>,
    worker_health_snapshot: &WorkerHealthSnapshot,
    message: String,
) -> MaximumMlxMemoryResponse {
    MaximumMlxMemoryResponse {
        configured_maximum_mlx_memory_gb,
        effective_mlx_memory_ceiling_bytes: worker_health_snapshot.mlx_memory_ceiling_bytes,
        machine_mlx_memory_ceiling_bytes: worker_health_snapshot.machine_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes: worker_health_snapshot.minimum_mlx_memory_ceiling_bytes,
        pending_mlx_memory_ceiling_bytes: worker_health_snapshot.pending_mlx_memory_ceiling_bytes,
        message,
    }
}

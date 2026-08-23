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
    let _configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    let mut pending_memory_config_generation = application_state
        .pending_memory_config_generation
        .lock()
        .await;
    if pending_memory_config_generation.is_some() {
        return (
            StatusCode::CONFLICT,
            Json(response_document(
                maximum_mlx_memory_request.maximum_mlx_memory_gb,
                &worker_handle.worker_health_snapshot(),
                "a queued MLX memory setting is still awaiting worker acknowledgement".to_owned(),
            )),
        )
            .into_response();
    }
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

    let prior_resolved_config = application_state
        .reloadable_config
        .as_ref()
        .and_then(|snapshot| {
            snapshot
                .read()
                .ok()
                .map(|resolved_config| resolved_config.clone())
        });
    let config_update = match astronomical_config::prepare_maximum_mlx_memory_gb_update(
        runtime_config_resolver.state_directory(),
        maximum_mlx_memory_request.maximum_mlx_memory_gb,
    ) {
        Ok(config_update) => config_update,
        Err(configuration_error) => {
            tracing::error!(error = %configuration_error, "could not prepare MLX memory configuration");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(response_document(
                    maximum_mlx_memory_request.maximum_mlx_memory_gb,
                    &worker_health_snapshot,
                    "could not prepare the MLX memory setting; inspect local diagnostics"
                        .to_owned(),
                )),
            )
                .into_response();
        }
    };
    let candidate_config = astronomical_config::AstronomicalConfig::load_v1_bytes(
        runtime_config_resolver.instance_paths().clone(),
        &config_update.candidate_config_bytes,
    );
    let candidate_resolved_config = match candidate_config
        .map_err(crate::ResolvedRuntimeConfigError::from)
        .and_then(|candidate_config| runtime_config_resolver.resolve(&candidate_config))
    {
        Ok(candidate_resolved_config) => candidate_resolved_config,
        Err(configuration_error) => {
            tracing::error!(error = %configuration_error, "prepared MLX memory configuration could not be resolved");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(response_document(
                    maximum_mlx_memory_request.maximum_mlx_memory_gb,
                    &worker_health_snapshot,
                    "prepared memory setting could not be resolved; inspect local diagnostics"
                        .to_owned(),
                )),
            )
                .into_response();
        }
    };
    let Some(prior_resolved_config_for_scope) = prior_resolved_config.as_ref() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(response_document(
                maximum_mlx_memory_request.maximum_mlx_memory_gb,
                &worker_health_snapshot,
                "live configuration state is unavailable".to_owned(),
            )),
        )
            .into_response();
    };
    if !memory_is_the_only_candidate_change(
        prior_resolved_config_for_scope,
        &candidate_resolved_config,
    ) {
        return (
            StatusCode::CONFLICT,
            Json(response_document(
                maximum_mlx_memory_request.maximum_mlx_memory_gb,
                &worker_health_snapshot,
                "other configuration changes are pending; reload the complete configuration first"
                    .to_owned(),
            )),
        )
            .into_response();
    }
    if let Err(commit_error) = astronomical_config::commit_maximum_mlx_memory_gb_update(
        runtime_config_resolver.state_directory(),
        &config_update,
    ) {
        let status_code = if matches!(
            commit_error,
            astronomical_config::AstronomicalConfigError::ConfigChangedDuringUpdate
        ) {
            StatusCode::CONFLICT
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        tracing::warn!(error = %commit_error, "MLX memory configuration commit was not accepted");
        return (
            status_code,
            Json(response_document(
                maximum_mlx_memory_request.maximum_mlx_memory_gb,
                &worker_health_snapshot,
                "configuration changed during the memory update; retry after reloading".to_owned(),
            )),
        )
            .into_response();
    }
    let astronomical_config::MaximumMlxMemoryConfigUpdate {
        prior_config_bytes: _,
        candidate_config_bytes,
    } = config_update;
    *pending_memory_config_generation =
        Some(candidate_resolved_config.configuration_generation.clone());
    worker_handle.stage_memory_configuration_generation(
        candidate_resolved_config.configuration_generation.clone(),
    );
    match worker_handle
        .update_mlx_memory_limit(
            requested_mlx_memory_ceiling_bytes,
            candidate_resolved_config.configuration_generation.clone(),
        )
        .await
    {
        Ok(
            update_outcome @ (MlxMemoryLimitUpdateOutcome::Applied
            | MlxMemoryLimitUpdateOutcome::Queued),
        ) => {
            worker_handle.record_memory_configuration_generation(
                candidate_resolved_config.configuration_generation.clone(),
                update_outcome,
            );
            crate::maximum_mlx_memory_transaction::commit_applied_config_snapshots(
                &application_state,
                runtime_config_resolver,
                &candidate_resolved_config,
                &candidate_config_bytes,
            );
            if update_outcome == MlxMemoryLimitUpdateOutcome::Queued {
                tokio::spawn(
                    crate::maximum_mlx_memory_transaction::reconcile_queued_memory_config(
                        application_state.clone(),
                        candidate_resolved_config,
                        candidate_config_bytes,
                        prior_resolved_config,
                    ),
                );
            } else {
                *pending_memory_config_generation = None;
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
            *pending_memory_config_generation = None;
            worker_handle.record_memory_configuration_generation(
                candidate_resolved_config.configuration_generation.clone(),
                MlxMemoryLimitUpdateOutcome::Rejected,
            );
            crate::maximum_mlx_memory_transaction::retain_rejected_persisted_config(
                &application_state,
                runtime_config_resolver,
            );
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
            *pending_memory_config_generation = None;
            worker_handle.record_memory_configuration_generation(
                candidate_resolved_config.configuration_generation.clone(),
                MlxMemoryLimitUpdateOutcome::Rejected,
            );
            crate::maximum_mlx_memory_transaction::retain_rejected_persisted_config(
                &application_state,
                runtime_config_resolver,
            );
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

fn memory_is_the_only_candidate_change(
    prior_resolved_config: &crate::ResolvedRuntimeConfig,
    candidate_resolved_config: &crate::ResolvedRuntimeConfig,
) -> bool {
    matches!(
        crate::ConfigReloadDiff::compare(prior_resolved_config, candidate_resolved_config),
        crate::ConfigReloadDecision::NoWorkerRestart { reloaded_fields, .. }
            if reloaded_fields.iter().all(|field_name| field_name == "maximum_mlx_memory_gb")
    )
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

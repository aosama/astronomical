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
    let worker_health_snapshot = worker_handle.worker_health_snapshot();
    let requested_mlx_memory_ceiling_bytes = match requested_ceiling_bytes(
        maximum_mlx_memory_request.maximum_mlx_memory_gb,
        &worker_health_snapshot,
    ) {
        Ok(requested_mlx_memory_ceiling_bytes) => requested_mlx_memory_ceiling_bytes,
        Err(ceiling_response) => return *ceiling_response,
    };
    let (candidate_resolved_config, prior_resolved_config, config_update) =
        match prepare_and_resolve_candidate(
            &application_state,
            runtime_config_resolver,
            maximum_mlx_memory_request.maximum_mlx_memory_gb,
            &worker_health_snapshot,
        ) {
            Ok(prepared_candidate) => prepared_candidate,
            Err(preparation_response) => return *preparation_response,
        };
    if let Err(commit_response) = commit_candidate_if_memory_only(
        prior_resolved_config.as_ref(),
        &candidate_resolved_config,
        runtime_config_resolver,
        &config_update,
        maximum_mlx_memory_request.maximum_mlx_memory_gb,
        &worker_health_snapshot,
    ) {
        return *commit_response;
    }
    let config::MaximumMlxMemoryConfigUpdate {
        prior_config_bytes: _,
        candidate_config_bytes,
    } = config_update;
    worker_handle.stage_memory_configuration_generation(
        candidate_resolved_config.configuration_generation.clone(),
    );
    apply_worker_memory_update(
        &application_state,
        worker_handle,
        runtime_config_resolver,
        &candidate_resolved_config,
        candidate_config_bytes,
        requested_mlx_memory_ceiling_bytes,
        maximum_mlx_memory_request.maximum_mlx_memory_gb,
    )
    .await
}

/// The requested ceiling in bytes: the configured gigabytes converted, or the
/// machine ceiling when the request clears the setting. Rejects conversion
/// failures and ceilings outside the worker's reported bounds.
fn requested_ceiling_bytes(
    maximum_mlx_memory_gb: Option<u64>,
    worker_health_snapshot: &WorkerHealthSnapshot,
) -> Result<u64, Box<Response>> {
    let requested_mlx_memory_ceiling_bytes = match maximum_mlx_memory_gb {
        Some(configured_maximum_mlx_memory_gb) => {
            match config::maximum_mlx_memory_gb_to_bytes(configured_maximum_mlx_memory_gb) {
                Ok(requested_mlx_memory_ceiling_bytes) => requested_mlx_memory_ceiling_bytes,
                Err(configuration_error) => {
                    return Err(Box::new(invalid_request_response(
                        maximum_mlx_memory_gb,
                        worker_health_snapshot,
                        configuration_error.to_string(),
                    )));
                }
            }
        }
        None => worker_health_snapshot.machine_mlx_memory_ceiling_bytes,
    };
    if worker_health_snapshot.machine_mlx_memory_ceiling_bytes == 0
        || requested_mlx_memory_ceiling_bytes
            > worker_health_snapshot.machine_mlx_memory_ceiling_bytes
    {
        return Err(Box::new(invalid_request_response(
            maximum_mlx_memory_gb,
            worker_health_snapshot,
            "requested MLX memory ceiling is outside the worker's reported bounds".to_owned(),
        )));
    }
    Ok(requested_mlx_memory_ceiling_bytes)
}

/// Prepares and resolves the candidate configuration carrying the requested
/// memory setting: snapshots the prior live config, prepares the on-disk
/// update bytes, and resolves them. Mirrors the inlined preparation block.
fn prepare_and_resolve_candidate(
    application_state: &ApplicationState,
    runtime_config_resolver: &crate::config::config_reload::ResolvedRuntimeConfigResolver,
    maximum_mlx_memory_gb: Option<u64>,
    worker_health_snapshot: &WorkerHealthSnapshot,
) -> Result<
    (
        crate::ResolvedRuntimeConfig,
        Option<crate::ResolvedRuntimeConfig>,
        config::MaximumMlxMemoryConfigUpdate,
    ),
    Box<Response>,
> {
    let prior_resolved_config = application_state
        .reloadable_config
        .as_ref()
        .and_then(|snapshot| {
            snapshot
                .read()
                .ok()
                .map(|resolved_config| resolved_config.clone())
        });
    let config_update = match config::prepare_maximum_mlx_memory_gb_update(
        runtime_config_resolver.state_directory(),
        maximum_mlx_memory_gb,
    ) {
        Ok(config_update) => config_update,
        Err(configuration_error) => {
            tracing::error!(error = %configuration_error, "could not prepare MLX memory configuration");
            return Err(Box::new(
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(response_document(
                        maximum_mlx_memory_gb,
                        worker_health_snapshot,
                        "could not prepare the MLX memory setting; inspect local diagnostics"
                            .to_owned(),
                    )),
                )
                    .into_response(),
            ));
        }
    };
    let candidate_config = config::AstronomicalConfig::load_v1_bytes(
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
            return Err(Box::new(
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(response_document(
                        maximum_mlx_memory_gb,
                        worker_health_snapshot,
                        "prepared memory setting could not be resolved; inspect local diagnostics"
                            .to_owned(),
                    )),
                )
                    .into_response(),
            ));
        }
    };
    Ok((
        candidate_resolved_config,
        prior_resolved_config,
        config_update,
    ))
}

/// Commits the prepared candidate only when memory is its sole change,
/// rejecting with a conflict otherwise. Mirrors the inlined scope check and
/// commit block, including the changed-during-update conflict status.
fn commit_candidate_if_memory_only(
    prior_resolved_config: Option<&crate::ResolvedRuntimeConfig>,
    candidate_resolved_config: &crate::ResolvedRuntimeConfig,
    runtime_config_resolver: &crate::config::config_reload::ResolvedRuntimeConfigResolver,
    config_update: &config::MaximumMlxMemoryConfigUpdate,
    maximum_mlx_memory_gb: Option<u64>,
    worker_health_snapshot: &WorkerHealthSnapshot,
) -> Result<(), Box<Response>> {
    let Some(prior_resolved_config_for_scope) = prior_resolved_config else {
        return Err(Box::new(
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(response_document(
                    maximum_mlx_memory_gb,
                    worker_health_snapshot,
                    "live configuration state is unavailable".to_owned(),
                )),
            )
                .into_response(),
        ));
    };
    if !memory_is_the_only_candidate_change(
        prior_resolved_config_for_scope,
        candidate_resolved_config,
    ) {
        return Err(Box::new((
            StatusCode::CONFLICT,
            Json(response_document(
                maximum_mlx_memory_gb,
                worker_health_snapshot,
                "other configuration changes are pending; reload the complete configuration first"
                    .to_owned(),
            )),
        )
            .into_response()));
    }
    if let Err(commit_error) = config::commit_maximum_mlx_memory_gb_update(
        runtime_config_resolver.state_directory(),
        config_update,
    ) {
        let status_code = if matches!(
            commit_error,
            config::AstronomicalConfigError::ConfigChangedDuringUpdate
        ) {
            StatusCode::CONFLICT
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        tracing::warn!(error = %commit_error, "MLX memory configuration commit was not accepted");
        return Err(Box::new(
            (
                status_code,
                Json(response_document(
                    maximum_mlx_memory_gb,
                    worker_health_snapshot,
                    "configuration changed during the memory update; retry after reloading"
                        .to_owned(),
                )),
            )
                .into_response(),
        ));
    }
    Ok(())
}

/// Applies the staged memory limit to the live worker, during a reply or
/// not, and answers with the outcome-specific response.
async fn apply_worker_memory_update(
    application_state: &ApplicationState,
    worker_handle: &crate::worker::handle::WorkerHandle,
    runtime_config_resolver: &crate::config::config_reload::ResolvedRuntimeConfigResolver,
    candidate_resolved_config: &crate::ResolvedRuntimeConfig,
    candidate_config_bytes: Vec<u8>,
    requested_mlx_memory_ceiling_bytes: u64,
    maximum_mlx_memory_gb: Option<u64>,
) -> Response {
    match worker_handle
        .update_mlx_memory_limit(
            requested_mlx_memory_ceiling_bytes,
            candidate_resolved_config.configuration_generation.clone(),
        )
        .await
    {
        Ok(MlxMemoryLimitUpdateOutcome::Applied) => applied_memory_update_response(
            application_state,
            worker_handle,
            runtime_config_resolver,
            candidate_resolved_config,
            &candidate_config_bytes,
            maximum_mlx_memory_gb,
        ),
        Ok(MlxMemoryLimitUpdateOutcome::Rejected) => rejected_memory_update_response(
            application_state,
            worker_handle,
            runtime_config_resolver,
            candidate_resolved_config,
            maximum_mlx_memory_gb,
        ),
        Err(worker_control_error) => memory_update_error_response(
            application_state,
            worker_handle,
            runtime_config_resolver,
            candidate_resolved_config,
            worker_control_error,
            maximum_mlx_memory_gb,
        ),
    }
}

/// The applied outcome: record the generation, commit the persisted
/// snapshots, and answer with the ceiling the worker now runs under.
fn applied_memory_update_response(
    application_state: &ApplicationState,
    worker_handle: &crate::worker::handle::WorkerHandle,
    runtime_config_resolver: &crate::config::config_reload::ResolvedRuntimeConfigResolver,
    candidate_resolved_config: &crate::ResolvedRuntimeConfig,
    candidate_config_bytes: &[u8],
    maximum_mlx_memory_gb: Option<u64>,
) -> Response {
    worker_handle.record_memory_configuration_generation(
        candidate_resolved_config.configuration_generation.clone(),
        MlxMemoryLimitUpdateOutcome::Applied,
    );
    crate::config::maximum_mlx_memory_transaction::commit_applied_config_snapshots(
        application_state,
        runtime_config_resolver,
        candidate_resolved_config,
        candidate_config_bytes,
    );
    // Both screens show this sentence to the person who moved the control,
    // so it names the setting as they see it, not the engine's internals.
    (
        StatusCode::OK,
        Json(response_document(
            maximum_mlx_memory_gb,
            &worker_handle.worker_health_snapshot(),
            "Maximum model RAM updated.".to_owned(),
        )),
    )
        .into_response()
}

/// The rejected outcome: the worker refused the ceiling; retain the persisted
/// intent and answer with the worker's own limit error, which names the
/// memory the model needs.
fn rejected_memory_update_response(
    application_state: &ApplicationState,
    worker_handle: &crate::worker::handle::WorkerHandle,
    runtime_config_resolver: &crate::config::config_reload::ResolvedRuntimeConfigResolver,
    candidate_resolved_config: &crate::ResolvedRuntimeConfig,
    maximum_mlx_memory_gb: Option<u64>,
) -> Response {
    worker_handle.record_memory_configuration_generation(
        candidate_resolved_config.configuration_generation.clone(),
        MlxMemoryLimitUpdateOutcome::Rejected,
    );
    crate::config::maximum_mlx_memory_transaction::retain_rejected_persisted_config(
        application_state,
        runtime_config_resolver,
    );
    let updated_worker_health_snapshot = worker_handle.worker_health_snapshot();
    invalid_request_response(
        maximum_mlx_memory_gb,
        &updated_worker_health_snapshot,
        updated_worker_health_snapshot
            .mlx_memory_limit_error
            .clone()
            .unwrap_or_else(|| "worker rejected the requested MLX memory ceiling".to_owned()),
    )
}

/// The control-error outcome: the update channel itself failed; retain the
/// persisted intent and answer 500.
fn memory_update_error_response(
    application_state: &ApplicationState,
    worker_handle: &crate::worker::handle::WorkerHandle,
    runtime_config_resolver: &crate::config::config_reload::ResolvedRuntimeConfigResolver,
    candidate_resolved_config: &crate::ResolvedRuntimeConfig,
    worker_control_error: crate::WorkerControlError,
    maximum_mlx_memory_gb: Option<u64>,
) -> Response {
    worker_handle.record_memory_configuration_generation(
        candidate_resolved_config.configuration_generation.clone(),
        MlxMemoryLimitUpdateOutcome::Rejected,
    );
    crate::config::maximum_mlx_memory_transaction::retain_rejected_persisted_config(
        application_state,
        runtime_config_resolver,
    );
    tracing::error!(error = %worker_control_error, "live MLX memory update failed");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(response_document(
            maximum_mlx_memory_gb,
            &worker_handle.worker_health_snapshot(),
            "worker control failed".to_owned(),
        )),
    )
        .into_response()
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
        message,
    }
}

use crate::{
    ActiveRequestProgress, ApplicationBuildIdentity, WorkerHandle, application::ApplicationState,
};
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};

pub(crate) async fn status_check(State(application_state): State<ApplicationState>) -> Response {
    let build_identity = ApplicationBuildIdentity::current();
    let (runtime_instance, state_directory_label) = reported_instance_identity(&application_state);
    // Activity is daemon-derived so the worker protocol stays focused on
    // generation data instead of duplicating phase-state events.
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    let mut status_json = serde_json::json!({
        "application": {
            "version": build_identity.version,
            "build_number": build_identity.build_number,
            "commit": build_identity.commit,
            "is_dirty": build_identity.is_dirty,
            "channel": runtime_instance.as_str(),
            "channel_display_name": runtime_instance.display_name(),
            "state_directory": state_directory_label,
        },
        "status": worker_health_snapshot.status.as_str(),
        "activity": worker_health_snapshot.activity.as_str(),
        "worker_runtime_feature_configuration_applied": worker_health_snapshot.worker_runtime_feature_configuration.is_some(),
        // Keep the exact worker acknowledgement available beside the derived convenience fields.
        // The menu compares this complete value with the reload response before declaring a
        // replacement applied, so a stale Ready status cannot masquerade as the new policy.
        "worker_runtime_feature_configuration": worker_health_snapshot.worker_runtime_feature_configuration.clone(),
    });
    status_json["configuration"] = serde_json::to_value(
        crate::config::configuration_status::ConfigurationStatusSummary::from_application(
            &application_state,
            &worker_health_snapshot,
        ),
    )
    .unwrap_or(serde_json::json!(null));
    status_json["configured_generation"] =
        status_json["configuration"]["configured_generation"].clone();
    status_json["resolved_generation"] =
        status_json["configuration"]["resolved_generation"].clone();
    status_json["effective_generation"] =
        status_json["configuration"]["effective_generation"].clone();
    if let Some(ready_model_id) = &worker_health_snapshot.ready_model_id {
        attach_ready_model_facts(&mut status_json, &application_state, ready_model_id);
    }
    attach_worker_memory_and_session_facts(
        &mut status_json,
        &worker_health_snapshot,
        &application_state,
    );
    attach_serving_and_prompt_cache_facts(&mut status_json, &worker_health_snapshot);
    attach_progress_or_idle_unload(
        &mut status_json,
        &worker_health_snapshot,
        &application_state,
    );
    Json(status_json).into_response()
}

/// The instance and state-directory label status reports. The label comes
/// from the instance paths, which know where the state really is; a daemon
/// without a resolver (a test harness) reports a Development instance in a
/// custom directory, which no installed application accepts as its own.
fn reported_instance_identity(
    application_state: &ApplicationState,
) -> (astronomical_config::AstronomicalRuntimeInstance, String) {
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.as_ref() else {
        return (
            astronomical_config::AstronomicalRuntimeInstance::Development,
            astronomical_config::CUSTOM_STATE_DIRECTORY_LABEL.to_owned(),
        );
    };
    let instance_paths = runtime_config_resolver.instance_paths();
    (
        instance_paths
            .runtime_instance()
            .unwrap_or(astronomical_config::AstronomicalRuntimeInstance::Development),
        instance_paths.state_directory_label().to_owned(),
    )
}

/// Pure: the client-facing expert-memory mode labels.
fn expert_memory_mode_label(
    expert_memory_mode: astronomical_ipc_protocol::ExpertMemoryMode,
) -> &'static str {
    match expert_memory_mode {
        astronomical_ipc_protocol::ExpertMemoryMode::Resident => "resident",
        astronomical_ipc_protocol::ExpertMemoryMode::Hybrid => "hybrid",
        astronomical_ipc_protocol::ExpertMemoryMode::Paged => "paged",
    }
}

/// Attaches the ready model's identifier and its resolved size, preferring
/// the reloadable configuration's discovery over the daemon-side snapshot.
fn attach_ready_model_facts(
    status_json: &mut serde_json::Value,
    application_state: &ApplicationState,
    ready_model_id: &str,
) {
    status_json["ready_model_id"] = serde_json::json!(ready_model_id);
    let ready_model_size_bytes = application_state
        .reloadable_config
        .as_ref()
        .and_then(|reloadable_config| reloadable_config.read().ok())
        .and_then(|resolved_runtime_config| {
            resolved_runtime_config
                .discovered_models
                .iter()
                .find(|discovered_model| discovered_model.model_id == *ready_model_id)
                .map(|discovered_model| discovered_model.model_size_bytes)
        })
        .or_else(|| {
            application_state
                .discovered_models
                .iter()
                .find(|discovered_model| discovered_model.model_id == *ready_model_id)
                .map(|discovered_model| discovered_model.model_size_bytes)
        });
    status_json["ready_model_size_bytes"] = serde_json::json!(ready_model_size_bytes);
}

/// Attaches the worker's expert-memory mode, expert residency, memory
/// ceilings, the configured user-facing gigabyte budget, and the live
/// serving session facts.
fn attach_worker_memory_and_session_facts(
    status_json: &mut serde_json::Value,
    worker_health_snapshot: &crate::WorkerHealthSnapshot,
    application_state: &ApplicationState,
) {
    status_json["expert_memory_mode"] = serde_json::json!(
        worker_health_snapshot
            .expert_memory_mode
            .map(expert_memory_mode_label)
    );
    status_json["neural_engine_prefill"] =
        serde_json::json!(worker_health_snapshot.neural_engine_prefill);
    status_json["model_load_in_flight"] =
        serde_json::json!(worker_health_snapshot.model_load_in_flight);
    super::expert_placement::attach_expert_placement_and_residency(
        status_json,
        worker_health_snapshot,
    );
    status_json["mlx_memory_snapshot"] =
        serde_json::json!(worker_health_snapshot.latest_mlx_memory_snapshot);
    status_json["mlx_memory_ceiling_bytes"] =
        serde_json::json!(worker_health_snapshot.mlx_memory_ceiling_bytes);
    status_json["machine_mlx_memory_ceiling_bytes"] =
        serde_json::json!(worker_health_snapshot.machine_mlx_memory_ceiling_bytes);
    status_json["wired_residency_budget_bytes"] =
        serde_json::json!(worker_health_snapshot.wired_residency_budget_bytes);
    status_json["minimum_mlx_memory_ceiling_bytes"] =
        serde_json::json!(worker_health_snapshot.minimum_mlx_memory_ceiling_bytes);
    status_json["pending_mlx_memory_ceiling_bytes"] =
        serde_json::json!(worker_health_snapshot.pending_mlx_memory_ceiling_bytes);
    status_json["mlx_memory_limit_error"] =
        serde_json::json!(worker_health_snapshot.mlx_memory_limit_error);
    status_json["configured_maximum_mlx_memory_gb"] = serde_json::json!(
        application_state
            .configured_config_snapshot
            .as_ref()
            .and_then(|configured_config| configured_config.read().ok())
            .and_then(|resolved_config| resolved_config.maximum_mlx_memory_bytes)
            .map(|maximum_mlx_memory_bytes| {
                // The user-facing gigabyte is the one Apple prints on the box
                // (AGENTS.md): 2³⁰ bytes.
                maximum_mlx_memory_bytes / 1_073_741_824
            })
    );
    status_json["serving_session"] = serde_json::json!({
        "completed_request_count": worker_health_snapshot.serving_session.completed_request_count,
        "total_prompt_token_count": worker_health_snapshot.serving_session.total_prompt_token_count,
        "total_reused_prompt_token_count": worker_health_snapshot.serving_session.total_reused_prompt_token_count,
        "target_prompt_work_token_count": worker_health_snapshot.serving_session.target_prompt_work_token_count,
        "target_reused_prompt_work_token_count": worker_health_snapshot.serving_session.target_reused_prompt_work_token_count,
        "average_prefill_tok_per_second": worker_health_snapshot.serving_session.average_prefill_tok_per_second,
        "average_generation_tok_per_second": worker_health_snapshot.serving_session.average_generation_tok_per_second,
        "neural_engine": neural_engine_session_facts(&worker_health_snapshot.serving_session.neural_engine),
    });
}

fn neural_engine_session_facts(
    totals: &crate::serving::neural_engine_session_totals::NeuralEngineSessionTotals,
) -> serde_json::Value {
    serde_json::json!({
        "prompt_token_count": totals.prompt_token_count,
        "neural_engine_token_count": totals.neural_engine_token_count,
        "estimated_saved_ms": totals.estimated_saved_millis,
        "last_chunk_placement": totals.last_chunk_placement,
    })
}

/// Attaches the prompt-cache hit summary derived from the worker's stats
/// event.
fn attach_serving_and_prompt_cache_facts(
    status_json: &mut serde_json::Value,
    worker_health_snapshot: &crate::WorkerHealthSnapshot,
) {
    let persistent_prompt_cache_summary = crate::PersistentPromptCacheSummary::from_worker_event(
        worker_health_snapshot
            .persistent_prompt_cache_stats
            .as_ref(),
    );
    status_json["persistent_prompt_cache"] = serde_json::json!({
        "hits": persistent_prompt_cache_summary.hits,
        "misses": persistent_prompt_cache_summary.misses,
        "tokens_saved": persistent_prompt_cache_summary.tokens_saved,
        "hit_rate": persistent_prompt_cache_summary.hit_rate(),
    });
}

/// Running work reports its progress; a loaded model with nothing running and
/// no load in flight reports when it will unload instead.
fn attach_progress_or_idle_unload(
    status_json: &mut serde_json::Value,
    worker_health_snapshot: &crate::WorkerHealthSnapshot,
    application_state: &ApplicationState,
) {
    if let Some(progress) = worker_health_snapshot.active_request_progress.clone() {
        attach_active_request_progress(status_json, progress);
        return;
    }
    if worker_health_snapshot.ready_model_id.is_some()
        && worker_health_snapshot.model_load_in_flight.is_none()
        && let Some(worker_handle) = application_state.worker_control.as_ref()
    {
        attach_idle_model_unload(status_json, worker_handle);
    }
}

/// Says when an idle loaded model unloads, so Overview can show it. Present
/// only while a model is loaded and nothing runs; `remaining_seconds` is null
/// when the span is 0, which keeps a model loaded.
fn attach_idle_model_unload(status_json: &mut serde_json::Value, worker_handle: &WorkerHandle) {
    let countdown = worker_handle.idle_model_unload_countdown();
    status_json["idle_model_unload"] = serde_json::json!({
        "after_seconds": countdown.after_seconds,
        "remaining_seconds": countdown.remaining_seconds,
    });
}

/// The preparation progress: no token progress to invent, the elapsed
/// times, and the residency counts only when the announcement carried them.
fn generation_preparation_progress(
    request_started_at: tokio::time::Instant,
    preparation_started_at: tokio::time::Instant,
    preparation_residency_counts: Option<crate::PreparationResidencyCounts>,
) -> serde_json::Value {
    let mut preparation_progress = serde_json::json!({
        "phase": "generation_preparation",
        "processed_tokens": 0,
        "total_tokens": 1,
        "elapsed_ms": u64::try_from(preparation_started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
        "request_elapsed_ms": u64::try_from(request_started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
    });
    if let Some(preparation_residency_counts) = preparation_residency_counts {
        preparation_progress["total_layer_count"] =
            serde_json::json!(preparation_residency_counts.total_layer_count);
        preparation_progress["resident_expert_count"] =
            serde_json::json!(preparation_residency_counts.resident_expert_count);
        preparation_progress["resident_expert_payload_bytes"] =
            serde_json::json!(preparation_residency_counts.resident_expert_payload_bytes);
    }
    preparation_progress
}

/// The prompt clock the status reports. While prompt tokens remain it runs
/// live from the request start, so a long chunk still shows time passing.
/// Once every prompt token is processed it stops at the worker's measured
/// prompt time: whatever runs after that (decode residency, the first
/// forward) is not prompt work, and letting the clock run made a finished
/// 20-token prompt read "100% · 2 avg tok/s".
fn prompt_progress_elapsed_millis(
    processed_tokens: u32,
    total_tokens: u32,
    measured_prompt_elapsed_millis: u64,
    request_elapsed: std::time::Duration,
) -> u64 {
    if processed_tokens >= total_tokens {
        return measured_prompt_elapsed_millis;
    }
    measured_prompt_elapsed_millis
        .max(u64::try_from(request_elapsed.as_millis()).unwrap_or(u64::MAX))
}

/// Attaches the live active-request progress, keeping the worker protocol's
/// phase state out of the daemon's JSON contract one arm at a time.
fn attach_active_request_progress(
    status_json: &mut serde_json::Value,
    progress: ActiveRequestProgress,
) {
    match progress {
        ActiveRequestProgress::Prefill {
            prompt_processing_phase,
            processed_tokens,
            total_tokens,
            elapsed_millis,
            request_started_at,
            completed_prefill_chunk_tokens,
        } => {
            let live_elapsed_millis = prompt_progress_elapsed_millis(
                processed_tokens,
                total_tokens,
                elapsed_millis,
                request_started_at.elapsed(),
            );
            status_json["progress"] = serde_json::json!({
                "phase": prompt_processing_phase,
                "processed_tokens": processed_tokens,
                "total_tokens": total_tokens,
                "elapsed_ms": live_elapsed_millis,
            });
            if let Some(completed_prefill_chunk_tokens) = completed_prefill_chunk_tokens {
                status_json["progress"]["completed_prefill_chunk_tokens"] =
                    serde_json::json!(completed_prefill_chunk_tokens);
            }
        }
        ActiveRequestProgress::Generation {
            generated_token_count,
            maximum_output_tokens,
            elapsed_millis,
        } => {
            status_json["progress"] = serde_json::json!({
                "phase": "generation",
                "processed_tokens": generated_token_count,
                "total_tokens": maximum_output_tokens,
                "elapsed_ms": elapsed_millis,
            });
        }
        ActiveRequestProgress::GenerationPreparation {
            request_started_at,
            preparation_started_at,
            preparation_residency_counts,
        } => {
            status_json["progress"] = generation_preparation_progress(
                request_started_at,
                preparation_started_at,
                preparation_residency_counts,
            );
        }
        ActiveRequestProgress::ImageGeneration {
            phase,
            completed_steps,
            total_steps,
            elapsed_millis,
        } => {
            status_json["progress"] = serde_json::json!({
                "phase": phase,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "elapsed_ms": elapsed_millis,
            });
        }
    }
}

#[cfg(test)]
#[path = "status_tests.rs"]
mod status_tests;

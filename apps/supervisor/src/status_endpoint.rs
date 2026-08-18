use crate::{ActiveRequestProgress, ApplicationBuildIdentity, application::ApplicationState};
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};

pub(super) async fn status_check(State(application_state): State<ApplicationState>) -> Response {
    let build_identity = ApplicationBuildIdentity::current();
    let runtime_instance = application_state
        .runtime_config_resolver
        .as_ref()
        .and_then(|runtime_config_resolver| {
            runtime_config_resolver.instance_paths().runtime_instance()
        })
        .unwrap_or(astronomical_config::AstronomicalRuntimeInstance::Development);
    let state_directory_label = match (
        application_state
            .runtime_config_resolver
            .as_ref()
            .map(|runtime_config_resolver| {
                runtime_config_resolver
                    .instance_paths()
                    .is_standard_state_directory()
            })
            .unwrap_or(false),
        runtime_instance,
    ) {
        (true, astronomical_config::AstronomicalRuntimeInstance::Stable) => "~/.astronomical",
        (true, astronomical_config::AstronomicalRuntimeInstance::Development) => {
            "~/.astronomical-dev"
        }
        (false, _) => "custom",
    };
    // Activity is supervisor-derived so the worker protocol stays focused on
    // generation data instead of duplicating phase-state events.
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    // Read both user-facing MTP fields under one lock so a concurrent config replacement cannot
    // publish an enabled flag from one generation and a draft depth from another.
    let reloadable_mtp_configuration = application_state
        .reloadable_config
        .as_ref()
        .and_then(|reloadable_config| reloadable_config.read().ok())
        .map(|resolved_config| (resolved_config.mtp_enabled, resolved_config.mtp_draft_depth));
    // Prefer the reloadable user policy when the application owns one. Test embeddings and other
    // non-reloadable hosts still receive the worker's explicit acknowledgement, so status never
    // reports MTP disabled while that same worker reports an active MTP runtime.
    let mtp_enabled = reloadable_mtp_configuration
        .map(|(mtp_enabled, _mtp_draft_depth)| mtp_enabled)
        .or_else(|| {
            worker_health_snapshot
                .worker_runtime_feature_configuration
                .map(|configuration| configuration.mtp_enabled)
        })
        .unwrap_or(false);
    let configured_mtp_pairing =
        worker_health_snapshot
            .ready_model_id
            .as_deref()
            .and_then(|ready_model_id| {
                application_state
                    .reloadable_config
                    .as_ref()
                    .and_then(|reloadable_config| reloadable_config.read().ok())
                    .and_then(|resolved_config| {
                        resolved_config
                            .mtp_pairings
                            .iter()
                            .find(|pairing| pairing.applies_to_loaded_model(ready_model_id))
                            .map(|pairing| {
                                (
                                    pairing.target_model_id.clone(),
                                    pairing.drafter_model_id.clone(),
                                    pairing.discovered_drafter_revision.clone(),
                                    pairing.drafter_model_directory.is_some(),
                                )
                            })
                    })
            });
    let configured_speculative_prefill_enabled = application_state
        .reloadable_config
        .as_ref()
        .and_then(|reloadable_config| reloadable_config.read().ok())
        .is_some_and(|resolved_runtime_config| {
            resolved_runtime_config.speculative_prefill.is_enabled()
        });
    let speculative_prefill_enabled = worker_health_snapshot
        .worker_runtime_feature_configuration
        .map(|worker_runtime_feature_configuration| {
            worker_runtime_feature_configuration.speculative_prefill_enabled
        })
        .unwrap_or_else(|| {
            !matches!(
                worker_health_snapshot.speculative_prefill_runtime_state,
                astronomical_ipc_protocol::SpeculativePrefillRuntimeState::Disabled
            )
        });
    let configured_speculative_prefill_draft_model_id = speculative_prefill_enabled
        .then(|| {
            application_state
                .reloadable_config
                .as_ref()
                .and_then(|reloadable_config| reloadable_config.read().ok())
                .and_then(|resolved_runtime_config| {
                    resolved_runtime_config
                        .speculative_prefill
                        .draft_model_id()
                        .map(str::to_owned)
                })
                .or_else(|| {
                    worker_health_snapshot
                        .speculative_prefill_draft_model_id
                        .clone()
                })
        })
        .flatten();
    let configured_speculative_prefill_target_model_id = speculative_prefill_enabled
        .then(|| application_state.configured_speculative_prefill_target_model_id())
        .flatten();
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
        "mtp_enabled": mtp_enabled,
        "mtp_configured_draft_depth": reloadable_mtp_configuration
            .and_then(|(_mtp_enabled, mtp_draft_depth)| mtp_draft_depth)
            .or_else(|| worker_health_snapshot
                .worker_runtime_feature_configuration
                .and_then(|configuration| configuration.mtp_draft_depth))
            .or(worker_health_snapshot.mtp_depth_status.configured_draft_depth),
        "mtp_artifact_maximum_draft_depth": worker_health_snapshot.mtp_depth_status.artifact_maximum_draft_depth,
        "mtp_artifact_default_draft_depth": worker_health_snapshot.mtp_depth_status.artifact_default_draft_depth,
        "mtp_resolved_requested_draft_depth": worker_health_snapshot.mtp_depth_status.resolved_requested_draft_depth,
        "mtp_effective_execution_draft_depth": worker_health_snapshot.mtp_depth_status.effective_execution_draft_depth,
        "mtp_runtime_state": serde_json::to_value(worker_health_snapshot.mtp_runtime_state())
            .unwrap_or_else(|_| serde_json::json!("disabled")),
        "mtp_unavailable_reason": worker_health_snapshot.mtp_unavailable_reason(),
        "mtp_pairing_target_model_id": configured_mtp_pairing
            .as_ref()
            .map(|(target_model_id, _, _, _)| target_model_id),
        "mtp_pairing_drafter_model_id": configured_mtp_pairing
            .as_ref()
            .map(|(_, drafter_model_id, _, _)| drafter_model_id),
        "mtp_pairing_drafter_revision": configured_mtp_pairing
            .as_ref()
            .and_then(|(_, _, drafter_revision, _)| drafter_revision.as_deref()),
        "mtp_pairing_drafter_discovered": configured_mtp_pairing
            .as_ref()
            .map(|(_, _, _, drafter_discovered)| drafter_discovered),
        "speculative_prefill_enabled": speculative_prefill_enabled,
        "configured_speculative_prefill_enabled": configured_speculative_prefill_enabled,
        "worker_runtime_feature_configuration_applied": worker_health_snapshot.worker_runtime_feature_configuration.is_some(),
        // Keep the exact worker acknowledgement available beside the derived convenience fields.
        // The menu compares this complete value with the reload response before declaring a
        // replacement applied, so a stale Ready status cannot masquerade as the new policy.
        "worker_runtime_feature_configuration": worker_health_snapshot.worker_runtime_feature_configuration,
        "speculative_prefill_runtime_state": serde_json::to_value(
            worker_health_snapshot.speculative_prefill_runtime_state,
        )
        .unwrap_or_else(|_| serde_json::json!("disabled")),
        "speculative_prefill_unavailable_reason": worker_health_snapshot
            .speculative_prefill_unavailable_reason
            .as_deref(),
        "speculative_prefill_draft_model_id": configured_speculative_prefill_draft_model_id,
        "speculative_prefill_target_model_id": configured_speculative_prefill_target_model_id,
        "speculative_prefill_draft_model_revision": worker_health_snapshot
            .speculative_prefill_draft_model_revision
            .as_deref(),
    });
    if let Some(ready_model_id) = &worker_health_snapshot.ready_model_id {
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
    status_json["expert_memory_mode"] = serde_json::json!(
        worker_health_snapshot.expert_memory_mode.map(
            |expert_memory_mode| match expert_memory_mode {
                astronomical_ipc_protocol::ExpertMemoryMode::Resident => "resident",
                astronomical_ipc_protocol::ExpertMemoryMode::Hybrid => "hybrid",
                astronomical_ipc_protocol::ExpertMemoryMode::Paged => "paged",
            }
        )
    );
    status_json["expert_residency"] =
        serde_json::json!(
        worker_health_snapshot.expert_residency.map(|expert_residency| {
            serde_json::json!({
                "total_layer_count": expert_residency.total_layer_count,
                "complete_layer_count": expert_residency.complete_layer_count,
                "complete_layer_payload_bytes": expert_residency.complete_layer_payload_bytes,
                "partial_layer_count": expert_residency.partial_layer_count,
                "partial_layer_payload_bytes": expert_residency.partial_layer_payload_bytes,
            })
        })
    );
    status_json["mlx_memory_snapshot"] =
        serde_json::json!(worker_health_snapshot.latest_mlx_memory_snapshot);
    status_json["mlx_memory_ceiling_bytes"] =
        serde_json::json!(worker_health_snapshot.mlx_memory_ceiling_bytes);
    status_json["machine_mlx_memory_ceiling_bytes"] =
        serde_json::json!(worker_health_snapshot.machine_mlx_memory_ceiling_bytes);
    status_json["minimum_mlx_memory_ceiling_bytes"] =
        serde_json::json!(worker_health_snapshot.minimum_mlx_memory_ceiling_bytes);
    status_json["pending_mlx_memory_ceiling_bytes"] =
        serde_json::json!(worker_health_snapshot.pending_mlx_memory_ceiling_bytes);
    status_json["mlx_memory_limit_error"] =
        serde_json::json!(worker_health_snapshot.mlx_memory_limit_error);
    status_json["configured_maximum_mlx_memory_gb"] = serde_json::json!(
        application_state
            .reloadable_config
            .as_ref()
            .and_then(|reloadable_config| reloadable_config.read().ok())
            .and_then(|resolved_config| resolved_config.maximum_mlx_memory_bytes)
            .map(|maximum_mlx_memory_bytes| maximum_mlx_memory_bytes / 1_000_000_000)
    );
    status_json["serving_session"] = serde_json::json!({
        "completed_request_count": worker_health_snapshot.serving_session.completed_request_count,
        "total_prompt_token_count": worker_health_snapshot.serving_session.total_prompt_token_count,
        "total_reused_prompt_token_count": worker_health_snapshot.serving_session.total_reused_prompt_token_count,
        "target_prompt_work_token_count": worker_health_snapshot.serving_session.target_prompt_work_token_count,
        "target_reused_prompt_work_token_count": worker_health_snapshot.serving_session.target_reused_prompt_work_token_count,
        "drafter_prompt_work_token_count": worker_health_snapshot.serving_session.drafter_prompt_work_token_count,
        "drafter_reused_prompt_work_token_count": worker_health_snapshot.serving_session.drafter_reused_prompt_work_token_count,
        "average_prefill_tok_per_second": worker_health_snapshot.serving_session.average_prefill_tok_per_second,
        "average_generation_tok_per_second": worker_health_snapshot.serving_session.average_generation_tok_per_second,
    });
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
    if let Some(progress) = worker_health_snapshot.active_request_progress {
        match progress {
            ActiveRequestProgress::Prefill {
                prompt_processing_phase,
                processed_tokens,
                total_tokens,
                elapsed_millis,
                request_started_at,
                completed_prefill_chunk_tokens,
            } => {
                let live_elapsed_millis = elapsed_millis.max(
                    u64::try_from(request_started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
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
                total_layer_count,
                complete_layer_count,
                partial_layer_count,
            } => {
                status_json["progress"] = serde_json::json!({
                    "phase": "generation_preparation",
                    "processed_tokens": 0,
                    "total_tokens": 1,
                    "elapsed_ms": u64::try_from(preparation_started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
                    "request_elapsed_ms": u64::try_from(request_started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
                    "total_layer_count": total_layer_count,
                    "complete_layer_count": complete_layer_count,
                    "partial_layer_count": partial_layer_count,
                });
            }
        }
    }
    Json(status_json).into_response()
}

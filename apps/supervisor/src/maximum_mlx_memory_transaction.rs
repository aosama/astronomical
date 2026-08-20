//! Owns snapshot commit and deferred rollback for the narrow live-memory mutation.

use tokio::time::{Duration, sleep};

use crate::application::ApplicationState;
use crate::{ResolvedRuntimeConfig, ResolvedRuntimeConfigResolver};

pub(crate) fn commit_applied_config_snapshots(
    application_state: &ApplicationState,
    runtime_config_resolver: &ResolvedRuntimeConfigResolver,
    candidate_resolved_config: &ResolvedRuntimeConfig,
    candidate_config_bytes: &[u8],
) {
    if let Some(reloadable_config) = application_state.reloadable_config.as_ref() {
        *reloadable_config
            .write()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) =
            candidate_resolved_config.clone();
    }
    let current_file_matches_candidate = std::fs::read(
        runtime_config_resolver
            .state_directory()
            .join("config.json"),
    )
    .ok()
    .is_some_and(|current_config_bytes| current_config_bytes == candidate_config_bytes);
    let configured_resolved_config = if current_file_matches_candidate {
        Some(candidate_resolved_config.clone())
    } else {
        runtime_config_resolver.load().ok()
    };
    if let Some(configured_resolved_config) = configured_resolved_config
        && let Some(configured_config_snapshot) =
            application_state.configured_config_snapshot.as_ref()
    {
        *configured_config_snapshot
            .write()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) =
            configured_resolved_config;
    }
}

pub(crate) async fn reconcile_queued_memory_config(
    application_state: ApplicationState,
    candidate_resolved_config: ResolvedRuntimeConfig,
    candidate_config_bytes: Vec<u8>,
    prior_resolved_config: Option<ResolvedRuntimeConfig>,
) {
    loop {
        let worker_health_snapshot = application_state
            .generation_executor
            .worker_health_snapshot();
        if worker_health_snapshot
            .pending_mlx_memory_ceiling_bytes
            .is_some()
        {
            sleep(Duration::from_millis(25)).await;
            continue;
        }

        let _configuration_transition_guard =
            application_state.configuration_transition_lock.lock().await;
        let mut pending_generation = application_state
            .pending_memory_config_generation
            .lock()
            .await;
        if pending_generation.as_deref()
            != Some(candidate_resolved_config.configuration_generation.as_str())
        {
            return;
        }
        let was_applied = worker_health_snapshot
            .worker_runtime_feature_configuration
            .as_ref()
            .is_some_and(|configuration| {
                configuration.configuration_generation
                    == candidate_resolved_config.configuration_generation
            })
            && worker_health_snapshot.mlx_memory_limit_error.is_none();
        let Some(runtime_config_resolver) = application_state.runtime_config_resolver.as_ref()
        else {
            *pending_generation = None;
            return;
        };
        let current_file_matches_candidate = std::fs::read(
            runtime_config_resolver
                .state_directory()
                .join("config.json"),
        )
        .ok()
        .is_some_and(|current_config_bytes| current_config_bytes == candidate_config_bytes);
        if was_applied {
            refresh_newer_configured_snapshot(
                &application_state,
                runtime_config_resolver,
                current_file_matches_candidate,
            );
            *pending_generation = None;
            return;
        }

        reconcile_rejected_candidate(
            &application_state,
            runtime_config_resolver,
            prior_resolved_config,
        );
        *pending_generation = None;
        return;
    }
}

fn refresh_newer_configured_snapshot(
    application_state: &ApplicationState,
    runtime_config_resolver: &ResolvedRuntimeConfigResolver,
    current_file_matches_candidate: bool,
) {
    if current_file_matches_candidate {
        return;
    }
    if let Ok(configured_resolved_config) = runtime_config_resolver.load()
        && let Some(configured_config_snapshot) =
            application_state.configured_config_snapshot.as_ref()
    {
        *configured_config_snapshot
            .write()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) =
            configured_resolved_config;
    }
}

fn reconcile_rejected_candidate(
    application_state: &ApplicationState,
    runtime_config_resolver: &ResolvedRuntimeConfigResolver,
    prior_resolved_config: Option<ResolvedRuntimeConfig>,
) {
    // Persisted intent may have been edited outside the process. Keeping it intact is safer than
    // a non-atomic read-then-restore that could overwrite a concurrent editor.
    if let Some(prior_resolved_config) = prior_resolved_config
        && let Some(reloadable_config) = application_state.reloadable_config.as_ref()
    {
        *reloadable_config
            .write()
            .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) = prior_resolved_config;
    }
    refresh_newer_configured_snapshot(application_state, runtime_config_resolver, false);
}

pub(crate) fn retain_rejected_persisted_config(
    application_state: &ApplicationState,
    runtime_config_resolver: &ResolvedRuntimeConfigResolver,
) {
    refresh_newer_configured_snapshot(application_state, runtime_config_resolver, false);
}

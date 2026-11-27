//! Owns the configuration snapshots a live memory-limit change commits, or
//! keeps, once the worker has answered it.

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

pub(crate) fn retain_rejected_persisted_config(
    application_state: &ApplicationState,
    runtime_config_resolver: &ResolvedRuntimeConfigResolver,
) {
    refresh_newer_configured_snapshot(application_state, runtime_config_resolver, false);
}

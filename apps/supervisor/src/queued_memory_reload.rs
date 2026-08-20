//! Reconciles a live memory limit queued by config reload after generation finalizes.

use crate::ResolvedRuntimeConfig;
use crate::application::ApplicationState;

pub(crate) async fn reconcile_reloaded_memory_config(
    application_state: ApplicationState,
    effective_memory_generation: String,
    prior_resolved_config: ResolvedRuntimeConfig,
) {
    loop {
        let worker_health_snapshot = application_state
            .generation_executor
            .worker_health_snapshot();
        if worker_health_snapshot
            .pending_mlx_memory_ceiling_bytes
            .is_some()
        {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            continue;
        }
        let _configuration_transition_guard =
            application_state.configuration_transition_lock.lock().await;
        let mut pending_generation = application_state
            .pending_memory_config_generation
            .lock()
            .await;
        if pending_generation.as_deref() != Some(effective_memory_generation.as_str()) {
            return;
        }
        let was_applied = worker_health_snapshot
            .worker_runtime_feature_configuration
            .as_ref()
            .is_some_and(|configuration| {
                configuration.configuration_generation == effective_memory_generation
            })
            && worker_health_snapshot.mlx_memory_limit_error.is_none();
        if !was_applied
            && let Some(reloadable_config) = application_state.reloadable_config.as_ref()
        {
            *reloadable_config
                .write()
                .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) = prior_resolved_config;
        }
        *pending_generation = None;
        return;
    }
}

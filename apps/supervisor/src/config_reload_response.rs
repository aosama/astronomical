//! Owns the internal reload response consumed by Astronomical's menu application.

use astronomical_ipc_protocol::WorkerRuntimeFeatureConfiguration;
use serde::Serialize;

#[derive(Serialize)]
pub(crate) struct ConfigReloadResponse {
    status: &'static str,
    message: String,
    worker_restart_completed: bool,
    rest_api_restart_required: bool,
    restart_required_fields: Vec<String>,
    reloaded_fields: Vec<String>,
    discovered_model_count: usize,
    worker_runtime_feature_configuration: Option<WorkerRuntimeFeatureConfiguration>,
    candidate_generation: Option<String>,
    effective_generation: Option<String>,
}

impl ConfigReloadResponse {
    pub(crate) fn reloaded(reloaded_fields: Vec<String>, discovered_model_count: usize) -> Self {
        Self {
            status: "reloaded",
            message: "Config reloaded".to_owned(),
            worker_restart_completed: false,
            rest_api_restart_required: false,
            restart_required_fields: Vec::new(),
            reloaded_fields,
            discovered_model_count,
            worker_runtime_feature_configuration: None,
            candidate_generation: None,
            effective_generation: None,
        }
    }

    pub(crate) fn restart_required(
        reloaded_fields: Vec<String>,
        restart_required_fields: Vec<String>,
        discovered_model_count: usize,
    ) -> Self {
        Self {
            status: "restart_required",
            message: "Config is valid, but a full server restart is required".to_owned(),
            worker_restart_completed: false,
            rest_api_restart_required: true,
            restart_required_fields,
            reloaded_fields,
            discovered_model_count,
            worker_runtime_feature_configuration: None,
            candidate_generation: None,
            effective_generation: None,
        }
    }

    pub(crate) fn worker_restart_completed(
        reloaded_fields: Vec<String>,
        discovered_model_count: usize,
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration,
    ) -> Self {
        Self {
            status: "reloaded",
            message: "Config reloaded and applied by the worker".to_owned(),
            worker_restart_completed: true,
            rest_api_restart_required: false,
            restart_required_fields: Vec::new(),
            reloaded_fields,
            discovered_model_count,
            worker_runtime_feature_configuration: Some(worker_runtime_feature_configuration),
            candidate_generation: None,
            effective_generation: None,
        }
    }

    pub(crate) fn invalid_config(validation_error: String) -> Self {
        Self::failure("invalid_config", validation_error, 0)
    }

    pub(crate) fn busy() -> Self {
        Self::failure(
            "busy",
            "A generation is active or queued; reload aborted".to_owned(),
            0,
        )
    }

    pub(crate) fn failed(message: String, discovered_model_count: usize) -> Self {
        Self::failure("failed", message, discovered_model_count)
    }

    pub(crate) fn with_generations(
        mut self,
        candidate_generation: &str,
        effective_generation: Option<String>,
    ) -> Self {
        self.candidate_generation = Some(candidate_generation.to_owned());
        self.effective_generation = effective_generation;
        self
    }

    fn failure(status: &'static str, message: String, discovered_model_count: usize) -> Self {
        Self {
            status,
            message,
            worker_restart_completed: false,
            rest_api_restart_required: false,
            restart_required_fields: Vec::new(),
            reloaded_fields: Vec::new(),
            discovered_model_count,
            worker_runtime_feature_configuration: None,
            candidate_generation: None,
            effective_generation: None,
        }
    }
}

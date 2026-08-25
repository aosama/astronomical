//! Builds the complete runtime policy catalog from resolved user configuration.
//!
//! Keeping tagged chat and image worker policy construction together ensures
//! discovery, configuration generations, and worker replacement compare the
//! same immutable execution policy.

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use astronomical_config::{
    AstronomicalConfig, AstronomicalConfigError, ChatModelCapabilities, DiscoveredModel,
    ModelCapabilities, ResolvedModelConfig, SpeculativePrefillConfig,
};
use astronomical_ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerFlux2KleinModelConfiguration,
    WorkerImageGenerationModelFamily, WorkerModelConfiguration,
    WorkerSpeculativePrefillConfiguration,
};

use crate::runtime_model_policy::{
    runtime_model_generation_defaults, worker_chunking_configuration,
};
use crate::{
    ConfiguredSpeculativePrefillPolicy, RuntimeModelAccelerationAvailability,
    RuntimeModelGenerationDefaults, RuntimeModelPolicy,
};

/// Resolves every discovered model into the exact policy sent to the worker.
pub(super) struct ResolvedModelPolicyCatalog;

impl ResolvedModelPolicyCatalog {
    pub(super) fn resolve(
        user_config: &AstronomicalConfig,
        discovered_models: &[DiscoveredModel],
        artifact_context_windows: &HashMap<String, u32>,
    ) -> Result<Arc<HashMap<String, RuntimeModelPolicy>>, AstronomicalConfigError> {
        let discovered_model_directories = discovered_models
            .iter()
            .map(|discovered_model| {
                (
                    discovered_model.model_id.clone(),
                    discovered_model.model_directory.clone(),
                )
            })
            .collect::<HashMap<_, _>>();
        let model_policies = discovered_models
            .iter()
            .map(|discovered_model| {
                let runtime_model_policy = match &discovered_model.capabilities {
                    ModelCapabilities::Chat(chat_capabilities) => Self::chat_policy(
                        user_config,
                        discovered_model,
                        chat_capabilities,
                        artifact_context_windows,
                        &discovered_model_directories,
                    )?,
                    ModelCapabilities::ImageGeneration(_) => Self::image_policy(discovered_model),
                };
                Ok((discovered_model.model_id.clone(), runtime_model_policy))
            })
            .collect::<Result<HashMap<_, _>, AstronomicalConfigError>>()?;

        Ok(Arc::new(model_policies))
    }

    fn chat_policy(
        user_config: &AstronomicalConfig,
        discovered_model: &DiscoveredModel,
        chat_capabilities: &ChatModelCapabilities,
        artifact_context_windows: &HashMap<String, u32>,
        discovered_model_directories: &HashMap<String, PathBuf>,
    ) -> Result<RuntimeModelPolicy, AstronomicalConfigError> {
        let resolved_model_config = user_config
            .resolved_model_config(&discovered_model.model_id, chat_capabilities.context_window)?;
        let (worker_model_configuration, acceleration_availability) =
            Self::worker_model_configuration(
                discovered_model,
                chat_capabilities,
                &resolved_model_config,
                discovered_model_directories,
            );

        Ok(RuntimeModelPolicy {
            model_directory: discovered_model.model_directory.clone(),
            generation_defaults: runtime_model_generation_defaults(&resolved_model_config),
            configured_maximum_context_tokens: resolved_model_config.maximum_context_tokens(),
            default_maximum_context_tokens: artifact_context_windows
                .get(&discovered_model.model_id)
                .copied()
                .unwrap_or(chat_capabilities.context_window),
            configured_chunking_fields: resolved_model_config.configured_chunking_fields(),
            acceleration_availability,
            worker_model_configuration,
        })
    }

    fn worker_model_configuration(
        discovered_model: &DiscoveredModel,
        chat_capabilities: &ChatModelCapabilities,
        resolved_model_config: &ResolvedModelConfig,
        discovered_model_directories: &HashMap<String, PathBuf>,
    ) -> (
        WorkerModelConfiguration,
        RuntimeModelAccelerationAvailability,
    ) {
        let configured_speculative_prefill = resolved_model_config.speculative_prefill();
        let speculative_prefill =
            configured_speculative_prefill.and_then(|speculative_prefill_config| {
                let draft_model_id = speculative_prefill_config
                    .draft_model_id()
                    .unwrap_or_default();
                discovered_model_directories
                    .get(draft_model_id)
                    .cloned()
                    .map(|draft_model_directory| {
                        speculative_prefill_configuration(
                            speculative_prefill_config,
                            draft_model_directory,
                        )
                    })
            });
        let acceleration_availability = RuntimeModelAccelerationAvailability {
            configured_mtp_enabled: resolved_model_config.configured_mtp_enabled(),
            configured_speculative_prefill: configured_speculative_prefill.map(|configuration| {
                ConfiguredSpeculativePrefillPolicy {
                    draft_model_id: configuration
                        .draft_model_id()
                        .unwrap_or_default()
                        .to_owned(),
                    keep_percentage: configuration.keep_percentage(),
                    minimum_prompt_tokens: configuration.minimum_prompt_tokens(),
                }
            }),
            speculative_prefill_unavailable_reason: configured_speculative_prefill
                .filter(|_| speculative_prefill.is_none())
                .map(|_| {
                    "configured speculative-prefill drafter is not currently discovered".to_owned()
                }),
        };

        (
            WorkerModelConfiguration::Autoregressive(WorkerAutoregressiveModelConfiguration {
                model_id: discovered_model.model_id.clone(),
                maximum_context_tokens: chat_capabilities.context_window,
                // Worker policy carries model capability rather than a request default.
                maximum_output_tokens: chat_capabilities.max_output_tokens,
                chunking: worker_chunking_configuration(resolved_model_config.chunking()),
                mtp_enabled: resolved_model_config.mtp_enabled(),
                mtp_draft_depth: resolved_model_config.mtp_draft_depth(),
                speculative_prefill,
            }),
            acceleration_availability,
        )
    }

    fn image_policy(discovered_model: &DiscoveredModel) -> RuntimeModelPolicy {
        RuntimeModelPolicy {
            model_directory: discovered_model.model_directory.clone(),
            // Chat request defaults remain inert for a typed image worker policy.
            generation_defaults: RuntimeModelGenerationDefaults {
                maximum_output_tokens: 0,
                configured_maximum_output_tokens: None,
                temperature_thousandths: None,
                top_p_thousandths: None,
            },
            configured_maximum_context_tokens: None,
            default_maximum_context_tokens: 0,
            configured_chunking_fields: Default::default(),
            acceleration_availability: Default::default(),
            worker_model_configuration: WorkerModelConfiguration::Flux2Klein(
                WorkerFlux2KleinModelConfiguration {
                    model_id: discovered_model.model_id.clone(),
                    model_family: WorkerImageGenerationModelFamily::Flux2Klein,
                    artifact_revision: discovered_model.revision.clone(),
                },
            ),
        }
    }
}

fn speculative_prefill_configuration(
    speculative_prefill_config: &SpeculativePrefillConfig,
    draft_model_directory: PathBuf,
) -> WorkerSpeculativePrefillConfiguration {
    WorkerSpeculativePrefillConfiguration {
        enabled: speculative_prefill_config.is_enabled(),
        target_model_id: speculative_prefill_config
            .target_model_id()
            .map(str::to_owned),
        draft_model_id: speculative_prefill_config
            .draft_model_id()
            .map(str::to_owned),
        draft_model_directory: Some(draft_model_directory),
        minimum_prompt_tokens: speculative_prefill_config.minimum_prompt_tokens(),
        keep_percentage: speculative_prefill_config.keep_percentage(),
        selection_chunck_token_count: speculative_prefill_config.selection_chunck_token_count(),
        mandatory_trailing_token_count: speculative_prefill_config.mandatory_trailing_token_count(),
        lookahead_token_count: speculative_prefill_config.lookahead_token_count(),
        importance_pooling_kernel_token_count: speculative_prefill_config
            .importance_pooling_kernel_token_count(),
    }
}

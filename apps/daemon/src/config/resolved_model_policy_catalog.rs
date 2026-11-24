//! Builds the complete runtime policy catalog from resolved user configuration.
//!
//! Keeping tagged chat and image worker policy construction together ensures
//! discovery, configuration generations, and worker replacement compare the
//! same immutable execution policy.

use std::{collections::HashMap, sync::Arc};

use config::{
    AstronomicalConfig, AstronomicalConfigError, ChatModelCapabilities, DiscoveredModel,
    ModelCapabilities, ResolvedModelConfig,
};
use ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerDecisionModelConfiguration,
    WorkerEmbeddingModelConfiguration, WorkerImageGenerationModelConfiguration,
    WorkerModelConfiguration, WorkerSpeechModelConfiguration,
};

use crate::config::runtime_model_policy::{
    runtime_model_generation_defaults, worker_chunking_configuration,
};
use crate::{RuntimeModelGenerationDefaults, RuntimeModelPolicy};

/// Resolves every discovered model into the exact policy sent to the worker.
pub(super) struct ResolvedModelPolicyCatalog;

impl ResolvedModelPolicyCatalog {
    pub(super) fn resolve(
        user_config: &AstronomicalConfig,
        discovered_models: &[DiscoveredModel],
        artifact_context_windows: &HashMap<String, u32>,
    ) -> Result<Arc<HashMap<String, RuntimeModelPolicy>>, AstronomicalConfigError> {
        let model_policies = discovered_models
            .iter()
            .map(|discovered_model| {
                let runtime_model_policy = match &discovered_model.capabilities {
                    ModelCapabilities::Chat(chat_capabilities) => Self::chat_policy(
                        user_config,
                        discovered_model,
                        chat_capabilities,
                        artifact_context_windows,
                    )?,
                    ModelCapabilities::ImageGeneration(_) => Self::image_policy(discovered_model),
                    ModelCapabilities::Embeddings(embedding_capabilities) => {
                        Self::embeddings_policy(discovered_model, embedding_capabilities)
                    }
                    ModelCapabilities::Decisions(decision_capabilities) => {
                        Self::decisions_policy(discovered_model, decision_capabilities)
                    }
                    ModelCapabilities::Speech(_) => Self::speech_policy(discovered_model),
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
    ) -> Result<RuntimeModelPolicy, AstronomicalConfigError> {
        let resolved_model_config = user_config.resolved_discovered_model_config(
            &discovered_model.model_id,
            discovered_model.provider_model_id.as_deref(),
            chat_capabilities.context_window,
        )?;
        let worker_model_configuration = Self::worker_model_configuration(
            discovered_model,
            chat_capabilities,
            &resolved_model_config,
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
            worker_model_configuration,
        })
    }

    fn worker_model_configuration(
        discovered_model: &DiscoveredModel,
        chat_capabilities: &ChatModelCapabilities,
        resolved_model_config: &ResolvedModelConfig,
    ) -> WorkerModelConfiguration {
        WorkerModelConfiguration::Autoregressive(WorkerAutoregressiveModelConfiguration {
            model_id: discovered_model.model_id.clone(),
            maximum_context_tokens: chat_capabilities.context_window,
            // Worker policy carries model capability rather than a request default.
            maximum_output_tokens: chat_capabilities.max_output_tokens,
            chunking: worker_chunking_configuration(resolved_model_config.chunking()),
        })
    }

    fn image_policy(discovered_model: &DiscoveredModel) -> RuntimeModelPolicy {
        // The worker classifies the folder again and refuses it unless it is
        // this image family, so the policy names the family discovery found.
        let worker_model_configuration =
            WorkerModelConfiguration::ImageGeneration(WorkerImageGenerationModelConfiguration {
                model_id: discovered_model.model_id.clone(),
                model_family: discovered_model.model_family.as_str().to_owned(),
                artifact_revision: discovered_model.revision.clone(),
            });
        RuntimeModelPolicy {
            model_directory: discovered_model.model_directory.clone(),
            // Chat request defaults remain inert for a typed image worker policy.
            generation_defaults: RuntimeModelGenerationDefaults {
                maximum_output_tokens: 0,
                configured_maximum_output_tokens: None,
                temperature_thousandths: None,
                top_p_thousandths: None,
                repetition_penalty_thousandths: None,
            },
            configured_maximum_context_tokens: None,
            default_maximum_context_tokens: 0,
            configured_chunking_fields: Default::default(),
            worker_model_configuration,
        }
    }

    fn speech_policy(discovered_model: &DiscoveredModel) -> RuntimeModelPolicy {
        RuntimeModelPolicy {
            model_directory: discovered_model.model_directory.clone(),
            // Chat request defaults remain inert for a typed speech worker policy.
            generation_defaults: RuntimeModelGenerationDefaults {
                maximum_output_tokens: 0,
                configured_maximum_output_tokens: None,
                temperature_thousandths: None,
                top_p_thousandths: None,
                repetition_penalty_thousandths: None,
            },
            configured_maximum_context_tokens: None,
            default_maximum_context_tokens: 0,
            configured_chunking_fields: Default::default(),
            worker_model_configuration: WorkerModelConfiguration::Speech(
                WorkerSpeechModelConfiguration {
                    model_id: discovered_model.model_id.clone(),
                    model_family: discovered_model.model_family.as_str().to_owned(),
                    artifact_revision: discovered_model.revision.clone(),
                },
            ),
        }
    }

    fn embeddings_policy(
        discovered_model: &DiscoveredModel,
        embedding_capabilities: &config::EmbeddingModelCapabilities,
    ) -> RuntimeModelPolicy {
        RuntimeModelPolicy {
            model_directory: discovered_model.model_directory.clone(),
            // Chat request defaults remain inert for a typed embedding worker policy.
            generation_defaults: RuntimeModelGenerationDefaults {
                maximum_output_tokens: 0,
                configured_maximum_output_tokens: None,
                temperature_thousandths: None,
                top_p_thousandths: None,
                repetition_penalty_thousandths: None,
            },
            configured_maximum_context_tokens: None,
            default_maximum_context_tokens: 0,
            configured_chunking_fields: Default::default(),
            worker_model_configuration: WorkerModelConfiguration::Embeddings(
                WorkerEmbeddingModelConfiguration {
                    model_id: discovered_model.model_id.clone(),
                    model_family: discovered_model.model_family.as_str().to_owned(),
                    artifact_revision: discovered_model.revision.clone(),
                    vector_width: embedding_capabilities.vector_width,
                    maximum_input_tokens: embedding_capabilities.max_input_tokens,
                },
            ),
        }
    }

    fn decisions_policy(
        discovered_model: &DiscoveredModel,
        decision_capabilities: &config::DecisionModelCapabilities,
    ) -> RuntimeModelPolicy {
        RuntimeModelPolicy {
            model_directory: discovered_model.model_directory.clone(),
            // Chat request defaults remain inert for a typed decision worker policy.
            generation_defaults: RuntimeModelGenerationDefaults {
                maximum_output_tokens: 0,
                configured_maximum_output_tokens: None,
                temperature_thousandths: None,
                top_p_thousandths: None,
                repetition_penalty_thousandths: None,
            },
            configured_maximum_context_tokens: None,
            default_maximum_context_tokens: 0,
            configured_chunking_fields: Default::default(),
            worker_model_configuration: WorkerModelConfiguration::Decisions(
                WorkerDecisionModelConfiguration {
                    model_id: discovered_model.model_id.clone(),
                    model_family: discovered_model.model_family.as_str().to_owned(),
                    artifact_revision: discovered_model.revision.clone(),
                    maximum_tokens_per_question: decision_capabilities.maximum_tokens_per_question,
                },
            ),
        }
    }
}

#[cfg(test)]
#[path = "resolved_model_policy_catalog_tests.rs"]
mod tests;

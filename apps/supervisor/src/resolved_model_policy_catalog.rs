//! Builds the complete runtime policy catalog from resolved user configuration.
//!
//! Keeping tagged chat and image worker policy construction together ensures
//! discovery, configuration generations, and worker replacement compare the
//! same immutable execution policy.

use std::{collections::HashMap, sync::Arc};

use astronomical_config::{
    AstronomicalConfig, AstronomicalConfigError, ChatModelCapabilities, DiscoveredModel,
    ModelCapabilities, ModelFamily, ResolvedModelConfig,
};
use astronomical_ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerEmbeddingModelConfiguration,
    WorkerEmbeddingModelFamily, WorkerFlux2KleinModelConfiguration,
    WorkerImageGenerationModelFamily, WorkerModelConfiguration,
    WorkerQwenImage21ModelConfiguration,
};

use crate::runtime_model_policy::{
    runtime_model_generation_defaults, worker_chunking_configuration,
};
use crate::{
    RuntimeModelAccelerationAvailability, RuntimeModelGenerationDefaults, RuntimeModelPolicy,
};

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
        let resolved_model_config = user_config
            .resolved_model_config(&discovered_model.model_id, chat_capabilities.context_window)?;
        let (worker_model_configuration, acceleration_availability) =
            Self::worker_model_configuration(
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
            acceleration_availability,
            worker_model_configuration,
        })
    }

    fn worker_model_configuration(
        discovered_model: &DiscoveredModel,
        chat_capabilities: &ChatModelCapabilities,
        resolved_model_config: &ResolvedModelConfig,
    ) -> (
        WorkerModelConfiguration,
        RuntimeModelAccelerationAvailability,
    ) {
        let acceleration_availability = RuntimeModelAccelerationAvailability {
            configured_mtp_enabled: resolved_model_config.configured_mtp_enabled(),
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
            }),
            acceleration_availability,
        )
    }

    fn image_policy(discovered_model: &DiscoveredModel) -> RuntimeModelPolicy {
        let worker_model_configuration = match discovered_model.model_family {
            ModelFamily::Flux2Klein => {
                WorkerModelConfiguration::Flux2Klein(WorkerFlux2KleinModelConfiguration {
                    model_id: discovered_model.model_id.clone(),
                    model_family: WorkerImageGenerationModelFamily::Flux2Klein,
                    artifact_revision: discovered_model.revision.clone(),
                })
            }
            ModelFamily::QwenImage21 => {
                WorkerModelConfiguration::QwenImage21(WorkerQwenImage21ModelConfiguration {
                    model_id: discovered_model.model_id.clone(),
                    model_family: WorkerImageGenerationModelFamily::QwenImage21,
                    artifact_revision: discovered_model.revision.clone(),
                })
            }
            // An image capability requires one of the image families above; a discovered
            // directory that classifies otherwise fails at worker selection instead of
            // silently receiving a Flux identity it never verified.
            ModelFamily::Qwen3_5
            | ModelFamily::Qwen4Exp
            | ModelFamily::Laguna
            | ModelFamily::DeepSeekV4
            | ModelFamily::K2HorizonMoVA
            | ModelFamily::ModernBert => {
                WorkerModelConfiguration::Flux2Klein(WorkerFlux2KleinModelConfiguration {
                    model_id: discovered_model.model_id.clone(),
                    model_family: WorkerImageGenerationModelFamily::Flux2Klein,
                    artifact_revision: discovered_model.revision.clone(),
                })
            }
        };
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
            worker_model_configuration,
        }
    }

    fn embeddings_policy(
        discovered_model: &DiscoveredModel,
        embedding_capabilities: &astronomical_config::EmbeddingModelCapabilities,
    ) -> RuntimeModelPolicy {
        RuntimeModelPolicy {
            model_directory: discovered_model.model_directory.clone(),
            // Chat request defaults remain inert for a typed embedding worker policy.
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
            worker_model_configuration: WorkerModelConfiguration::Embeddings(
                WorkerEmbeddingModelConfiguration {
                    model_id: discovered_model.model_id.clone(),
                    model_family: WorkerEmbeddingModelFamily::ModernBert,
                    artifact_revision: discovered_model.revision.clone(),
                    vector_width: embedding_capabilities.vector_width,
                    maximum_input_tokens: embedding_capabilities.max_input_tokens,
                },
            ),
        }
    }
}

//! Per-model execution policy carried with each model swap.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{WorkerChunkingConfiguration, WorkerSpeculativePrefillConfiguration};

/// Path-free speculative-prefill policy acknowledged after model binding.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSpeculativePrefillRuntimeConfiguration {
    pub draft_model_id: String,
    pub minimum_prompt_tokens: u32,
    pub keep_percentage: u32,
}

/// Identity and resolved directory for an explicitly configured auxiliary model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerAuxiliaryModelConfiguration {
    pub model_id: String,
    pub model_directory: PathBuf,
}

/// Complete autoregressive execution policy for one canonical requestable model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerAutoregressiveModelConfiguration {
    pub model_id: String,
    /// Effective prompt-plus-output context capability.
    pub maximum_context_tokens: u32,
    /// Independent output capability; request defaults remain supervisor-owned.
    pub maximum_output_tokens: u32,
    pub chunking: WorkerChunkingConfiguration,
    pub mtp_draft_depth: Option<u8>,
    pub mtp_head_model: Option<WorkerAuxiliaryModelConfiguration>,
    pub speculative_prefill: Option<WorkerSpeculativePrefillConfiguration>,
}

/// Path-free autoregressive policy acknowledged after model binding.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerLoadedAutoregressiveModelRuntimeConfiguration {
    pub model_id: String,
    /// Effective prompt-plus-output context capability.
    pub maximum_context_tokens: u32,
    /// Independent output capability, not the configured request default.
    pub maximum_output_tokens: u32,
    pub chunking: WorkerChunkingConfiguration,
    pub mtp_draft_depth: Option<u8>,
    pub mtp_head_model_id: Option<String>,
    pub speculative_prefill_enabled: bool,
    pub speculative_prefill: Option<WorkerSpeculativePrefillRuntimeConfiguration>,
}

/// Typed image profile identifier carried without autoregressive placeholders.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerImageGenerationModelFamily {
    Flux2Klein,
}

/// Exact FLUX artifact identity required by the selected image profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerFlux2KleinModelConfiguration {
    pub model_id: String,
    pub model_family: WorkerImageGenerationModelFamily,
    pub artifact_revision: String,
}

/// Complete effective execution policy for one canonical requestable model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(
    tag = "kind",
    content = "configuration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum WorkerModelConfiguration {
    Autoregressive(WorkerAutoregressiveModelConfiguration),
    Flux2Klein(WorkerFlux2KleinModelConfiguration),
}

/// Path-free loaded-model policy acknowledged to the supervisor and local status API.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(
    tag = "kind",
    content = "configuration",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum WorkerLoadedModelRuntimeConfiguration {
    Autoregressive(WorkerLoadedAutoregressiveModelRuntimeConfiguration),
    Flux2Klein(WorkerFlux2KleinModelConfiguration),
}

impl WorkerModelConfiguration {
    /// Returns the canonical requestable model identity for either execution family.
    #[must_use]
    pub fn model_id(&self) -> &str {
        match self {
            Self::Autoregressive(configuration) => &configuration.model_id,
            Self::Flux2Klein(configuration) => &configuration.model_id,
        }
    }

    /// Returns chat policy only when this model owns autoregressive execution.
    #[must_use]
    pub const fn autoregressive(&self) -> Option<&WorkerAutoregressiveModelConfiguration> {
        match self {
            Self::Autoregressive(configuration) => Some(configuration),
            Self::Flux2Klein(_) => None,
        }
    }

    /// Returns mutable chat policy only when this model owns autoregressive execution.
    #[must_use]
    pub const fn autoregressive_mut(
        &mut self,
    ) -> Option<&mut WorkerAutoregressiveModelConfiguration> {
        match self {
            Self::Autoregressive(configuration) => Some(configuration),
            Self::Flux2Klein(_) => None,
        }
    }

    /// Removes local auxiliary paths while retaining the exact policy bound by the worker.
    #[must_use]
    pub fn runtime_configuration(&self) -> WorkerLoadedModelRuntimeConfiguration {
        match self {
            Self::Autoregressive(configuration) => {
                WorkerLoadedModelRuntimeConfiguration::Autoregressive(
                    WorkerLoadedAutoregressiveModelRuntimeConfiguration {
                        model_id: configuration.model_id.clone(),
                        maximum_context_tokens: configuration.maximum_context_tokens,
                        maximum_output_tokens: configuration.maximum_output_tokens,
                        chunking: configuration.chunking.clone(),
                        mtp_draft_depth: configuration.mtp_draft_depth,
                        // A configured relationship is not effective until a model factory
                        // validates and binds the auxiliary artifact into the execution graph.
                        mtp_head_model_id: None,
                        speculative_prefill_enabled: configuration.speculative_prefill.is_some(),
                        speculative_prefill: configuration.speculative_prefill.as_ref().and_then(
                            |speculative_prefill| {
                                speculative_prefill
                                    .draft_model_id
                                    .as_ref()
                                    .map(|draft_model_id| {
                                        WorkerSpeculativePrefillRuntimeConfiguration {
                                            draft_model_id: draft_model_id.clone(),
                                            minimum_prompt_tokens: speculative_prefill
                                                .minimum_prompt_tokens,
                                            keep_percentage: speculative_prefill.keep_percentage,
                                        }
                                    })
                            },
                        ),
                    },
                )
            }
            Self::Flux2Klein(configuration) => {
                WorkerLoadedModelRuntimeConfiguration::Flux2Klein(configuration.clone())
            }
        }
    }
}

impl WorkerLoadedModelRuntimeConfiguration {
    /// Returns the canonical requestable model identity for either execution family.
    #[must_use]
    pub fn model_id(&self) -> &str {
        match self {
            Self::Autoregressive(configuration) => &configuration.model_id,
            Self::Flux2Klein(configuration) => &configuration.model_id,
        }
    }

    /// Returns chat runtime policy only when this model owns autoregressive execution.
    #[must_use]
    pub const fn autoregressive(
        &self,
    ) -> Option<&WorkerLoadedAutoregressiveModelRuntimeConfiguration> {
        match self {
            Self::Autoregressive(configuration) => Some(configuration),
            Self::Flux2Klein(_) => None,
        }
    }
}

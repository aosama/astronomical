//! Per-model execution policy carried with each model swap.

use serde::{Deserialize, Serialize};

use crate::WorkerChunkingConfiguration;
use crate::exhaustive_table::exhaustive_table;

/// Complete autoregressive execution policy for one canonical requestable model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerAutoregressiveModelConfiguration {
    pub model_id: String,
    /// Effective prompt-plus-output context capability.
    pub maximum_context_tokens: u32,
    /// Independent output capability; request defaults remain daemon-owned.
    pub maximum_output_tokens: u32,
    pub chunking: WorkerChunkingConfiguration,
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
}

/// Typed image profile identifier carried without autoregressive placeholders.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerImageGenerationModelFamily {
    Flux2Klein,
    QwenImage21,
}

/// Typed embedding profile identifier carried without autoregressive placeholders.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerEmbeddingModelFamily {
    ModernBert,
}

/// Exact embedding artifact identity required by the selected profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerEmbeddingModelConfiguration {
    pub model_id: String,
    pub model_family: WorkerEmbeddingModelFamily,
    pub artifact_revision: String,
    /// Native vector width produced by the loaded artifact.
    pub vector_width: u32,
    /// Maximum accepted prompt tokens per embedding input.
    pub maximum_input_tokens: u32,
}

/// Typed decision-model family identifier.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerDecisionModelFamily {
    /// A ModernBERT encoder that scores one marker token per option.
    Laya,
    /// A Qwen3.5 backbone whose pointer head scores each option's closing
    /// token against its question's decision token.
    Kev,
}

/// Exact decision artifact identity required by the selected profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerDecisionModelConfiguration {
    pub model_id: String,
    pub model_family: WorkerDecisionModelFamily,
    pub artifact_revision: String,
    /// The longest sequence one question may occupy.
    pub maximum_tokens_per_question: u32,
}

/// Exact FLUX artifact identity required by the selected image profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerFlux2KleinModelConfiguration {
    pub model_id: String,
    pub model_family: WorkerImageGenerationModelFamily,
    pub artifact_revision: String,
}

/// Exact Qwen-Image-2.1 artifact identity required by the selected image profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerQwenImage21ModelConfiguration {
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
    QwenImage21(WorkerQwenImage21ModelConfiguration),
    Embeddings(WorkerEmbeddingModelConfiguration),
    Decisions(WorkerDecisionModelConfiguration),
}

/// Path-free loaded-model policy acknowledged to the daemon and local status API.
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
    QwenImage21(WorkerQwenImage21ModelConfiguration),
    Embeddings(WorkerEmbeddingModelConfiguration),
    Decisions(WorkerDecisionModelConfiguration),
}

impl WorkerModelConfiguration {
    /// Returns the canonical requestable model identity for either execution family.
    #[must_use]
    pub fn model_id(&self) -> &str {
        exhaustive_table!(
            self,
            configuration => &configuration.model_id,
            [
                Self::Autoregressive,
                Self::Flux2Klein,
                Self::QwenImage21,
                Self::Embeddings,
                Self::Decisions,
            ]
        )
    }

    /// Returns chat policy only when this model owns autoregressive execution.
    #[must_use]
    pub const fn autoregressive(&self) -> Option<&WorkerAutoregressiveModelConfiguration> {
        match self {
            Self::Autoregressive(configuration) => Some(configuration),
            Self::Flux2Klein(_)
            | Self::QwenImage21(_)
            | Self::Embeddings(_)
            | Self::Decisions(_) => None,
        }
    }

    /// Returns mutable chat policy only when this model owns autoregressive execution.
    #[must_use]
    pub const fn autoregressive_mut(
        &mut self,
    ) -> Option<&mut WorkerAutoregressiveModelConfiguration> {
        match self {
            Self::Autoregressive(configuration) => Some(configuration),
            Self::Flux2Klein(_)
            | Self::QwenImage21(_)
            | Self::Embeddings(_)
            | Self::Decisions(_) => None,
        }
    }

    /// Removes local auxiliary paths while retaining the exact policy bound by the worker.
    #[must_use]
    pub fn runtime_configuration(&self) -> WorkerLoadedModelRuntimeConfiguration {
        exhaustive_table!(
            self,
            configuration => configuration.into(),
            [
                Self::Autoregressive,
                Self::Flux2Klein,
                Self::QwenImage21,
                Self::Embeddings,
                Self::Decisions,
            ]
        )
    }
}

/// What the worker acknowledges for a loaded chat model: its policy without
/// anything that names a place on disk.
impl From<&WorkerAutoregressiveModelConfiguration> for WorkerLoadedModelRuntimeConfiguration {
    fn from(configuration: &WorkerAutoregressiveModelConfiguration) -> Self {
        Self::Autoregressive(WorkerLoadedAutoregressiveModelRuntimeConfiguration {
            model_id: configuration.model_id.clone(),
            maximum_context_tokens: configuration.maximum_context_tokens,
            maximum_output_tokens: configuration.maximum_output_tokens,
            chunking: configuration.chunking.clone(),
        })
    }
}

// The remaining families carry no path, so the loaded form is the
// configuration itself.
impl From<&WorkerFlux2KleinModelConfiguration> for WorkerLoadedModelRuntimeConfiguration {
    fn from(configuration: &WorkerFlux2KleinModelConfiguration) -> Self {
        Self::Flux2Klein(configuration.clone())
    }
}

impl From<&WorkerQwenImage21ModelConfiguration> for WorkerLoadedModelRuntimeConfiguration {
    fn from(configuration: &WorkerQwenImage21ModelConfiguration) -> Self {
        Self::QwenImage21(configuration.clone())
    }
}

impl From<&WorkerEmbeddingModelConfiguration> for WorkerLoadedModelRuntimeConfiguration {
    fn from(configuration: &WorkerEmbeddingModelConfiguration) -> Self {
        Self::Embeddings(configuration.clone())
    }
}

impl From<&WorkerDecisionModelConfiguration> for WorkerLoadedModelRuntimeConfiguration {
    fn from(configuration: &WorkerDecisionModelConfiguration) -> Self {
        Self::Decisions(configuration.clone())
    }
}

impl WorkerLoadedModelRuntimeConfiguration {
    /// Returns the canonical requestable model identity for either execution family.
    #[must_use]
    pub fn model_id(&self) -> &str {
        exhaustive_table!(
            self,
            configuration => &configuration.model_id,
            [
                Self::Autoregressive,
                Self::Flux2Klein,
                Self::QwenImage21,
                Self::Embeddings,
                Self::Decisions,
            ]
        )
    }

    /// Returns chat runtime policy only when this model owns autoregressive execution.
    #[must_use]
    pub const fn autoregressive(
        &self,
    ) -> Option<&WorkerLoadedAutoregressiveModelRuntimeConfiguration> {
        match self {
            Self::Autoregressive(configuration) => Some(configuration),
            Self::Flux2Klein(_)
            | Self::QwenImage21(_)
            | Self::Embeddings(_)
            | Self::Decisions(_) => None,
        }
    }
}

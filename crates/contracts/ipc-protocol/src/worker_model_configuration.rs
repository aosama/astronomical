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

/// Exact embedding artifact identity required by the selected profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerEmbeddingModelConfiguration {
    pub model_id: String,
    /// The family's identifier, as discovery labels it (`modernbert`). The
    /// worker serves the folder only when it classifies as this family.
    pub model_family: String,
    pub artifact_revision: String,
    /// Native vector width produced by the loaded artifact.
    pub vector_width: u32,
    /// Maximum accepted prompt tokens per embedding input.
    pub maximum_input_tokens: u32,
}

/// Exact decision artifact identity required by the selected profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerDecisionModelConfiguration {
    pub model_id: String,
    /// The family's identifier, as discovery labels it (`laya`, `kev` or
    /// `clef`). The worker serves the folder only when it classifies as this
    /// family.
    pub model_family: String,
    pub artifact_revision: String,
    /// The longest sequence one question may occupy.
    pub maximum_tokens_per_question: u32,
}

/// Exact speech artifact identity required by the selected profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSpeechModelConfiguration {
    pub model_id: String,
    /// The family's identifier, as discovery labels it (`kokoro` or
    /// `phonon`). The worker serves the folder only when it classifies as
    /// this family.
    pub model_family: String,
    pub artifact_revision: String,
}

/// Exact image artifact identity required by the selected image profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerImageGenerationModelConfiguration {
    pub model_id: String,
    /// The family's identifier, as discovery labels it (`flux2_klein` or
    /// `qwen_image_21`). The worker serves the folder only when it
    /// classifies as this family.
    pub model_family: String,
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
    ImageGeneration(WorkerImageGenerationModelConfiguration),
    Embeddings(WorkerEmbeddingModelConfiguration),
    Decisions(WorkerDecisionModelConfiguration),
    Speech(WorkerSpeechModelConfiguration),
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
    ImageGeneration(WorkerImageGenerationModelConfiguration),
    Embeddings(WorkerEmbeddingModelConfiguration),
    Decisions(WorkerDecisionModelConfiguration),
    Speech(WorkerSpeechModelConfiguration),
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
                Self::ImageGeneration,
                Self::Embeddings,
                Self::Decisions,
                Self::Speech,
            ]
        )
    }

    /// Returns chat policy only when this model owns autoregressive execution.
    #[must_use]
    pub const fn autoregressive(&self) -> Option<&WorkerAutoregressiveModelConfiguration> {
        match self {
            Self::Autoregressive(configuration) => Some(configuration),
            Self::ImageGeneration(_)
            | Self::Embeddings(_)
            | Self::Decisions(_)
            | Self::Speech(_) => None,
        }
    }

    /// Returns mutable chat policy only when this model owns autoregressive execution.
    #[must_use]
    pub const fn autoregressive_mut(
        &mut self,
    ) -> Option<&mut WorkerAutoregressiveModelConfiguration> {
        match self {
            Self::Autoregressive(configuration) => Some(configuration),
            Self::ImageGeneration(_)
            | Self::Embeddings(_)
            | Self::Decisions(_)
            | Self::Speech(_) => None,
        }
    }

    /// The family identifier this policy was resolved for. A chat policy
    /// names none, because every chat family takes the same configuration.
    #[must_use]
    pub fn model_family(&self) -> Option<&str> {
        match self {
            Self::Autoregressive(_) => None,
            Self::ImageGeneration(configuration) => Some(&configuration.model_family),
            Self::Embeddings(configuration) => Some(&configuration.model_family),
            Self::Decisions(configuration) => Some(&configuration.model_family),
            Self::Speech(configuration) => Some(&configuration.model_family),
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
                Self::ImageGeneration,
                Self::Embeddings,
                Self::Decisions,
                Self::Speech,
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
impl From<&WorkerImageGenerationModelConfiguration> for WorkerLoadedModelRuntimeConfiguration {
    fn from(configuration: &WorkerImageGenerationModelConfiguration) -> Self {
        Self::ImageGeneration(configuration.clone())
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

impl From<&WorkerSpeechModelConfiguration> for WorkerLoadedModelRuntimeConfiguration {
    fn from(configuration: &WorkerSpeechModelConfiguration) -> Self {
        Self::Speech(configuration.clone())
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
                Self::ImageGeneration,
                Self::Embeddings,
                Self::Decisions,
                Self::Speech,
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
            Self::ImageGeneration(_)
            | Self::Embeddings(_)
            | Self::Decisions(_)
            | Self::Speech(_) => None,
        }
    }
}

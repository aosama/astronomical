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

/// Complete effective execution policy for one canonical requestable model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelConfiguration {
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

/// Path-free loaded-model policy acknowledged to the supervisor and local status API.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerLoadedModelRuntimeConfiguration {
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

impl WorkerModelConfiguration {
    /// Removes local auxiliary paths while retaining the policy actually bound by the worker.
    #[must_use]
    pub fn runtime_configuration(&self) -> WorkerLoadedModelRuntimeConfiguration {
        WorkerLoadedModelRuntimeConfiguration {
            model_id: self.model_id.clone(),
            maximum_context_tokens: self.maximum_context_tokens,
            maximum_output_tokens: self.maximum_output_tokens,
            chunking: self.chunking.clone(),
            mtp_draft_depth: self.mtp_draft_depth,
            // A configured relationship is not effective until a model factory
            // validates and binds the auxiliary artifact into the execution graph.
            mtp_head_model_id: None,
            speculative_prefill_enabled: self.speculative_prefill.is_some(),
            speculative_prefill: self.speculative_prefill.as_ref().and_then(
                |speculative_prefill| {
                    speculative_prefill
                        .draft_model_id
                        .as_ref()
                        .map(
                            |draft_model_id| WorkerSpeculativePrefillRuntimeConfiguration {
                                draft_model_id: draft_model_id.clone(),
                                minimum_prompt_tokens: speculative_prefill.minimum_prompt_tokens,
                                keep_percentage: speculative_prefill.keep_percentage,
                            },
                        )
                },
            ),
        }
    }
}

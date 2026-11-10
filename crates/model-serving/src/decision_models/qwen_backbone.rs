//! A Qwen3.5 backbone loaded to be read, not sampled from: what Kev and Clef
//! share.
//!
//! Both load a Qwen3.5 checkpoint the way the chat engine does (validated by
//! the same validator, bound by the same loader, which reads both the
//! published Hugging Face layout and MLX conversions), prefill the input
//! that is only context, and read final hidden states from the part their
//! head looks at.

use std::path::Path;

use astronomical_decision_contract::DecisionFailure;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_bridge::MlxArray;

use crate::{
    Qwen3_5ArtifactValidator, Qwen3_5ModelChunkingConfiguration, Qwen3_5ResidentModel,
    RequestDecoderStateStack,
};

/// The KV-cache growth step and graph-submission intervals the chat engine
/// uses by default; decision reads are prefills of the same shapes.
const FULL_ATTENTION_KV_GROWTH_TOKENS: u32 = 256;
const PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 0;
const SSD_PAGING_PREFILL_LAYER_INTERVAL: u32 = 1;
const SSD_PAGING_GENERATION_LAYER_INTERVAL: u32 = 3;
/// A decision model generates nothing; validation still asks for an output
/// budget, and one token is the smallest it accepts.
const VALIDATION_OUTPUT_TOKENS: u32 = 1;
/// Context is prefilled in runs of this many tokens, the chat engine's
/// standard chunk: one graph per run bounds the forward's workspace however
/// long the context is.
pub(crate) const PREFILL_CHUNK_TOKENS: usize = 2_048;

/// One loaded backbone.
pub(crate) struct QwenDecisionBackbone {
    pub(crate) model: Qwen3_5ResidentModel,
}

impl QwenDecisionBackbone {
    pub(crate) fn load(
        model_directory: &Path,
        memory_limits: MlxMemoryLimits,
    ) -> Result<Self, DecisionFailure> {
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(model_directory, VALIDATION_OUTPUT_TOKENS)
            .map_err(|validation_error| {
                DecisionFailure::fatal_execution(&format!(
                    "the decision model's backbone is not a valid Qwen3.5 checkpoint: {validation_error}"
                ))
            })?;
        let model_chunking = Qwen3_5ModelChunkingConfiguration::new(
            FULL_ATTENTION_KV_GROWTH_TOKENS,
            PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
            SSD_PAGING_PREFILL_LAYER_INTERVAL,
            SSD_PAGING_GENERATION_LAYER_INTERVAL,
        )
        .map_err(|chunking_error| backbone_failure("chunking", &chunking_error.to_string()))?;
        let runtime = MlxRuntime::initialize(memory_limits).map_err(|runtime_error| {
            backbone_failure("runtime start", &runtime_error.to_string())
        })?;
        let mut model = Qwen3_5ResidentModel::load(
            runtime,
            validated_artifact,
            model_directory,
            model_chunking,
        )
        .map_err(|load_error| backbone_failure("load", &load_error.to_string()))?;
        model
            .materialize_complete_weights_matching_production()
            .map_err(|load_error| backbone_failure("load", &load_error.to_string()))?;
        Ok(Self { model })
    }

    pub(crate) fn fresh_state(&self) -> Result<RequestDecoderStateStack, DecisionFailure> {
        self.model
            .empty_request_decoder_state_matching_loaded_weights(
                FULL_ATTENTION_KV_GROWTH_TOKENS as i32,
            )
            .map_err(|state_error| backbone_failure("state allocation", &state_error.to_string()))
    }

    /// Prefills `token_ids` from `starting_position`, in standard runs.
    pub(crate) fn prefill(
        &self,
        token_ids: &[u32],
        starting_position: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<(), DecisionFailure> {
        let mut run_start = 0;
        for token_run in token_ids.chunks(PREFILL_CHUNK_TOKENS) {
            self.model
                .prefill_chunk(
                    token_run,
                    starting_position + run_start as u32,
                    request_decoder_state,
                )
                .map_err(|forward_error| backbone_failure("prefill", &forward_error.to_string()))?;
            run_start += token_run.len();
        }
        Ok(())
    }

    /// The final hidden states of `token_ids` continuing the state.
    pub(crate) fn read_hidden_states(
        &self,
        token_ids: &[u32],
        starting_position: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, DecisionFailure> {
        self.model
            .final_normalized_hidden_states(token_ids, starting_position, request_decoder_state)
            .map_err(|forward_error| backbone_failure("readout", &forward_error.to_string()))
    }
}

/// A backbone operation's failure, bounded and free of local paths: the
/// reasons come from the engine, which never puts a path in its errors.
fn backbone_failure(operation: &str, reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!("decision backbone {operation} failed: {reason}"))
}

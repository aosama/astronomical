//! A Qwen3.5 backbone loaded to be read, not sampled from: what Kev and Clef
//! share.
//!
//! Both load a Qwen3.5 checkpoint the way the chat engine does (validated by
//! the same validator, bound by the same loader, which reads both the
//! published Hugging Face layout and MLX conversions), prefill the input
//! that is only context, and read final hidden states from the part their
//! head looks at. The backbone is the chat engine's own model with every
//! decoder layer kept in memory, so a decision reads the forward serving
//! runs.
//!
//! A folder written by a whole-model training also holds
//! `decision_adapters.safetensors`: add-ons for the backbone's dense
//! projections, checked against the loaded layers and put in force for
//! every forward pass. A folder without the file serves the shipped
//! backbone exactly.

use std::path::Path;
use std::rc::Rc;

use decision_contract::DecisionFailure;
use mlx_bridge::MlxArray;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::decision_models::decision_adapters::{
    check_adapter_targets, read_decision_adapters, unusable_adapters,
};
use crate::linear_adapters::LinearAdapters;
use crate::qwen3_5_core::model::layer_additions::Qwen3_5LayerAdditions;
use crate::{
    Qwen3_5ArtifactValidator, Qwen3_5StreamingModel, RequestDecoderStateStack,
    TextModelChunkingConfiguration,
};

/// The `astro_decoder_state_cache` KV growth step and graph-submission intervals the chat engine
/// uses by default; decision reads are prefills of the same shapes.
const FULL_ATTENTION_KV_GROWTH_TOKENS: u32 = 256;
const PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 0;
const SSD_PAGING_GENERATION_LAYER_INTERVAL: u32 = 3;
/// A decision model generates nothing; validation still asks for an output
/// budget, and one token is the smallest it accepts.
const VALIDATION_OUTPUT_TOKENS: u32 = 1;
/// Context is prefilled in runs of this many tokens, the chat engine's
/// standard chunk: one graph per run bounds the forward's workspace however
/// long the context is.
pub const PREFILL_CHUNK_TOKENS: usize = 2_048;

/// One loaded backbone.
pub struct QwenDecisionBackbone {
    pub model: Qwen3_5StreamingModel,
    /// Whether the folder's add-ons are in force.
    pub has_add_ons: bool,
}

/// A loaded decision model that reads a Qwen3.5 backbone (Kev, Clef), as
/// work that must outlive one call, such as a layer MLX recomputes during
/// the backward pass, reaches it.
pub trait HoldsDecisionBackbone {
    fn decision_backbone(&self) -> &QwenDecisionBackbone;
}

impl QwenDecisionBackbone {
    pub fn load(
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
        let model_chunking = TextModelChunkingConfiguration::new(
            FULL_ATTENTION_KV_GROWTH_TOKENS,
            PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
            SSD_PAGING_GENERATION_LAYER_INTERVAL,
        )
        .map_err(|chunking_error| backbone_failure("chunking", &chunking_error.to_string()))?;
        let runtime = MlxRuntime::initialize(memory_limits).map_err(|runtime_error| {
            backbone_failure("runtime start", &runtime_error.to_string())
        })?;
        let model = Qwen3_5StreamingModel::load(
            runtime,
            validated_artifact,
            model_directory,
            model_chunking,
        )
        .map_err(|load_error| backbone_failure("load", &load_error.to_string()))?;
        // Every layer stays in memory: a training runs each one many times,
        // and a decision reads a whole state, so reading layers from the SSD
        // on every forward would cost far more than holding them.
        model
            .materialize_and_keep_every_dense_layer()
            .map_err(|load_error| backbone_failure("load", &load_error.to_string()))?;
        let has_add_ons = put_folder_add_ons_in_force(&model, model_directory)?;
        Ok(Self { model, has_add_ons })
    }

    pub fn fresh_state(&self) -> Result<RequestDecoderStateStack, DecisionFailure> {
        self.model
            .empty_request_decoder_state_matching_loaded_weights(
                FULL_ATTENTION_KV_GROWTH_TOKENS as i32,
            )
            .map_err(|state_error| backbone_failure("state allocation", &state_error.to_string()))
    }

    /// Prefills `token_ids` from `starting_position`, in standard runs.
    pub fn prefill(
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
    pub fn read_hidden_states(
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

/// The runtime memory limit a training sizes its passes by.
pub fn backbone_memory_limit_bytes(
    backbone: &QwenDecisionBackbone,
) -> Result<usize, DecisionFailure> {
    backbone
        .model
        .runtime()
        .configured_memory_limit_bytes()
        .map_err(|limit_error| {
            DecisionFailure::fatal_execution(&format!(
                "the training's memory limit could not be read: {limit_error}"
            ))
        })
}

/// Reads, checks and uploads the folder's add-ons, if it has any, and puts
/// them in force. Returns whether there were any.
fn put_folder_add_ons_in_force(
    model: &Qwen3_5StreamingModel,
    model_directory: &Path,
) -> Result<bool, DecisionFailure> {
    let decision_adapters = read_decision_adapters(model_directory)?;
    if decision_adapters.is_empty() {
        return Ok(false);
    }
    let adapter_targets = model
        .adapter_targets()
        .map_err(|target_error| unusable_adapters(&target_error.to_string()))?;
    check_adapter_targets(&decision_adapters, &adapter_targets)?;
    let adapters = LinearAdapters::upload(model.runtime(), &decision_adapters)
        .map_err(|upload_error| backbone_failure("add-on upload", &upload_error.to_string()))?;
    model.set_layer_additions(Some(Rc::new(Qwen3_5LayerAdditions {
        adapters,
        differentiable_recurrence: None,
    })));
    Ok(true)
}

/// A backbone operation's failure, bounded and free of local paths: the
/// reasons come from the engine, which never puts a path in its errors.
fn backbone_failure(operation: &str, reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!("decision backbone {operation} failed: {reason}"))
}

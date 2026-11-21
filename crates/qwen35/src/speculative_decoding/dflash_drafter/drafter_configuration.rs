//! A DFlash drafter's `config.json`, read strictly.
//!
//! Two generations share one design. DFlash (`DFlashDraftModel`) is a few
//! layers that attend to fused target hidden rows; DFlash2
//! (`DFlash2DraftModel`) adds a grouped dynamic convolution around each
//! sublayer and a candidate selector, exactly as z-lab's reference subclasses
//! the one from the other.
//!
//! The drafter is not a standalone model: it reads the target's embedding,
//! head and hidden-state taps. Everything that shapes its forward pass is
//! checked here once, and anything this port does not implement (logit
//! soft-capping, scaled embeddings or outputs, rope scaling) is refused by
//! name instead of being silently ignored.

use serde_json::Value;
use thiserror::Error;

use super::drafter_configuration_fields::{
    integer, integer_list, layer_attention, number, read_generation, refuse_unimplemented_fields,
    rope_theta,
};

/// Candidates a drafter without a selector keeps at each slot for the draft
/// tree to search: the selector's own top-k on every published DFlash2
/// checkpoint, and TensorFold's `head_topk`.
pub const HEAD_ONLY_CANDIDATE_COUNT: i32 = 16;

/// Which drafter design a checkpoint declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterGeneration {
    /// `DFlashDraftModel`: attention and MLP layers only.
    DFlash,
    /// `DFlash2DraftModel`: adds the convolutions and the candidate selector.
    DFlash2,
}

/// How one drafter layer attends to the fused target context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterLayerAttention {
    /// Keeps only the last `sliding_window - 1` context rows.
    SlidingWindow,
    /// Keeps every context row.
    Full,
}

/// DFlash2's grouped dynamic causal convolution, the same in every layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedConvolutionShape {
    pub kernel_size: i32,
    pub group_size: i32,
    /// Channels convolved: the drafter's hidden width.
    pub channel_count: i32,
}

impl GroupedConvolutionShape {
    /// Convolution groups across the channels.
    pub fn group_count(&self) -> i32 {
        self.channel_count / self.group_size
    }
}

/// DFlash2's candidate selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateSelectorShape {
    pub rank: i32,
    /// Candidates kept per proposal position before the selector walks a path.
    pub candidate_count: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DFlashDrafterConfiguration {
    pub generation: DrafterGeneration,
    pub hidden_size: i32,
    pub layer_count: i32,
    pub attention_head_count: i32,
    pub key_value_head_count: i32,
    pub head_dimension: i32,
    pub intermediate_size: i32,
    pub vocabulary_size: i32,
    pub rms_norm_epsilon: f32,
    pub rope_theta: f32,
    pub sliding_window: i32,
    pub layer_attention: Vec<DrafterLayerAttention>,
    /// The checkpoint's `is_causal`, or None when it does not say.
    declared_block_causality: Option<bool>,
    /// Rows a drafting pass fills: the anchor token plus `block_size - 1`
    /// masked slots, which become the proposals.
    pub block_size: i32,
    pub mask_token_id: u32,
    /// Target decoder layers whose outputs (counted from 0) feed the drafter.
    pub target_tap_layer_ids: Vec<i32>,
    pub target_layer_count: i32,
    /// Present for DFlash2 only.
    pub convolution: Option<GroupedConvolutionShape>,
    /// Present for DFlash2 only.
    pub candidate_selector: Option<CandidateSelectorShape>,
}

/// Why a drafter configuration was refused.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum DFlashDrafterConfigurationError {
    #[error("the drafter declares neither DFlashDraftModel nor DFlash2DraftModel")]
    Architecture,
    #[error("the drafter configuration is missing or has a malformed `{field}`")]
    Field { field: &'static str },
    #[error("the drafter configuration sets `{field}`, which this drafter does not implement")]
    Unsupported { field: &'static str },
    #[error("the drafter configuration is inconsistent: {description}")]
    Inconsistent { description: String },
}

impl DFlashDrafterConfiguration {
    pub fn from_config_document(document: &Value) -> Result<Self, DFlashDrafterConfigurationError> {
        let generation = read_generation(document)?;
        refuse_unimplemented_fields(document)?;
        let dflash =
            document
                .get("dflash_config")
                .ok_or(DFlashDrafterConfigurationError::Field {
                    field: "dflash_config",
                })?;
        let model_shape = read_model_shape(document)?;
        let drafting_shape = read_drafting_shape(dflash, generation, model_shape.hidden_size)?;
        let configuration = Self::from_parts(generation, model_shape, drafting_shape);
        configuration.check_consistency()?;
        Ok(configuration)
    }

    fn from_parts(
        generation: DrafterGeneration,
        model_shape: ModelShape,
        drafting_shape: DraftingShape,
    ) -> Self {
        let second_generation_parts = drafting_shape.second_generation_parts;
        Self {
            generation,
            hidden_size: model_shape.hidden_size,
            layer_count: model_shape.layer_count,
            attention_head_count: model_shape.attention_head_count,
            key_value_head_count: model_shape.key_value_head_count,
            head_dimension: model_shape.head_dimension,
            intermediate_size: model_shape.intermediate_size,
            vocabulary_size: model_shape.vocabulary_size,
            rms_norm_epsilon: model_shape.rms_norm_epsilon,
            rope_theta: model_shape.rope_theta,
            sliding_window: model_shape.sliding_window,
            layer_attention: model_shape.layer_attention,
            declared_block_causality: model_shape.declared_block_causality,
            block_size: drafting_shape.block_size,
            mask_token_id: drafting_shape.mask_token_id,
            target_tap_layer_ids: drafting_shape.target_tap_layer_ids,
            target_layer_count: model_shape.target_layer_count,
            convolution: second_generation_parts.convolution,
            candidate_selector: second_generation_parts.candidate_selector,
        }
    }

    /// Width of the concatenated target taps the fusion projection reads.
    pub fn tap_width(&self) -> i32 {
        self.hidden_size * self.target_tap_layer_ids.len() as i32
    }

    /// Proposals the trained block holds: every block row after the anchor.
    /// A tree round drafts a longer block (`tree_block_row_count`).
    pub fn proposal_count(&self) -> i32 {
        self.block_size - 1
    }

    /// Whether block rows in layer `layer_index` see only earlier block rows.
    ///
    /// The reference's rule: a declared `is_causal` applies to every layer;
    /// without one, sliding-window layers are causal and full-attention layers
    /// see the whole block. Every published DFlash2 checkpoint declares false;
    /// the published DFlash checkpoints declare nothing, which makes their
    /// sliding layers causal and their one full layer two-way.
    pub fn layer_block_is_causal(&self, layer_index: usize) -> bool {
        match self.declared_block_causality {
            Some(declared) => declared,
            None => self.layer_attention[layer_index] == DrafterLayerAttention::SlidingWindow,
        }
    }

    /// Candidates kept at each slot for the draft tree: the selector's top-k,
    /// or `HEAD_ONLY_CANDIDATE_COUNT` for a drafter without one.
    pub fn lattice_candidate_count(&self) -> i32 {
        match self.candidate_selector {
            Some(selector) => selector.candidate_count,
            None => HEAD_ONLY_CANDIDATE_COUNT,
        }
    }

    /// Context rows the drafter can ever attend to, or `None` when a layer
    /// attends to the whole context. With every layer on a sliding window,
    /// rows older than the window are never read, so a caller holding
    /// target taps for the drafter needs only this many of the newest.
    pub fn context_row_limit(&self) -> Option<i32> {
        for layer_attention in &self.layer_attention {
            if *layer_attention == DrafterLayerAttention::Full {
                return None;
            }
        }
        Some(self.sliding_window - 1)
    }

    fn check_consistency(&self) -> Result<(), DFlashDrafterConfigurationError> {
        let convolution_does_not_divide = self
            .convolution
            .is_some_and(|convolution| self.hidden_size % convolution.group_size.max(1) != 0);
        let selector_is_out_of_range = self.candidate_selector.is_some_and(|selector| {
            selector.candidate_count < 1 || selector.candidate_count > self.vocabulary_size
        });
        let problems = [
            (
                self.layer_attention.len() != self.layer_count as usize,
                "layer_types must name every layer",
            ),
            (
                self.attention_head_count % self.key_value_head_count.max(1) != 0,
                "attention heads must be a multiple of key-value heads",
            ),
            (
                convolution_does_not_divide,
                "conv_group_size must divide hidden_size",
            ),
            (
                self.block_size < 2,
                "block_size must leave at least one proposal",
            ),
            (
                self.sliding_window < 2,
                "sliding_window must keep at least one context row",
            ),
            (
                selector_is_out_of_range,
                "selector_top_k must be within the vocabulary",
            ),
            (
                self.lattice_candidate_count() > self.vocabulary_size,
                "the vocabulary must hold the candidates kept at each slot",
            ),
            (
                self.mask_token_id as i32 >= self.vocabulary_size,
                "mask_token_id must be within the vocabulary",
            ),
            (
                self.target_tap_layer_ids.is_empty(),
                "target_layer_ids must name at least one layer",
            ),
            (
                self.target_tap_layer_ids
                    .iter()
                    .any(|layer_id| !(0..self.target_layer_count).contains(layer_id)),
                "target_layer_ids must be below num_target_layers",
            ),
        ];
        match problems.iter().find(|(is_problem, _)| *is_problem) {
            Some((_, description)) => Err(DFlashDrafterConfigurationError::Inconsistent {
                description: (*description).to_owned(),
            }),
            None => Ok(()),
        }
    }
}

/// The drafter's transformer shape, from the top level of `config.json`.
struct ModelShape {
    hidden_size: i32,
    layer_count: i32,
    attention_head_count: i32,
    key_value_head_count: i32,
    head_dimension: i32,
    intermediate_size: i32,
    vocabulary_size: i32,
    rms_norm_epsilon: f32,
    rope_theta: f32,
    sliding_window: i32,
    layer_attention: Vec<DrafterLayerAttention>,
    declared_block_causality: Option<bool>,
    target_layer_count: i32,
}

fn read_model_shape(document: &Value) -> Result<ModelShape, DFlashDrafterConfigurationError> {
    Ok(ModelShape {
        hidden_size: integer(document, "hidden_size")?,
        layer_count: integer(document, "num_hidden_layers")?,
        attention_head_count: integer(document, "num_attention_heads")?,
        key_value_head_count: integer(document, "num_key_value_heads")?,
        head_dimension: integer(document, "head_dim")?,
        intermediate_size: integer(document, "intermediate_size")?,
        vocabulary_size: integer(document, "vocab_size")?,
        rms_norm_epsilon: number(document, "rms_norm_eps")? as f32,
        rope_theta: rope_theta(document)?,
        sliding_window: integer(document, "sliding_window")?,
        layer_attention: layer_attention(document)?,
        declared_block_causality: document.get("is_causal").and_then(Value::as_bool),
        target_layer_count: integer(document, "num_target_layers")?,
    })
}

/// The drafting parameters, from `config.json`'s `dflash_config`.
struct DraftingShape {
    block_size: i32,
    mask_token_id: u32,
    target_tap_layer_ids: Vec<i32>,
    second_generation_parts: SecondGenerationParts,
}

fn read_drafting_shape(
    dflash: &Value,
    generation: DrafterGeneration,
    hidden_size: i32,
) -> Result<DraftingShape, DFlashDrafterConfigurationError> {
    Ok(DraftingShape {
        block_size: integer(dflash, "block_size")?,
        mask_token_id: integer(dflash, "mask_token_id")? as u32,
        target_tap_layer_ids: integer_list(dflash, "target_layer_ids")?,
        second_generation_parts: read_second_generation_parts(generation, dflash, hidden_size)?,
    })
}

/// DFlash2's convolutions and selector; both absent for DFlash, whose
/// reference ignores those fields even when a configuration carries them.
struct SecondGenerationParts {
    convolution: Option<GroupedConvolutionShape>,
    candidate_selector: Option<CandidateSelectorShape>,
}

fn read_second_generation_parts(
    generation: DrafterGeneration,
    dflash: &Value,
    hidden_size: i32,
) -> Result<SecondGenerationParts, DFlashDrafterConfigurationError> {
    if generation == DrafterGeneration::DFlash {
        return Ok(SecondGenerationParts {
            convolution: None,
            candidate_selector: None,
        });
    }
    Ok(SecondGenerationParts {
        convolution: Some(GroupedConvolutionShape {
            kernel_size: integer(dflash, "conv_kernel_size")?,
            group_size: integer(dflash, "conv_group_size")?,
            channel_count: hidden_size,
        }),
        candidate_selector: Some(CandidateSelectorShape {
            rank: integer(dflash, "selector_rank")?,
            candidate_count: integer(dflash, "selector_top_k")?,
        }),
    })
}

//! The DFlash2 drafter's `config.json`, read strictly.
//!
//! The drafter is not a standalone model: it reads the target's embedding,
//! head and hidden-state taps. Everything that shapes its forward pass is
//! checked here once, and anything this port does not implement (a causal
//! block, logit soft-capping, scaled embeddings or outputs, rope scaling) is
//! refused by name instead of being silently ignored.

use serde_json::Value;
use thiserror::Error;

const DFLASH2_ARCHITECTURE: &str = "DFlash2DraftModel";

/// How one drafter layer attends to the fused target context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterLayerAttention {
    /// Keeps only the last `sliding_window - 1` context rows.
    SlidingWindow,
    /// Keeps every context row.
    Full,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DFlash2DrafterConfiguration {
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
    /// Whether block rows see only earlier block rows. DFlash2 drafts the
    /// whole block at once, so its checkpoints set this false.
    pub block_is_causal: bool,
    /// Rows a drafting pass fills: the anchor token plus `block_size - 1`
    /// masked slots, which become the proposals.
    pub block_size: i32,
    pub mask_token_id: u32,
    /// Target decoder layers whose outputs (counted from 0) feed the drafter.
    pub target_tap_layer_ids: Vec<i32>,
    pub target_layer_count: i32,
    pub convolution_kernel_size: i32,
    pub convolution_group_size: i32,
    pub selector_rank: i32,
    /// Candidates kept per proposal position before the selector walks a path.
    pub selector_candidate_count: i32,
}

/// Why a drafter configuration was refused.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum DFlash2DrafterConfigurationError {
    #[error("the drafter declares no {DFLASH2_ARCHITECTURE} architecture")]
    Architecture,
    #[error("the drafter configuration is missing or has a malformed `{field}`")]
    Field { field: &'static str },
    #[error("the drafter configuration sets `{field}`, which this drafter does not implement")]
    Unsupported { field: &'static str },
    #[error("the drafter configuration is inconsistent: {description}")]
    Inconsistent { description: String },
}

impl DFlash2DrafterConfiguration {
    pub fn from_config_document(
        document: &Value,
    ) -> Result<Self, DFlash2DrafterConfigurationError> {
        require_architecture(document)?;
        refuse_unimplemented_fields(document)?;
        let dflash =
            document
                .get("dflash_config")
                .ok_or(DFlash2DrafterConfigurationError::Field {
                    field: "dflash_config",
                })?;
        let configuration =
            Self::from_parts(read_model_shape(document)?, read_drafting_shape(dflash)?);
        configuration.check_consistency()?;
        Ok(configuration)
    }

    fn from_parts(model_shape: ModelShape, drafting_shape: DraftingShape) -> Self {
        Self {
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
            block_is_causal: model_shape.block_is_causal,
            block_size: drafting_shape.block_size,
            mask_token_id: drafting_shape.mask_token_id,
            target_tap_layer_ids: drafting_shape.target_tap_layer_ids,
            target_layer_count: model_shape.target_layer_count,
            convolution_kernel_size: drafting_shape.convolution_kernel_size,
            convolution_group_size: drafting_shape.convolution_group_size,
            selector_rank: drafting_shape.selector_rank,
            selector_candidate_count: drafting_shape.selector_candidate_count,
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

    /// Convolution groups across the hidden width.
    pub fn convolution_group_count(&self) -> i32 {
        self.hidden_size / self.convolution_group_size
    }

    fn check_consistency(&self) -> Result<(), DFlash2DrafterConfigurationError> {
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
                self.hidden_size % self.convolution_group_size.max(1) != 0,
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
                self.selector_candidate_count < 1
                    || self.selector_candidate_count > self.vocabulary_size,
                "selector_top_k must be within the vocabulary",
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
            Some((_, description)) => Err(DFlash2DrafterConfigurationError::Inconsistent {
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
    block_is_causal: bool,
    target_layer_count: i32,
}

fn read_model_shape(document: &Value) -> Result<ModelShape, DFlash2DrafterConfigurationError> {
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
        block_is_causal: document
            .get("is_causal")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        target_layer_count: integer(document, "num_target_layers")?,
    })
}

/// The drafting parameters, from `config.json`'s `dflash_config`.
struct DraftingShape {
    block_size: i32,
    mask_token_id: u32,
    target_tap_layer_ids: Vec<i32>,
    convolution_kernel_size: i32,
    convolution_group_size: i32,
    selector_rank: i32,
    selector_candidate_count: i32,
}

fn read_drafting_shape(dflash: &Value) -> Result<DraftingShape, DFlash2DrafterConfigurationError> {
    Ok(DraftingShape {
        block_size: integer(dflash, "block_size")?,
        mask_token_id: integer(dflash, "mask_token_id")? as u32,
        target_tap_layer_ids: integer_list(dflash, "target_layer_ids")?,
        convolution_kernel_size: integer(dflash, "conv_kernel_size")?,
        convolution_group_size: integer(dflash, "conv_group_size")?,
        selector_rank: integer(dflash, "selector_rank")?,
        selector_candidate_count: integer(dflash, "selector_top_k")?,
    })
}

fn require_architecture(document: &Value) -> Result<(), DFlash2DrafterConfigurationError> {
    let declares_dflash2 = document
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|architectures| {
            architectures
                .iter()
                .any(|name| name.as_str() == Some(DFLASH2_ARCHITECTURE))
        });
    if declares_dflash2 {
        Ok(())
    } else {
        Err(DFlash2DrafterConfigurationError::Architecture)
    }
}

/// The reference model supports these, but the published drafters leave them
/// at their identity values; a checkpoint that changes one must not be served
/// with the wrong arithmetic.
fn refuse_unimplemented_fields(document: &Value) -> Result<(), DFlash2DrafterConfigurationError> {
    let dflash = document.get("dflash_config");
    if is_set_beyond_identity(Some(document), "final_logit_softcapping", 0.0)
        || is_set_beyond_identity(dflash, "final_logit_softcapping", 0.0)
    {
        return Err(DFlash2DrafterConfigurationError::Unsupported {
            field: "final_logit_softcapping",
        });
    }
    if is_set_beyond_identity(dflash, "input_embedding_scale", 1.0) {
        return Err(DFlash2DrafterConfigurationError::Unsupported {
            field: "input_embedding_scale",
        });
    }
    if is_set_beyond_identity(dflash, "output_multiplier", 1.0) {
        return Err(DFlash2DrafterConfigurationError::Unsupported {
            field: "output_multiplier",
        });
    }
    let rope_type = document
        .get("rope_parameters")
        .or_else(|| document.get("rope_scaling"))
        .and_then(|rope| rope.get("rope_type"))
        .and_then(Value::as_str);
    if rope_type.is_some_and(|rope_type| rope_type != "default") {
        return Err(DFlash2DrafterConfigurationError::Unsupported {
            field: "rope_parameters.rope_type",
        });
    }
    Ok(())
}

/// True when `field` is present with a value other than its identity; an
/// absent field or JSON null means the identity.
fn is_set_beyond_identity(container: Option<&Value>, field: &str, identity: f64) -> bool {
    match container
        .and_then(|object| object.get(field))
        .and_then(Value::as_f64)
    {
        Some(value) => value != identity,
        None => false,
    }
}

fn rope_theta(document: &Value) -> Result<f32, DFlash2DrafterConfigurationError> {
    document
        .get("rope_parameters")
        .and_then(|rope| rope.get("rope_theta"))
        .or_else(|| document.get("rope_theta"))
        .and_then(Value::as_f64)
        .map(|theta| theta as f32)
        .ok_or(DFlash2DrafterConfigurationError::Field {
            field: "rope_parameters.rope_theta",
        })
}

fn layer_attention(
    document: &Value,
) -> Result<Vec<DrafterLayerAttention>, DFlash2DrafterConfigurationError> {
    let field = DFlash2DrafterConfigurationError::Field {
        field: "layer_types",
    };
    let layer_types = document
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or(field.clone())?;
    let mut attention_kinds = Vec::with_capacity(layer_types.len());
    for layer_type in layer_types {
        attention_kinds.push(match layer_type.as_str() {
            Some("sliding_attention") => DrafterLayerAttention::SlidingWindow,
            Some("full_attention") => DrafterLayerAttention::Full,
            _ => return Err(field),
        });
    }
    Ok(attention_kinds)
}

fn integer(
    container: &Value,
    field: &'static str,
) -> Result<i32, DFlash2DrafterConfigurationError> {
    container
        .get(field)
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .filter(|value| *value >= 0)
        .ok_or(DFlash2DrafterConfigurationError::Field { field })
}

fn number(container: &Value, field: &'static str) -> Result<f64, DFlash2DrafterConfigurationError> {
    container
        .get(field)
        .and_then(Value::as_f64)
        .ok_or(DFlash2DrafterConfigurationError::Field { field })
}

fn integer_list(
    container: &Value,
    field: &'static str,
) -> Result<Vec<i32>, DFlash2DrafterConfigurationError> {
    let malformed = DFlash2DrafterConfigurationError::Field { field };
    let values = container
        .get(field)
        .and_then(Value::as_array)
        .ok_or(malformed.clone())?;
    let mut integers = Vec::with_capacity(values.len());
    for value in values {
        integers.push(
            value
                .as_i64()
                .and_then(|integer| i32::try_from(integer).ok())
                .ok_or(malformed.clone())?,
        );
    }
    Ok(integers)
}

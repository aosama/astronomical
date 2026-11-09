//! Parsed and validated DFlash draft-model configuration.
//!
//! The DFlash draft checkpoint (`architectures: ["DFlashDraftModel"]`) is a
//! companion artifact that only runs beside its target model. Its `config.json`
//! carries one `dflash_config` block (block size, mask token, target-layer
//! taps) plus the geometry of a small dense Qwen3-style trunk. Everything in
//! this file is pure parsing and validation: no graphics-processor work, no
//! file access beyond the bytes handed in, so every rule it enforces is unit
//! testable in isolation.

use thiserror::Error;

/// How one draft layer attends inside a speculation block.
///
/// Mirrors the reference runtime's layer resolution: sliding-window layers
/// stay causal within the block (a mask position never sees a later mask
/// position), while full-attention layers attend bidirectionally so the
/// block's positions can refine each other in parallel — the block-diffusion
/// heritage of the drafting scheme. Every layer always sees the target-derived
/// context keys and values, bounded by its window when sliding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashDraftLayerAttentionKind {
    SlidingWindowCausal { window_size: i32 },
    FullBidirectional,
}

/// One validated DFlash draft configuration.
#[derive(Debug, Clone)]
pub struct DFlashDraftConfiguration {
    hidden_size: i32,
    intermediate_size: i32,
    layer_count: i32,
    attention_head_count: i32,
    key_value_head_count: i32,
    head_dimension: i32,
    rms_norm_epsilon: f32,
    rope_theta: f32,
    vocabulary_size: i32,
    maximum_position_count: i32,
    mask_token_id: i32,
    draft_block_size: i32,
    target_layer_ids: Vec<i32>,
    target_layer_count: i32,
    layer_attention_kinds: Vec<DFlashDraftLayerAttentionKind>,
}

impl DFlashDraftConfiguration {
    /// Parses and validates one draft `config.json` document.
    pub fn from_config_value(
        config_document: &serde_json::Value,
    ) -> Result<Self, DFlashDraftConfigurationError> {
        require_draft_architecture(config_document)?;
        let hidden_size = required_i32(config_document, "hidden_size")?;
        let attention_head_count = required_i32(config_document, "num_attention_heads")?;
        let dflash_config = config_document.get("dflash_config").ok_or(
            DFlashDraftConfigurationError::MissingField {
                field: "dflash_config",
            },
        )?;
        let draft_configuration = Self {
            hidden_size,
            intermediate_size: required_i32(config_document, "intermediate_size")?,
            layer_count: required_i32(config_document, "num_hidden_layers")?,
            attention_head_count,
            key_value_head_count: required_i32(config_document, "num_key_value_heads")?,
            // A head count that is not positive is refused below; max(1)
            // only keeps this default from dividing by zero first.
            head_dimension: optional_i32(config_document, "head_dim")?
                .unwrap_or(hidden_size / attention_head_count.max(1)),
            rms_norm_epsilon: required_f32(config_document.get("rms_norm_eps"), "rms_norm_eps")?,
            rope_theta: required_f32(
                config_document
                    .get("rope_parameters")
                    .and_then(|rope_parameters| rope_parameters.get("rope_theta")),
                "rope_parameters.rope_theta",
            )?,
            vocabulary_size: required_i32(config_document, "vocab_size")?,
            maximum_position_count: required_i32(config_document, "max_position_embeddings")?,
            mask_token_id: required_i32(dflash_config, "mask_token_id")?,
            draft_block_size: required_i32(dflash_config, "block_size")?,
            target_layer_ids: parse_target_layer_ids(dflash_config)?,
            target_layer_count: required_i32(config_document, "num_target_layers")?,
            layer_attention_kinds: Vec::new(),
        };
        draft_configuration.validate()?;
        let layer_attention_kinds =
            resolve_layer_attention_kinds(config_document, draft_configuration.layer_count)?;
        Ok(Self {
            layer_attention_kinds,
            ..draft_configuration
        })
    }

    /// Every rule that relates parsed fields to each other, checked before
    /// the per-layer attention kinds, which rely on a positive layer count.
    fn validate(&self) -> Result<(), DFlashDraftConfigurationError> {
        self.validate_geometry()?;
        Self::validate_target_taps(&self.target_layer_ids, self.target_layer_count)?;
        Self::validate_token_identifiers(self.mask_token_id, self.vocabulary_size)?;
        let positive_draft_scalars = [
            (
                self.draft_block_size > 0,
                "dflash_config.block_size must be positive",
            ),
            (self.rms_norm_epsilon > 0.0, "rms_norm_eps must be positive"),
            (
                self.rope_theta > 0.0,
                "rope_parameters.rope_theta must be positive",
            ),
        ];
        for (is_positive, description) in positive_draft_scalars {
            if !is_positive {
                return Err(DFlashDraftConfigurationError::InvalidGeometry { description });
            }
        }
        Ok(())
    }

    fn validate_geometry(&self) -> Result<(), DFlashDraftConfigurationError> {
        let positive_dimensions = [
            (self.hidden_size, "hidden_size must be positive"),
            // The MLP projections are shaped [intermediate_size, hidden];
            // a zero width would only surface later as a shape mismatch.
            (self.intermediate_size, "intermediate_size must be positive"),
            (self.layer_count, "num_hidden_layers must be positive"),
            (
                self.attention_head_count,
                "num_attention_heads must be positive",
            ),
            (
                self.key_value_head_count,
                "num_key_value_heads must be positive",
            ),
            (self.head_dimension, "head_dim must be positive"),
        ];
        for (dimension, description) in positive_dimensions {
            if dimension <= 0 {
                return Err(DFlashDraftConfigurationError::InvalidGeometry { description });
            }
        }
        if self.attention_head_count % self.key_value_head_count != 0 {
            return Err(DFlashDraftConfigurationError::InvalidGeometry {
                description: "num_attention_heads must be a multiple of num_key_value_heads for grouped-query attention",
            });
        }
        // head_dim × heads may exceed hidden_size in this family (the shipped
        // draft is 32 × 128 = 4096 attention width against a 2048 trunk); the
        // projections intentionally bridge the two widths, so no equality
        // between them is required here. The loader's shape checks pin the
        // exact projection geometry instead.
        Ok(())
    }

    fn validate_target_taps(
        target_layer_ids: &[i32],
        target_layer_count: i32,
    ) -> Result<(), DFlashDraftConfigurationError> {
        if target_layer_ids.is_empty() {
            return Err(DFlashDraftConfigurationError::InvalidGeometry {
                description: "dflash_config.target_layer_ids must name at least one target layer",
            });
        }
        if target_layer_count <= 0 {
            return Err(DFlashDraftConfigurationError::InvalidGeometry {
                description: "num_target_layers must be positive",
            });
        }
        for &target_layer_id in target_layer_ids {
            if target_layer_id < 0 || target_layer_id >= target_layer_count {
                return Err(DFlashDraftConfigurationError::InvalidGeometry {
                    description: "dflash_config.target_layer_ids entries must fall inside the target layer range",
                });
            }
        }
        Ok(())
    }

    fn validate_token_identifiers(
        mask_token_id: i32,
        vocabulary_size: i32,
    ) -> Result<(), DFlashDraftConfigurationError> {
        if vocabulary_size <= 0 {
            return Err(DFlashDraftConfigurationError::InvalidGeometry {
                description: "vocab_size must be positive",
            });
        }
        if mask_token_id < 0 || mask_token_id >= vocabulary_size {
            return Err(DFlashDraftConfigurationError::InvalidGeometry {
                description: "dflash_config.mask_token_id must fall inside the vocabulary",
            });
        }
        Ok(())
    }

    #[must_use]
    pub fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    #[must_use]
    pub fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }

    #[must_use]
    pub fn layer_count(&self) -> i32 {
        self.layer_count
    }

    #[must_use]
    pub fn attention_head_count(&self) -> i32 {
        self.attention_head_count
    }

    #[must_use]
    pub fn key_value_head_count(&self) -> i32 {
        self.key_value_head_count
    }

    #[must_use]
    pub fn head_dimension(&self) -> i32 {
        self.head_dimension
    }

    #[must_use]
    pub fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_epsilon
    }

    #[must_use]
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    /// The draft shares the target's vocabulary; this size must match it.
    #[must_use]
    pub fn vocabulary_size(&self) -> i32 {
        self.vocabulary_size
    }

    #[must_use]
    pub fn maximum_position_count(&self) -> i32 {
        self.maximum_position_count
    }

    /// The token whose embedding row the draft places at every masked slot.
    #[must_use]
    pub fn mask_token_id(&self) -> i32 {
        self.mask_token_id
    }

    /// The training-time diffusion block size; serving may draft fewer tokens.
    #[must_use]
    pub fn draft_block_size(&self) -> i32 {
        self.draft_block_size
    }

    /// Target layer indices whose hidden states are concatenated as draft input.
    #[must_use]
    pub fn target_layer_ids(&self) -> &[i32] {
        &self.target_layer_ids
    }

    #[must_use]
    pub fn target_layer_count(&self) -> i32 {
        self.target_layer_count
    }

    #[must_use]
    pub fn layer_attention_kinds(&self) -> &[DFlashDraftLayerAttentionKind] {
        &self.layer_attention_kinds
    }

    /// Width of the concatenated target hidden states fed to the conditioning
    /// projection: one target hidden size per tap.
    #[must_use]
    pub fn target_conditioning_input_width(&self, target_hidden_size: i32) -> i32 {
        target_hidden_size * self.target_layer_ids.len() as i32
    }
}

/// Resolves per-layer attention kinds exactly as the reference runtime does:
/// `layer_types[i] == "sliding_attention"` selects a causal windowed layer
/// (window from `sliding_window`), anything else selects a bidirectional
/// full-attention layer.
fn resolve_layer_attention_kinds(
    config_document: &serde_json::Value,
    layer_count: i32,
) -> Result<Vec<DFlashDraftLayerAttentionKind>, DFlashDraftConfigurationError> {
    let layer_types = config_document
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or(DFlashDraftConfigurationError::MissingField {
            field: "layer_types",
        })?;
    if layer_types.len() != layer_count as usize {
        return Err(DFlashDraftConfigurationError::InvalidGeometry {
            description: "layer_types must name exactly num_hidden_layers layers",
        });
    }
    let sliding_window = required_i32(config_document, "sliding_window")?;
    if sliding_window <= 0 {
        return Err(DFlashDraftConfigurationError::InvalidGeometry {
            description: "sliding_window must be positive when layer_types names sliding layers",
        });
    }
    layer_types
        .iter()
        .map(|layer_type| {
            if layer_type.as_str() == Some("sliding_attention") {
                Ok(DFlashDraftLayerAttentionKind::SlidingWindowCausal {
                    window_size: sliding_window,
                })
            } else {
                Ok(DFlashDraftLayerAttentionKind::FullBidirectional)
            }
        })
        .collect()
}

fn require_draft_architecture(
    config_document: &serde_json::Value,
) -> Result<(), DFlashDraftConfigurationError> {
    let architectures = config_document
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .ok_or(DFlashDraftConfigurationError::MissingField {
            field: "architectures",
        })?;
    for architecture in architectures {
        if architecture.as_str() == Some("DFlashDraftModel") {
            return Ok(());
        }
    }
    Err(DFlashDraftConfigurationError::NotADFlashDraftModel)
}

/// The target layers whose hidden states the draft reads, in config order.
fn parse_target_layer_ids(
    dflash_config: &serde_json::Value,
) -> Result<Vec<i32>, DFlashDraftConfigurationError> {
    const FIELD: &str = "dflash_config.target_layer_ids";
    let listed_layer_ids = dflash_config
        .get("target_layer_ids")
        .and_then(serde_json::Value::as_array)
        .ok_or(DFlashDraftConfigurationError::MissingField { field: FIELD })?;
    let mut target_layer_ids = Vec::with_capacity(listed_layer_ids.len());
    for listed_layer_id in listed_layer_ids {
        let target_layer_id = listed_layer_id
            .as_i64()
            .and_then(|wide_layer_id| i32::try_from(wide_layer_id).ok())
            .ok_or(DFlashDraftConfigurationError::MalformedField { field: FIELD })?;
        target_layer_ids.push(target_layer_id);
    }
    Ok(target_layer_ids)
}

/// A JSON number narrowed to `f32`, the precision the draft computes in.
fn required_f32(
    field_value: Option<&serde_json::Value>,
    field: &'static str,
) -> Result<f32, DFlashDraftConfigurationError> {
    field_value
        .and_then(serde_json::Value::as_f64)
        .map(|wide_number| wide_number as f32)
        .ok_or(DFlashDraftConfigurationError::MissingField { field })
}

fn required_i32(
    document: &serde_json::Value,
    field: &'static str,
) -> Result<i32, DFlashDraftConfigurationError> {
    document
        .get(field)
        .and_then(serde_json::Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .ok_or(DFlashDraftConfigurationError::MissingField { field })
}

fn optional_i32(
    document: &serde_json::Value,
    field: &'static str,
) -> Result<Option<i32>, DFlashDraftConfigurationError> {
    match document.get(field) {
        None => Ok(None),
        Some(value) => value
            .as_i64()
            .and_then(|parsed| i32::try_from(parsed).ok())
            .map(Some)
            .ok_or(DFlashDraftConfigurationError::MalformedField { field }),
    }
}

#[derive(Debug, Error)]
pub enum DFlashDraftConfigurationError {
    #[error("the artifact does not declare the DFlashDraftModel architecture")]
    NotADFlashDraftModel,
    #[error("DFlash draft config.json is missing field {field}")]
    MissingField { field: &'static str },
    #[error("DFlash draft config.json field {field} is malformed")]
    MalformedField { field: &'static str },
    #[error("DFlash draft config.json is invalid: {description}")]
    InvalidGeometry { description: &'static str },
}

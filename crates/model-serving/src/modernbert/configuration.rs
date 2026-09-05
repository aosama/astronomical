//! Validated ModernBERT artifact geometry.
//!
//! Every value that shapes the forward pass is bounded and path-free so a
//! hostile artifact fails startup before any GPU memory is allocated.

use thiserror::Error;

const MAXIMUM_SUPPORTED_HIDDEN_SIZE: u32 = 4_096;
const MAXIMUM_SUPPORTED_LAYER_COUNT: u32 = 64;
const MINIMUM_LAYER_NORM_EPSILON: f32 = 1e-9;
const MAXIMUM_SUPPORTED_QUANTIZATION_GROUP_SIZE: u32 = 128;
const MAXIMUM_SUPPORTED_QUANTIZATION_BITS: u32 = 8;

/// Geometry and sampling controls derived from one validated artifact config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModernBertConfiguration {
    pub hidden_size: u32,
    pub layer_count: u32,
    pub attention_head_count: u32,
    pub maximum_position_count: u32,
    /// Bidirectional local attention total window; each side attends to half.
    pub local_attention_window: u32,
    /// Layers with `layer % global_attn_every_n_layers == 0` use full attention.
    pub global_attn_every_n_layers: u32,
    pub global_rope_theta: f32,
    pub local_rope_theta: f32,
    pub layer_norm_epsilon: f32,
    pub pad_token_id: u32,
    pub cls_token_id: u32,
    pub sep_token_id: u32,
    pub quantization_group_size: u32,
    pub quantization_bits: u32,
}

impl ModernBertConfiguration {
    /// Derives one bounded forward-pass geometry or fails closed on impossible values.
    pub fn from_config_value(
        config_value: &serde_json::Value,
    ) -> Result<Self, ModernBertConfigurationError> {
        let hidden_size = positive_u32(config_value, "hidden_size")?;
        let layer_count = positive_u32(config_value, "num_hidden_layers")?;
        let attention_head_count = positive_u32(config_value, "num_attention_heads")?;
        let maximum_position_count = positive_u32(config_value, "max_position_embeddings")?;
        let local_attention_window = positive_u32(config_value, "local_attention")?;
        let global_attn_every_n_layers = positive_u32(config_value, "global_attn_every_n_layers")?;
        let global_rope_theta = positive_f32(config_value, "global_rope_theta")?;
        let local_rope_theta = positive_f32(config_value, "local_rope_theta")?;
        let layer_norm_epsilon = positive_f32(config_value, "layer_norm_eps")?;
        let pad_token_id = token_id_field(config_value, "pad_token_id")?;
        let cls_token_id = token_id_field(config_value, "cls_token_id")?;
        let sep_token_id = token_id_field(config_value, "sep_token_id")?;
        let quantization_value = config_value
            .get("quantization")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let quantization_group_size = positive_u32(&quantization_value, "group_size")?;
        let quantization_bits = positive_u32(&quantization_value, "bits")?;

        if hidden_size > MAXIMUM_SUPPORTED_HIDDEN_SIZE {
            return Err(ModernBertConfigurationError::HiddenSizeTooLarge {
                hidden_size,
                maximum_hidden_size: MAXIMUM_SUPPORTED_HIDDEN_SIZE,
            });
        }
        if layer_count > MAXIMUM_SUPPORTED_LAYER_COUNT {
            return Err(ModernBertConfigurationError::LayerCountExceeded {
                actual_layer_count: layer_count,
                maximum_layer_count: MAXIMUM_SUPPORTED_LAYER_COUNT,
            });
        }
        if attention_head_count > hidden_size || hidden_size % attention_head_count != 0 {
            return Err(
                ModernBertConfigurationError::HiddenSizeNotDivisibleByHeads {
                    hidden_size,
                    attention_head_count,
                },
            );
        }
        if hidden_size % quantization_group_size != 0
            || quantization_group_size > MAXIMUM_SUPPORTED_QUANTIZATION_GROUP_SIZE
        {
            return Err(ModernBertConfigurationError::QuantizationGroupMisaligned {
                hidden_size,
                quantization_group_size,
            });
        }
        if quantization_bits == 0 || quantization_bits > MAXIMUM_SUPPORTED_QUANTIZATION_BITS {
            return Err(ModernBertConfigurationError::UnsupportedQuantization {
                quantization_group_size,
                quantization_bits,
            });
        }
        if layer_norm_epsilon < MINIMUM_LAYER_NORM_EPSILON {
            return Err(ModernBertConfigurationError::UnsupportedLayerNormEpsilon {
                actual: layer_norm_epsilon,
                minimum: MINIMUM_LAYER_NORM_EPSILON,
            });
        }
        Ok(Self {
            hidden_size,
            layer_count,
            attention_head_count,
            maximum_position_count,
            local_attention_window,
            global_attn_every_n_layers,
            global_rope_theta,
            local_rope_theta,
            layer_norm_epsilon,
            pad_token_id,
            cls_token_id,
            sep_token_id,
            quantization_group_size,
            quantization_bits,
        })
    }

    /// Native head dimension derived from validated geometry.
    #[must_use]
    pub const fn head_dimension(&self) -> u32 {
        self.hidden_size / self.attention_head_count
    }

    /// Specials participate in attention; they must not dominate the pooled mean.
    #[must_use]
    pub const fn is_excluded_from_pooled_mean(self, token_id: u32) -> bool {
        token_id == self.pad_token_id
            || token_id == self.cls_token_id
            || token_id == self.sep_token_id
    }
}

fn token_id_field(
    document: &serde_json::Value,
    field_name: &'static str,
) -> Result<u32, ModernBertConfigurationError> {
    u32::try_from(
        document
            .get(field_name)
            .and_then(serde_json::Value::as_u64)
            .ok_or(ModernBertConfigurationError::MissingField { field_name })?,
    )
    .map_err(|_| ModernBertConfigurationError::MissingField { field_name })
}

fn positive_u32(
    document: &serde_json::Value,
    field_name: &'static str,
) -> Result<u32, ModernBertConfigurationError> {
    let value = document
        .get(field_name)
        .and_then(serde_json::Value::as_u64)
        .ok_or(match field_name {
            "group_size" | "bits" => ModernBertConfigurationError::MissingQuantization,
            _ => ModernBertConfigurationError::MissingField { field_name },
        })?;
    let parsed_value = u32::try_from(value)
        .map_err(|_| ModernBertConfigurationError::MissingField { field_name })?;
    if parsed_value == 0 {
        return Err(ModernBertConfigurationError::MissingField { field_name });
    }
    Ok(parsed_value)
}

fn positive_f32(
    document: &serde_json::Value,
    field_name: &'static str,
) -> Result<f32, ModernBertConfigurationError> {
    let value = document
        .get(field_name)
        .and_then(serde_json::Value::as_f64)
        .ok_or(ModernBertConfigurationError::MissingField { field_name })?;
    if !value.is_finite() || value <= 0.0 {
        return Err(ModernBertConfigurationError::MissingField { field_name });
    }
    Ok(value as f32)
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum ModernBertConfigurationError {
    #[error("ModernBERT configuration field {field_name} is missing or invalid")]
    MissingField { field_name: &'static str },
    #[error(
        "ModernBERT hidden size is {hidden_size}, exceeding the supported {maximum_hidden_size}"
    )]
    HiddenSizeTooLarge {
        hidden_size: u32,
        maximum_hidden_size: u32,
    },
    #[error(
        "ModernBERT layer count is {actual_layer_count}, exceeding the {maximum_layer_count} bound"
    )]
    LayerCountExceeded {
        actual_layer_count: u32,
        maximum_layer_count: u32,
    },
    #[error(
        "ModernBERT hidden size {hidden_size} is not divisible by {attention_head_count} heads"
    )]
    HiddenSizeNotDivisibleByHeads {
        hidden_size: u32,
        attention_head_count: u32,
    },
    #[error(
        "ModernBERT hidden size {hidden_size} is not aligned to quantization group {quantization_group_size}"
    )]
    QuantizationGroupMisaligned {
        hidden_size: u32,
        quantization_group_size: u32,
    },
    #[error(
        "ModernBERT quantization group {quantization_group_size} or bit width {quantization_bits} is unsupported"
    )]
    UnsupportedQuantization {
        quantization_group_size: u32,
        quantization_bits: u32,
    },
    #[error("ModernBERT layer norm epsilon {actual} is below the supported floor {minimum}")]
    UnsupportedLayerNormEpsilon { actual: f32, minimum: f32 },
    #[error("ModernBERT missing quantization metadata")]
    MissingQuantization,
}

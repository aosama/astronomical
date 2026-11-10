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
        let (quantization_group_size, quantization_bits) = parse_quantization(config_value)?;
        let configuration = Self {
            hidden_size: positive_u32(config_value, "hidden_size")?,
            layer_count: positive_u32(config_value, "num_hidden_layers")?,
            attention_head_count: positive_u32(config_value, "num_attention_heads")?,
            maximum_position_count: positive_u32(config_value, "max_position_embeddings")?,
            local_attention_window: positive_u32(config_value, "local_attention")?,
            global_attn_every_n_layers: positive_u32(config_value, "global_attn_every_n_layers")?,
            global_rope_theta: rope_theta(config_value, "global_rope_theta", "full_attention")?,
            local_rope_theta: rope_theta(config_value, "local_rope_theta", "sliding_attention")?,
            layer_norm_epsilon: positive_f32(config_value, "layer_norm_eps")?,
            pad_token_id: token_id_field(config_value, "pad_token_id")?,
            cls_token_id: token_id_field(config_value, "cls_token_id")?,
            sep_token_id: token_id_field(config_value, "sep_token_id")?,
            quantization_group_size,
            quantization_bits,
        };
        configuration.validate_geometry()?;
        configuration.validate_quantization()?;
        if configuration.layer_norm_epsilon < MINIMUM_LAYER_NORM_EPSILON {
            return Err(ModernBertConfigurationError::UnsupportedLayerNormEpsilon {
                actual: configuration.layer_norm_epsilon,
                minimum: MINIMUM_LAYER_NORM_EPSILON,
            });
        }
        Ok(configuration)
    }

    /// Bounds the trunk so a hostile artifact cannot size GPU buffers.
    fn validate_geometry(&self) -> Result<(), ModernBertConfigurationError> {
        if self.hidden_size > MAXIMUM_SUPPORTED_HIDDEN_SIZE {
            return Err(ModernBertConfigurationError::HiddenSizeTooLarge {
                hidden_size: self.hidden_size,
                maximum_hidden_size: MAXIMUM_SUPPORTED_HIDDEN_SIZE,
            });
        }
        if self.layer_count > MAXIMUM_SUPPORTED_LAYER_COUNT {
            return Err(ModernBertConfigurationError::LayerCountExceeded {
                actual_layer_count: self.layer_count,
                maximum_layer_count: MAXIMUM_SUPPORTED_LAYER_COUNT,
            });
        }
        if self.attention_head_count > self.hidden_size
            || !self.hidden_size.is_multiple_of(self.attention_head_count)
        {
            return Err(
                ModernBertConfigurationError::HiddenSizeNotDivisibleByHeads {
                    hidden_size: self.hidden_size,
                    attention_head_count: self.attention_head_count,
                },
            );
        }
        Ok(())
    }

    /// Quantization groups tile each hidden row exactly; both values are
    /// already positive, so only the upper bounds and alignment remain.
    fn validate_quantization(&self) -> Result<(), ModernBertConfigurationError> {
        if !self
            .hidden_size
            .is_multiple_of(self.quantization_group_size)
            || self.quantization_group_size > MAXIMUM_SUPPORTED_QUANTIZATION_GROUP_SIZE
        {
            return Err(ModernBertConfigurationError::QuantizationGroupMisaligned {
                hidden_size: self.hidden_size,
                quantization_group_size: self.quantization_group_size,
            });
        }
        if self.quantization_bits > MAXIMUM_SUPPORTED_QUANTIZATION_BITS {
            return Err(ModernBertConfigurationError::UnsupportedQuantization {
                quantization_group_size: self.quantization_group_size,
                quantization_bits: self.quantization_bits,
            });
        }
        Ok(())
    }

    /// Native head dimension derived from validated geometry.
    #[must_use]
    pub const fn head_dimension(&self) -> u32 {
        self.hidden_size / self.attention_head_count
    }
}

/// One layer type's rotary base, from whichever of the two config forms is present.
///
/// Older configs carry flat `global_rope_theta` / `local_rope_theta` fields.
/// Newer ones nest the same numbers as
/// `rope_parameters.<layer type>.rope_theta`. The flat field wins when both
/// are present, so a config this engine already read keeps its meaning.
fn rope_theta(
    config_value: &serde_json::Value,
    flat_field_name: &'static str,
    layer_type_name: &str,
) -> Result<f32, ModernBertConfigurationError> {
    if config_value.get(flat_field_name).is_some() {
        return positive_f32(config_value, flat_field_name);
    }
    let missing_rotary_base = ModernBertConfigurationError::MissingField {
        field_name: flat_field_name,
    };
    let Some(rope_parameters) = config_value.get("rope_parameters") else {
        return Err(missing_rotary_base);
    };
    let Some(layer_type_parameters) = rope_parameters.get(layer_type_name) else {
        return Err(missing_rotary_base);
    };
    match positive_f32(layer_type_parameters, "rope_theta") {
        Ok(nested_rope_theta) => Ok(nested_rope_theta),
        // Reported under the flat name so one field name identifies a missing
        // rotary base whichever form the config uses.
        Err(_) => Err(missing_rotary_base),
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

/// The `quantization` block's group size and bits. An absent block or
/// field is the unquantized artifact this engine does not serve, which is a
/// different failure from a malformed top-level field.
fn parse_quantization(
    config_value: &serde_json::Value,
) -> Result<(u32, u32), ModernBertConfigurationError> {
    let quantization_block = config_value
        .get("quantization")
        .ok_or(ModernBertConfigurationError::MissingQuantization)?;
    Ok((
        positive_u32_or(
            quantization_block,
            "group_size",
            ModernBertConfigurationError::MissingQuantization,
        )?,
        positive_u32_or(
            quantization_block,
            "bits",
            ModernBertConfigurationError::MissingQuantization,
        )?,
    ))
}

fn positive_u32(
    document: &serde_json::Value,
    field_name: &'static str,
) -> Result<u32, ModernBertConfigurationError> {
    positive_u32_or(
        document,
        field_name,
        ModernBertConfigurationError::MissingField { field_name },
    )
}

/// A positive `u32` field; `absent_error` when it is missing or not an
/// unsigned integer, `MissingField` when present but zero or too wide.
fn positive_u32_or(
    document: &serde_json::Value,
    field_name: &'static str,
    absent_error: ModernBertConfigurationError,
) -> Result<u32, ModernBertConfigurationError> {
    let unsigned_field = document
        .get(field_name)
        .and_then(serde_json::Value::as_u64)
        .ok_or(absent_error)?;
    match u32::try_from(unsigned_field) {
        Ok(positive_field) if positive_field > 0 => Ok(positive_field),
        _ => Err(ModernBertConfigurationError::MissingField { field_name }),
    }
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

#[cfg(test)]
#[path = "configuration_tests.rs"]
mod configuration_tests;

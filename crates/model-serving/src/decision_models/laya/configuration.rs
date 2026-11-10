//! Validated settings of one Laya artifact's `config.json`.
//!
//! Everything that sizes a sequence, a batch or a graph is bounded here, so a
//! malformed or hostile artifact fails at startup and never reaches the
//! graphics processor.

use astronomical_decision_contract::calibration::{
    BucketTemperature, KindTemperatures, NEUTRAL_TEMPERATURE, TemperatureTable,
};
use astronomical_decision_contract::laya_sequence::LayaSequenceLimits;
use thiserror::Error;

use crate::modernbert::configuration::{ModernBertConfiguration, ModernBertConfigurationError};

/// The sequence length a checkpoint reads when its config does not say.
const DEFAULT_MAXIMUM_SEQUENCE_TOKENS: u64 = 512;
/// The instruction-and-options budget when the config does not say.
const DEFAULT_HEAD_TOKEN_BUDGET: u64 = 192;
/// The head depth when the config does not say; every published checkpoint has two.
const DEFAULT_HEAD_LAYER_COUNT: u64 = 2;
/// The deepest head accepted. Published checkpoints have two layers; the
/// bound only stops a malformed config from building an enormous graph.
const MAXIMUM_HEAD_LAYER_COUNT: u64 = 8;
/// A sequence must at least hold the three separators, one instruction token
/// and two option markers.
const MINIMUM_SEQUENCE_TOKENS: u64 = 8;
/// The only option layout this backend runs. A checkpoint trained on the
/// parallel layout reads a sequential sequence as a different input.
const SEQUENTIAL_OPTION_LAYOUT: &str = "sequential";
/// The head's attention uses heads of this width, as the checkpoints were built.
const HEAD_ATTENTION_HEAD_WIDTH: u32 = 64;

/// Everything `config.json` says about one Laya checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayaConfiguration {
    pub(crate) encoder: ModernBertConfiguration,
    pub(crate) head_layer_count: u32,
    pub(crate) sequence_limits: LayaSequenceLimits,
    pub(crate) temperature_table: TemperatureTable,
}

/// Why a Laya `config.json` cannot be served.
#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum LayaConfigurationError {
    #[error("the Laya config has no `encoder` object")]
    MissingEncoder,
    #[error("the Laya encoder config is unsupported: {0}")]
    Encoder(#[from] ModernBertConfigurationError),
    #[error("the Laya config field {field_name} is not a usable number")]
    InvalidField { field_name: &'static str },
    #[error("the Laya head has {actual} layers; at most {maximum} are supported")]
    TooManyHeadLayers { actual: u64, maximum: u64 },
    #[error(
        "the Laya sequence limits are unusable: max_len {maximum_sequence_tokens}, head_max_len {head_token_budget}, encoder positions {encoder_position_count}"
    )]
    UnusableSequenceLimits {
        maximum_sequence_tokens: u64,
        head_token_budget: u64,
        encoder_position_count: u32,
    },
    #[error(
        "the Laya head cannot split hidden size {hidden_size} into {head_count} attention heads of equal width"
    )]
    UnevenHeadAttentionWidth { hidden_size: u32, head_count: u32 },
    #[error("the Laya option layout {actual:?} is not supported; only \"sequential\" is")]
    UnsupportedOptionLayout { actual: String },
    #[error("the Laya config `temperature` must be three numbers: choice, score, noul")]
    MalformedTemperatures,
    #[error("the Laya config `temperature_by_options` must map bucket names to numbers")]
    MalformedBucketTemperatures,
}

impl LayaConfiguration {
    pub(crate) fn from_config_value(
        config_value: &serde_json::Value,
    ) -> Result<Self, LayaConfigurationError> {
        let encoder = parse_encoder(config_value)?;
        reject_unsupported_option_layout(config_value)?;
        let head_layer_count =
            unsigned_field_or(config_value, "head_layers", DEFAULT_HEAD_LAYER_COUNT)?;
        if head_layer_count > MAXIMUM_HEAD_LAYER_COUNT {
            return Err(LayaConfigurationError::TooManyHeadLayers {
                actual: head_layer_count,
                maximum: MAXIMUM_HEAD_LAYER_COUNT,
            });
        }
        let sequence_limits = parse_sequence_limits(config_value, &encoder)?;
        let configuration = Self {
            encoder,
            head_layer_count: head_layer_count as u32,
            sequence_limits,
            temperature_table: parse_temperature_table(config_value)?,
        };
        // The head reshapes its projections into equal-width heads; a hidden
        // size that does not divide evenly must fail here, not in a request.
        let head_count = configuration.head_attention_head_count();
        if !configuration.encoder.hidden_size.is_multiple_of(head_count) {
            return Err(LayaConfigurationError::UnevenHeadAttentionWidth {
                hidden_size: configuration.encoder.hidden_size,
                head_count,
            });
        }
        Ok(configuration)
    }

    /// How many attention heads each head layer has: one per 64 hidden channels.
    pub(crate) fn head_attention_head_count(&self) -> u32 {
        (self.encoder.hidden_size / HEAD_ATTENTION_HEAD_WIDTH).max(1)
    }
}

/// Reads the nested encoder config with the artifact's top-level quantization.
///
/// The conversion keeps the encoder's own config under `encoder` and records
/// the quantization once at the top level, while the trunk's configuration
/// reader expects both in one object, so the two are joined here.
fn parse_encoder(
    config_value: &serde_json::Value,
) -> Result<ModernBertConfiguration, LayaConfigurationError> {
    let Some(encoder_fields) = config_value
        .get("encoder")
        .and_then(serde_json::Value::as_object)
    else {
        return Err(LayaConfigurationError::MissingEncoder);
    };
    let mut joined_fields = encoder_fields.clone();
    if let Some(quantization_block) = config_value.get("quantization") {
        joined_fields.insert("quantization".to_owned(), quantization_block.clone());
    }
    Ok(ModernBertConfiguration::from_config_value(
        &serde_json::Value::Object(joined_fields),
    )?)
}

fn reject_unsupported_option_layout(
    config_value: &serde_json::Value,
) -> Result<(), LayaConfigurationError> {
    match config_value.get("option_layout") {
        None | Some(serde_json::Value::Null) => Ok(()),
        Some(serde_json::Value::String(layout)) if layout == SEQUENTIAL_OPTION_LAYOUT => Ok(()),
        Some(other_layout) => Err(LayaConfigurationError::UnsupportedOptionLayout {
            actual: other_layout.to_string(),
        }),
    }
}

fn parse_sequence_limits(
    config_value: &serde_json::Value,
    encoder: &ModernBertConfiguration,
) -> Result<LayaSequenceLimits, LayaConfigurationError> {
    let maximum_sequence_tokens =
        unsigned_field_or(config_value, "max_len", DEFAULT_MAXIMUM_SEQUENCE_TOKENS)?;
    let head_token_budget =
        unsigned_field_or(config_value, "head_max_len", DEFAULT_HEAD_TOKEN_BUDGET)?;
    let limits_are_usable = maximum_sequence_tokens >= MINIMUM_SEQUENCE_TOKENS
        && maximum_sequence_tokens <= u64::from(encoder.maximum_position_count)
        && head_token_budget >= 1
        && head_token_budget < maximum_sequence_tokens;
    if !limits_are_usable {
        return Err(LayaConfigurationError::UnusableSequenceLimits {
            maximum_sequence_tokens,
            head_token_budget,
            encoder_position_count: encoder.maximum_position_count,
        });
    }
    Ok(LayaSequenceLimits {
        maximum_sequence_tokens: maximum_sequence_tokens as usize,
        head_token_budget: head_token_budget as usize,
    })
}

fn parse_temperature_table(
    config_value: &serde_json::Value,
) -> Result<TemperatureTable, LayaConfigurationError> {
    let kind_temperatures = match config_value.get("temperature") {
        None | Some(serde_json::Value::Null) => KindTemperatures {
            choice: NEUTRAL_TEMPERATURE,
            score: NEUTRAL_TEMPERATURE,
            noul: NEUTRAL_TEMPERATURE,
        },
        Some(serde_json::Value::Array(temperatures)) if temperatures.len() == 3 => {
            // A checkpoint lists its temperatures in its own type order:
            // choice, score, noul.
            KindTemperatures {
                choice: temperature_number(&temperatures[0])?,
                score: temperature_number(&temperatures[1])?,
                noul: temperature_number(&temperatures[2])?,
            }
        }
        Some(_) => return Err(LayaConfigurationError::MalformedTemperatures),
    };
    let mut bucket_temperatures = Vec::new();
    match config_value.get("temperature_by_options") {
        None | Some(serde_json::Value::Null) => {}
        Some(serde_json::Value::Object(bucket_fields)) => {
            for (bucket_key, raw_temperature) in bucket_fields {
                let Some(temperature) = raw_temperature.as_f64() else {
                    return Err(LayaConfigurationError::MalformedBucketTemperatures);
                };
                bucket_temperatures.push(BucketTemperature {
                    bucket_key: bucket_key.clone(),
                    temperature,
                });
            }
        }
        Some(_) => return Err(LayaConfigurationError::MalformedBucketTemperatures),
    }
    Ok(TemperatureTable::from_raw_temperatures(
        kind_temperatures,
        &bucket_temperatures,
    ))
}

fn temperature_number(raw_temperature: &serde_json::Value) -> Result<f64, LayaConfigurationError> {
    raw_temperature
        .as_f64()
        .ok_or(LayaConfigurationError::MalformedTemperatures)
}

/// An unsigned integer field, or `default_value` when the config omits it.
fn unsigned_field_or(
    config_value: &serde_json::Value,
    field_name: &'static str,
    default_value: u64,
) -> Result<u64, LayaConfigurationError> {
    match config_value.get(field_name) {
        None | Some(serde_json::Value::Null) => Ok(default_value),
        Some(present_field) => present_field
            .as_u64()
            .ok_or(LayaConfigurationError::InvalidField { field_name }),
    }
}

#[cfg(test)]
#[path = "configuration_tests.rs"]
mod tests;

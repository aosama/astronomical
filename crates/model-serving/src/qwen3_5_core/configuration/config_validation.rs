use serde::{Deserialize, Deserializer};
use thiserror::Error;

/// Converts a JSON float into exact IEEE-754 bits for deterministic config validation.
pub(super) fn deserialize_f32_bits<'de, DeserializerType>(
    deserializer: DeserializerType,
) -> Result<u32, DeserializerType::Error>
where
    DeserializerType: Deserializer<'de>,
{
    let float_value = f32::deserialize(deserializer)?;
    Ok(float_value.to_bits())
}

/// Converts an optional JSON float into exact IEEE-754 bits.
pub(super) fn deserialize_optional_f32_bits<'de, DeserializerType>(
    deserializer: DeserializerType,
) -> Result<Option<u32>, DeserializerType::Error>
where
    DeserializerType: Deserializer<'de>,
{
    let optional_float_value = Option::<f32>::deserialize(deserializer)?;
    Ok(optional_float_value.map(f32::to_bits))
}

/// Same as the single-integer/array normalization but returns `Option<Vec<u32>>`
/// to handle configs where the `eos_token_id` field is absent (e.g., Agents-A1).
pub(super) fn deserialize_optional_eos_token_ids<'de, DeserializerType>(
    deserializer: DeserializerType,
) -> Result<Option<Vec<u32>>, DeserializerType::Error>
where
    DeserializerType: Deserializer<'de>,
{
    Option::<EosTokenId>::deserialize(deserializer).map(|opt| {
        opt.map(|eos_token_id_variant| match eos_token_id_variant {
            EosTokenId::Single(token_id) => vec![token_id],
            EosTokenId::Array(token_ids) => token_ids,
        })
    })
}

/// The standard Qwen chat end-of-sequence token ID.
/// When `eos_token_id` is absent at the top level and resolved from `text_config`,
/// this token is appended to the list if not already present.
pub(super) const QWEN_CHAT_EOS_TOKEN_ID: u32 = 248046;

/// Intermediate enum for deserializing `eos_token_id` from either a single integer
/// or an array of integers.
#[derive(Deserialize)]
#[serde(untagged)]
enum EosTokenId {
    Single(u32),
    Array(Vec<u32>),
}

pub(in crate::qwen3_5) fn validate_exact_value(
    field_name: &'static str,
    actual_value: &str,
    expected_value: &'static str,
) -> Result<(), Qwen3_5ConfigError> {
    if actual_value == expected_value {
        return Ok(());
    }
    Err(Qwen3_5ConfigError::UnexpectedStringValue {
        field_name,
        expected_value,
        actual_value: actual_value.to_owned(),
    })
}

pub(super) fn validate_exact_boolean(
    field_name: &'static str,
    actual_value: bool,
    expected_value: bool,
) -> Result<(), Qwen3_5ConfigError> {
    if actual_value == expected_value {
        return Ok(());
    }
    Err(Qwen3_5ConfigError::UnexpectedBooleanValue {
        field_name,
        expected_value,
        actual_value,
    })
}

/// A config-decoding or implemented-execution-contract failure for Qwen3.5.
#[derive(Debug, Error)]
pub enum Qwen3_5ConfigError {
    /// The retained config bytes were not valid JSON for the expected document shape.
    #[error("failed to decode Qwen3.5 config JSON")]
    DeserializeConfig(#[source] serde_json::Error),
    /// A required string field selected behavior this executor does not implement.
    #[error("Qwen3.5 config field '{field_name}' must be '{expected_value}', got '{actual_value}'")]
    UnexpectedStringValue {
        field_name: &'static str,
        expected_value: &'static str,
        actual_value: String,
    },
    /// A required boolean field selected behavior this executor does not implement.
    #[error("Qwen3.5 config field '{field_name}' must be {expected_value}, got {actual_value}")]
    UnexpectedBooleanValue {
        field_name: &'static str,
        expected_value: bool,
        actual_value: bool,
    },
    /// The decoder did not declare exactly one attention kind for every text layer.
    #[error(
        "Qwen3.5 config declares {actual_layer_type_count} layer types, expected {expected_layer_type_count}"
    )]
    LayerTypeCountMismatch {
        actual_layer_type_count: usize,
        expected_layer_type_count: usize,
    },
    /// An affine override used a bit width unsupported by MLX.
    #[error("module '{module_name}' uses unsupported {actual_value}-bit affine quantization")]
    UnsupportedQuantizationOverrideBits {
        module_name: String,
        actual_value: u32,
    },
    /// The multimodal rotary section differed from the pinned text configuration.
    #[error("Qwen3.5 mrope section {actual_section:?} differs from [11, 11, 10]")]
    MropeSectionMismatch { actual_section: [u32; 3] },
    /// The duplicated MLX quantization documents did not match exactly.
    #[error("Qwen3.5 quantization and quantization_config fields differ")]
    QuantizationCopiesDiffer,
    /// The retained config bytes were not valid JSON for the expected vision_config shape.
    #[error("failed to decode Qwen3.5 vision config JSON")]
    DeserializeVisionConfig(#[source] serde_json::Error),
    #[error("Qwen3.5 config does not declare vision_config")]
    MissingVisionConfig,
    /// A structural sanity check failed (e.g., non-positive dimension, inconsistent counts).
    #[error("invalid Qwen3.5 config: {description}")]
    InvalidConfigValue { description: &'static str },
    /// A structural sanity check failed with a dynamic description.
    #[error("invalid Qwen3.5 config: {description}")]
    InvalidConfigValueDynamic { description: String },
    /// The activation dtype (`dtype` or `torch_dtype`) was absent from both the
    /// top level and `text_config`. At least one must be present.
    #[error("Qwen3.5 config must specify `dtype` at the top level or inside `text_config`")]
    MissingActivationDtype,
    /// The `eos_token_id` field was absent from both the top level and `text_config`.
    /// At least one must be present so the model can identify stop tokens.
    #[error("Qwen3.5 config must specify `eos_token_id` at the top level or inside `text_config`")]
    MissingEosTokenId,
}

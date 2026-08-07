use serde::Deserialize;
use thiserror::Error;

const EXPECTED_ARCHITECTURE: &str = "DeepseekV4ForCausalLM";
const EXPECTED_MODEL_TYPE: &str = "deepseek_v4";
const EXPECTED_VOCABULARY_SIZE: u32 = 129_280;
const EXPECTED_HIDDEN_SIZE: u32 = 4_096;
const EXPECTED_MOE_INTERMEDIATE_SIZE: u32 = 2_048;
const EXPECTED_LAYER_COUNT: u32 = 43;
const EXPECTED_ATTENTION_HEAD_COUNT: u32 = 64;
const EXPECTED_KEY_VALUE_HEAD_COUNT: u32 = 1;
const EXPECTED_ROUTED_EXPERT_COUNT: u32 = 256;
const EXPECTED_SHARED_EXPERT_COUNT: u32 = 1;
const EXPECTED_EXPERTS_PER_TOKEN: u32 = 6;
const EXPECTED_HASH_LAYER_COUNT: u32 = 3;
const EXPECTED_QUANTIZATION_BITS: u32 = 8;
const EXPECTED_DSPARK_BLOCK_SIZE: u32 = 5;
const EXPECTED_DSPARK_TARGET_LAYER_IDS: [u32; 3] = [40, 41, 42];
const EXPECTED_DSPARK_NOISE_TOKEN_ID: u32 = 128_799;
const EXPECTED_DSPARK_MARKOV_RANK: u32 = 256;

/// DSpark metadata declared by a selected DeepSeek-V4 Flash artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4DsparkArtifactCapability {
    /// The artifact contains only the autoregressive target model.
    TargetOnly,
    /// The artifact declares the compatible 0731 DSpark metadata.
    Declared,
}

impl DeepSeekV4DsparkArtifactCapability {
    /// Returns whether the artifact declares the supported DSpark metadata.
    #[must_use]
    pub const fn is_declared(&self) -> bool {
        matches!(self, Self::Declared)
    }
}

/// Validated configuration identity for DeepSeek-V4-Flash-0731.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4Flash0731Config {
    dspark_artifact_capability: DeepSeekV4DsparkArtifactCapability,
}

impl DeepSeekV4Flash0731Config {
    /// Parses the subset of configuration required by the selected architecture.
    pub fn from_json_bytes(config_bytes: &[u8]) -> Result<Self, DeepSeekV4ConfigError> {
        let config_document = serde_json::from_slice::<DeepSeekV4ConfigDocument>(config_bytes)
            .map_err(DeepSeekV4ConfigError::DeserializeConfig)?;
        validate_exact_string(
            "model_type",
            &config_document.model_type,
            EXPECTED_MODEL_TYPE,
        )?;
        if config_document.architectures.len() != 1
            || config_document.architectures[0] != EXPECTED_ARCHITECTURE
        {
            return Err(DeepSeekV4ConfigError::InvalidConfigValue {
                description: "architectures must contain only DeepseekV4ForCausalLM",
            });
        }
        validate_exact_number(
            "vocab_size",
            config_document.vocab_size,
            EXPECTED_VOCABULARY_SIZE,
        )?;
        validate_exact_number(
            "hidden_size",
            config_document.hidden_size,
            EXPECTED_HIDDEN_SIZE,
        )?;
        validate_exact_number(
            "moe_intermediate_size",
            config_document.moe_intermediate_size,
            EXPECTED_MOE_INTERMEDIATE_SIZE,
        )?;
        validate_exact_number(
            "num_hidden_layers",
            config_document.num_hidden_layers,
            EXPECTED_LAYER_COUNT,
        )?;
        validate_exact_number(
            "num_attention_heads",
            config_document.num_attention_heads,
            EXPECTED_ATTENTION_HEAD_COUNT,
        )?;
        validate_exact_number(
            "num_key_value_heads",
            config_document.num_key_value_heads,
            EXPECTED_KEY_VALUE_HEAD_COUNT,
        )?;
        validate_exact_number(
            "n_routed_experts",
            config_document.n_routed_experts,
            EXPECTED_ROUTED_EXPERT_COUNT,
        )?;
        validate_exact_number(
            "n_shared_experts",
            config_document.n_shared_experts,
            EXPECTED_SHARED_EXPERT_COUNT,
        )?;
        validate_exact_number(
            "num_experts_per_tok",
            config_document.num_experts_per_tok,
            EXPECTED_EXPERTS_PER_TOKEN,
        )?;
        validate_exact_number(
            "num_hash_layers",
            config_document.num_hash_layers,
            EXPECTED_HASH_LAYER_COUNT,
        )?;
        validate_exact_number(
            "quantization.bits",
            config_document.quantization.bits,
            EXPECTED_QUANTIZATION_BITS,
        )?;

        let dspark_artifact_capability = match (
            config_document.dspark_block_size,
            config_document.dspark_target_layer_ids,
            config_document.dspark_noise_token_id,
            config_document.dspark_markov_rank,
        ) {
            (None, None, None, None) => DeepSeekV4DsparkArtifactCapability::TargetOnly,
            (Some(block_size), Some(target_layer_ids), Some(noise_token_id), Some(markov_rank)) => {
                validate_exact_number("dspark_block_size", block_size, EXPECTED_DSPARK_BLOCK_SIZE)?;
                if target_layer_ids.as_slice() != EXPECTED_DSPARK_TARGET_LAYER_IDS.as_slice() {
                    return Err(DeepSeekV4ConfigError::InvalidConfigValue {
                        description: "dspark_target_layer_ids must equal [40, 41, 42]",
                    });
                }
                validate_exact_number(
                    "dspark_noise_token_id",
                    noise_token_id,
                    EXPECTED_DSPARK_NOISE_TOKEN_ID,
                )?;
                validate_exact_number(
                    "dspark_markov_rank",
                    markov_rank,
                    EXPECTED_DSPARK_MARKOV_RANK,
                )?;
                DeepSeekV4DsparkArtifactCapability::Declared
            }
            _ => return Err(DeepSeekV4ConfigError::IncompleteDsparkMetadata),
        };
        Ok(Self {
            dspark_artifact_capability,
        })
    }

    /// Returns whether this configuration declares target-only or DSpark execution.
    #[must_use]
    pub const fn dspark_artifact_capability(&self) -> &DeepSeekV4DsparkArtifactCapability {
        &self.dspark_artifact_capability
    }
}

#[derive(Debug, Deserialize)]
struct DeepSeekV4ConfigDocument {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: u32,
    hidden_size: u32,
    moe_intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    n_routed_experts: u32,
    n_shared_experts: u32,
    num_experts_per_tok: u32,
    num_hash_layers: u32,
    quantization: DeepSeekV4QuantizationDocument,
    #[serde(default)]
    dspark_block_size: Option<u32>,
    #[serde(default)]
    dspark_target_layer_ids: Option<Vec<u32>>,
    #[serde(default)]
    dspark_noise_token_id: Option<u32>,
    #[serde(default)]
    dspark_markov_rank: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct DeepSeekV4QuantizationDocument {
    bits: u32,
}

fn validate_exact_string(
    field_name: &'static str,
    actual_value: &str,
    expected_value: &'static str,
) -> Result<(), DeepSeekV4ConfigError> {
    if actual_value == expected_value {
        return Ok(());
    }
    Err(DeepSeekV4ConfigError::UnexpectedStringValue {
        field_name,
        expected_value,
        actual_value: actual_value.to_owned(),
    })
}

fn validate_exact_number(
    field_name: &'static str,
    actual_value: u32,
    expected_value: u32,
) -> Result<(), DeepSeekV4ConfigError> {
    if actual_value == expected_value {
        return Ok(());
    }
    Err(DeepSeekV4ConfigError::UnexpectedNumberValue {
        field_name,
        expected_value,
        actual_value,
    })
}

/// A selected-architecture configuration failure.
#[derive(Debug, Error)]
pub enum DeepSeekV4ConfigError {
    /// The configuration document was not valid JSON for the expected shape.
    #[error("failed to decode DeepSeek-V4 config JSON")]
    DeserializeConfig(#[source] serde_json::Error),
    /// A string field selected a different model architecture.
    #[error(
        "DeepSeek-V4 config field '{field_name}' must be '{expected_value}', got '{actual_value}'"
    )]
    UnexpectedStringValue {
        field_name: &'static str,
        expected_value: &'static str,
        actual_value: String,
    },
    /// A numeric field selected a different model architecture.
    #[error("DeepSeek-V4 config field '{field_name}' must be {expected_value}, got {actual_value}")]
    UnexpectedNumberValue {
        field_name: &'static str,
        expected_value: u32,
        actual_value: u32,
    },
    /// The selected DSpark artifact declared only part of its required metadata.
    #[error("DeepSeek-V4 DSpark metadata must be absent or complete")]
    IncompleteDsparkMetadata,
    /// A structural selected-architecture contract was not met.
    #[error("invalid DeepSeek-V4 config: {description}")]
    InvalidConfigValue { description: &'static str },
}

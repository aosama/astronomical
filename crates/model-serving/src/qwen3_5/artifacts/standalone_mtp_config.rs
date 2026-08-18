//! Strict semantic owner for a standalone Qwen MTP configuration document.

use serde::Deserialize;

use crate::{OptiQQuantizationProfile, Qwen3_5FeedForwardArchitecture};

/// Independently parsed geometry and storage evidence from a standalone MTP artifact.
#[derive(Clone, Debug)]
pub struct Qwen3_5StandaloneMtpConfig {
    block_size: u32,
    text_config: StandaloneTextConfig,
    tie_word_embeddings: bool,
    quantization_profile: Option<OptiQQuantizationProfile>,
    raw_text_config: serde_json::Value,
}

impl Qwen3_5StandaloneMtpConfig {
    /// Parses and validates the standalone publisher schema without consulting a target artifact.
    pub fn from_json_bytes(config_bytes: &[u8]) -> Result<Self, Qwen3_5StandaloneMtpConfigError> {
        let config_document: StandaloneConfigDocument = serde_json::from_slice(config_bytes)
            .map_err(Qwen3_5StandaloneMtpConfigError::Malformed)?;
        if config_document.model_type != "qwen3_5_mtp" {
            return Err(Qwen3_5StandaloneMtpConfigError::UnsupportedModelType);
        }
        if config_document.block_size < 2 {
            return Err(Qwen3_5StandaloneMtpConfigError::InvalidBlockSize);
        }
        let text_config: StandaloneTextConfig =
            serde_json::from_value(config_document.text_config.clone())
                .map_err(Qwen3_5StandaloneMtpConfigError::Malformed)?;
        validate_text_config(&text_config)?;
        if text_config
            .tie_word_embeddings
            .is_some_and(|text_tied_embeddings| {
                config_document.tie_word_embeddings != text_tied_embeddings
            })
        {
            return Err(Qwen3_5StandaloneMtpConfigError::TiedEmbeddingDisagreement);
        }
        if config_document.quantization.is_some()
            && config_document.quantization_config.is_some()
            && config_document.quantization != config_document.quantization_config
        {
            return Err(Qwen3_5StandaloneMtpConfigError::QuantizationDocumentDisagreement);
        }
        let quantization_profile = config_document
            .quantization
            .or(config_document.quantization_config)
            .map(parse_quantization_profile)
            .transpose()?;
        Ok(Self {
            block_size: config_document.block_size,
            tie_word_embeddings: config_document.tie_word_embeddings,
            raw_text_config: config_document.text_config,
            text_config,
            quantization_profile,
        })
    }

    #[must_use]
    pub const fn maximum_draft_depth(&self) -> u8 {
        let publisher_depth = self.block_size.saturating_sub(1);
        if publisher_depth > 3 {
            3
        } else {
            publisher_depth as u8
        }
    }

    #[must_use]
    pub const fn hidden_size(&self) -> u32 {
        self.text_config.hidden_size
    }

    #[must_use]
    pub const fn vocabulary_size(&self) -> u32 {
        self.text_config.vocab_size
    }

    #[must_use]
    pub const fn mtp_layer_count(&self) -> u32 {
        self.text_config.mtp_num_hidden_layers
    }

    #[must_use]
    pub fn feed_forward_architecture(&self) -> Qwen3_5FeedForwardArchitecture {
        if self.text_config.model_type.contains("moe") {
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts
        } else {
            Qwen3_5FeedForwardArchitecture::Dense
        }
    }

    #[must_use]
    pub const fn has_tied_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    #[must_use]
    pub const fn quantization_profile(&self) -> Option<OptiQQuantizationProfile> {
        self.quantization_profile
    }

    #[must_use]
    pub const fn raw_text_config(&self) -> &serde_json::Value {
        &self.raw_text_config
    }
}

/// Invalid or unsupported standalone Qwen MTP configuration evidence.
#[derive(Debug, thiserror::Error)]
pub enum Qwen3_5StandaloneMtpConfigError {
    #[error("standalone MTP config is malformed")]
    Malformed(#[source] serde_json::Error),
    #[error("standalone MTP model_type must be qwen3_5_mtp")]
    UnsupportedModelType,
    #[error("standalone MTP block_size must be at least two")]
    InvalidBlockSize,
    #[error("standalone MTP text geometry is incomplete or unsupported")]
    InvalidTextGeometry,
    #[error("standalone MTP must contain exactly one hidden layer")]
    UnsupportedMtpLayerCount,
    #[error("standalone MTP must reuse target embeddings")]
    DedicatedEmbeddingsUnsupported,
    #[error("standalone MTP tied-embedding declarations disagree")]
    TiedEmbeddingDisagreement,
    #[error("standalone MTP quantization and quantization_config disagree")]
    QuantizationDocumentDisagreement,
    #[error("standalone MTP affine quantization geometry is unsupported")]
    UnsupportedQuantization,
}

#[derive(Deserialize)]
struct StandaloneConfigDocument {
    model_type: String,
    block_size: u32,
    tie_word_embeddings: bool,
    text_config: serde_json::Value,
    #[serde(default)]
    quantization: Option<serde_json::Value>,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
struct StandaloneTextConfig {
    model_type: String,
    hidden_size: u32,
    intermediate_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    mtp_num_hidden_layers: u32,
    #[serde(default)]
    mtp_use_dedicated_embeddings: bool,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

fn validate_text_config(
    text_config: &StandaloneTextConfig,
) -> Result<(), Qwen3_5StandaloneMtpConfigError> {
    if !text_config.model_type.starts_with("qwen3_5")
        || text_config.hidden_size == 0
        || text_config.intermediate_size == 0
        || text_config.num_attention_heads == 0
        || text_config.num_key_value_heads == 0
        || text_config.head_dim == 0
        || text_config.vocab_size == 0
    {
        return Err(Qwen3_5StandaloneMtpConfigError::InvalidTextGeometry);
    }
    if text_config.mtp_num_hidden_layers != 1 {
        return Err(Qwen3_5StandaloneMtpConfigError::UnsupportedMtpLayerCount);
    }
    if text_config.mtp_use_dedicated_embeddings {
        return Err(Qwen3_5StandaloneMtpConfigError::DedicatedEmbeddingsUnsupported);
    }
    Ok(())
}

fn parse_quantization_profile(
    quantization_document: serde_json::Value,
) -> Result<OptiQQuantizationProfile, Qwen3_5StandaloneMtpConfigError> {
    let bits = quantization_document
        .get("bits")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let group_size = quantization_document
        .get("group_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    match (bits, group_size) {
        (Some(bits @ (2 | 3 | 4 | 5 | 6 | 8)), Some(group_size @ (32 | 64 | 128))) => {
            Ok(OptiQQuantizationProfile { bits, group_size })
        }
        _ => Err(Qwen3_5StandaloneMtpConfigError::UnsupportedQuantization),
    }
}

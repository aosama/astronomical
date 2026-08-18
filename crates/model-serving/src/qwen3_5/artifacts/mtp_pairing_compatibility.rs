//! Pure compatibility comparison between independently validated target and drafter contracts.

use crate::{
    OptiQQuantizationProfile, Qwen3_5Config, Qwen3_5FeedForwardArchitecture,
    Qwen3_5StandaloneMtpConfig, Qwen3_5Tokenizer,
};

/// Compatibility evidence authorized for runtime source selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5MtpPairingCompatibility {
    pub maximum_draft_depth: u8,
    pub requested_draft_depth: u8,
    pub quantization_profile: Option<OptiQQuantizationProfile>,
}

/// One bounded reason that an explicit target/drafter pairing cannot execute.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Qwen3_5MtpPairingCompatibilityError {
    #[error("standalone MTP metadata is missing required field '{field_name}'")]
    MissingField { field_name: &'static str },
    #[error("standalone MTP metadata conflicts with target field '{field_name}'")]
    FieldMismatch { field_name: &'static str },
    #[error("standalone MTP tokenizer is malformed")]
    InvalidTokenizer,
    #[error("standalone MTP tokenizer mapping differs from the target")]
    TokenizerMappingMismatch,
    #[error("configured MTP draft depth exceeds standalone artifact maximum")]
    UnsupportedDraftDepth,
}

/// Proves target geometry, tokenizer mapping, and requested depth before weight binding.
pub fn compare_qwen3_5_mtp_pairing_contracts(
    target_config: &Qwen3_5Config,
    target_tokenizer_bytes: &[u8],
    drafter_config: &Qwen3_5StandaloneMtpConfig,
    drafter_tokenizer_bytes: &[u8],
    configured_draft_depth: Option<u8>,
) -> Result<Qwen3_5MtpPairingCompatibility, Qwen3_5MtpPairingCompatibilityError> {
    compare_typed_geometry(target_config, drafter_config)?;
    let text_config = drafter_config.raw_text_config();
    compare_u32(
        text_config,
        "num_hidden_layers",
        target_config.layer_count(),
    )?;
    compare_u32(
        text_config,
        "max_position_embeddings",
        target_config.maximum_position_count(),
    )?;
    compare_u32(
        text_config,
        "num_attention_heads",
        target_config.query_head_count(),
    )?;
    compare_u32(
        text_config,
        "num_key_value_heads",
        target_config.key_value_head_count(),
    )?;
    compare_u32(text_config, "head_dim", target_config.head_dimension())?;
    compare_bool(
        text_config,
        "attention_bias",
        target_config.has_attention_bias(),
    )?;
    compare_string(text_config, "hidden_act", target_config.hidden_activation())?;
    compare_f32_bits(
        text_config,
        "rms_norm_eps",
        target_config.rms_norm_epsilon_bits(),
    )?;
    compare_f32_bits(
        text_config,
        "partial_rotary_factor",
        target_config.partial_rotary_factor_bits(),
    )?;
    compare_u32(
        text_config,
        "linear_conv_kernel_dim",
        target_config.linear_convolution_kernel_dimension(),
    )?;
    compare_u32(
        text_config,
        "linear_num_key_heads",
        target_config.linear_key_head_count(),
    )?;
    compare_u32(
        text_config,
        "linear_num_value_heads",
        target_config.linear_value_head_count(),
    )?;
    compare_u32(
        text_config,
        "linear_key_head_dim",
        target_config.linear_key_head_dimension(),
    )?;
    compare_u32(
        text_config,
        "linear_value_head_dim",
        target_config.linear_value_head_dimension(),
    )?;
    compare_layer_types(text_config, target_config.layer_types())?;
    compare_rope_theta(text_config, target_config.rope_theta_bits())?;
    compare_feed_forward_geometry(target_config, text_config)?;

    let target_tokenizer_digest =
        Qwen3_5Tokenizer::token_identifier_mapping_digest(target_tokenizer_bytes)
            .map_err(|_| Qwen3_5MtpPairingCompatibilityError::InvalidTokenizer)?;
    let drafter_tokenizer_digest =
        Qwen3_5Tokenizer::token_identifier_mapping_digest(drafter_tokenizer_bytes)
            .map_err(|_| Qwen3_5MtpPairingCompatibilityError::InvalidTokenizer)?;
    if target_tokenizer_digest != drafter_tokenizer_digest {
        return Err(Qwen3_5MtpPairingCompatibilityError::TokenizerMappingMismatch);
    }
    let maximum_draft_depth = drafter_config.maximum_draft_depth();
    let requested_draft_depth = configured_draft_depth.unwrap_or(1);
    if requested_draft_depth == 0 || requested_draft_depth > maximum_draft_depth {
        return Err(Qwen3_5MtpPairingCompatibilityError::UnsupportedDraftDepth);
    }
    Ok(Qwen3_5MtpPairingCompatibility {
        maximum_draft_depth,
        requested_draft_depth,
        quantization_profile: drafter_config.quantization_profile(),
    })
}

fn compare_typed_geometry(
    target_config: &Qwen3_5Config,
    drafter_config: &Qwen3_5StandaloneMtpConfig,
) -> Result<(), Qwen3_5MtpPairingCompatibilityError> {
    for (field_name, target_value, drafter_value) in [
        (
            "hidden_size",
            target_config.hidden_size(),
            drafter_config.hidden_size(),
        ),
        (
            "vocab_size",
            target_config.vocabulary_size(),
            drafter_config.vocabulary_size(),
        ),
        (
            "mtp_num_hidden_layers",
            target_config.mtp_layer_count(),
            drafter_config.mtp_layer_count(),
        ),
    ] {
        if target_value != drafter_value {
            return Err(Qwen3_5MtpPairingCompatibilityError::FieldMismatch { field_name });
        }
    }
    if target_config.feed_forward_architecture() != drafter_config.feed_forward_architecture() {
        return Err(Qwen3_5MtpPairingCompatibilityError::FieldMismatch {
            field_name: "model_type",
        });
    }
    if target_config.has_tied_embeddings() != drafter_config.has_tied_embeddings() {
        return Err(Qwen3_5MtpPairingCompatibilityError::FieldMismatch {
            field_name: "tie_word_embeddings",
        });
    }
    Ok(())
}

fn compare_feed_forward_geometry(
    target_config: &Qwen3_5Config,
    text_config: &serde_json::Value,
) -> Result<(), Qwen3_5MtpPairingCompatibilityError> {
    match target_config.feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => compare_u32(
            text_config,
            "intermediate_size",
            target_config.dense_intermediate_size(),
        ),
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
            compare_u32(text_config, "num_experts", target_config.expert_count())?;
            compare_u32(
                text_config,
                "num_experts_per_tok",
                target_config.experts_per_token(),
            )?;
            compare_u32(
                text_config,
                "moe_intermediate_size",
                target_config.expert_intermediate_size(),
            )?;
            compare_u32(
                text_config,
                "shared_expert_intermediate_size",
                target_config.shared_expert_intermediate_size(),
            )
        }
    }
}

fn compare_u32(
    document: &serde_json::Value,
    field_name: &'static str,
    expected: u32,
) -> Result<(), Qwen3_5MtpPairingCompatibilityError> {
    let actual = document
        .get(field_name)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or(Qwen3_5MtpPairingCompatibilityError::MissingField { field_name })?;
    (actual == expected)
        .then_some(())
        .ok_or(Qwen3_5MtpPairingCompatibilityError::FieldMismatch { field_name })
}

fn compare_bool(
    document: &serde_json::Value,
    field_name: &'static str,
    expected: bool,
) -> Result<(), Qwen3_5MtpPairingCompatibilityError> {
    let actual = document
        .get(field_name)
        .and_then(serde_json::Value::as_bool)
        .ok_or(Qwen3_5MtpPairingCompatibilityError::MissingField { field_name })?;
    (actual == expected)
        .then_some(())
        .ok_or(Qwen3_5MtpPairingCompatibilityError::FieldMismatch { field_name })
}

fn compare_string(
    document: &serde_json::Value,
    field_name: &'static str,
    expected: &str,
) -> Result<(), Qwen3_5MtpPairingCompatibilityError> {
    let actual = document
        .get(field_name)
        .and_then(serde_json::Value::as_str)
        .ok_or(Qwen3_5MtpPairingCompatibilityError::MissingField { field_name })?;
    (actual == expected)
        .then_some(())
        .ok_or(Qwen3_5MtpPairingCompatibilityError::FieldMismatch { field_name })
}

fn compare_f32_bits(
    document: &serde_json::Value,
    field_name: &'static str,
    expected_bits: u32,
) -> Result<(), Qwen3_5MtpPairingCompatibilityError> {
    let actual_bits = document
        .get(field_name)
        .and_then(serde_json::Value::as_f64)
        .map(|value| (value as f32).to_bits())
        .ok_or(Qwen3_5MtpPairingCompatibilityError::MissingField { field_name })?;
    (actual_bits == expected_bits)
        .then_some(())
        .ok_or(Qwen3_5MtpPairingCompatibilityError::FieldMismatch { field_name })
}

fn compare_layer_types(
    document: &serde_json::Value,
    expected: &[String],
) -> Result<(), Qwen3_5MtpPairingCompatibilityError> {
    let actual = document
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or(Qwen3_5MtpPairingCompatibilityError::MissingField {
            field_name: "layer_types",
        })?
        .iter()
        .map(|value| value.as_str())
        .collect::<Option<Vec<_>>>()
        .ok_or(Qwen3_5MtpPairingCompatibilityError::MissingField {
            field_name: "layer_types",
        })?;
    (actual
        .iter()
        .copied()
        .eq(expected.iter().map(String::as_str)))
    .then_some(())
    .ok_or(Qwen3_5MtpPairingCompatibilityError::FieldMismatch {
        field_name: "layer_types",
    })
}

fn compare_rope_theta(
    document: &serde_json::Value,
    expected_bits: u32,
) -> Result<(), Qwen3_5MtpPairingCompatibilityError> {
    let rope_theta = document
        .get("rope_parameters")
        .and_then(|rope| rope.get("rope_theta"))
        .or_else(|| document.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .ok_or(Qwen3_5MtpPairingCompatibilityError::MissingField {
            field_name: "rope_theta",
        })?;
    ((rope_theta as f32).to_bits() == expected_bits)
        .then_some(())
        .ok_or(Qwen3_5MtpPairingCompatibilityError::FieldMismatch {
            field_name: "rope_theta",
        })
}

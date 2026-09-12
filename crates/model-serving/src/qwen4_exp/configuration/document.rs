//! The `qwen4_exp` wire configuration document and its quantization profile.
//!
//! Parsing accepts the documented field set and hands the validated geometry
//! to the owner in `config.rs`. Quantization is read here because it lives
//! beside the text configuration: a default profile plus any number of
//! per-tensor overrides. An unknown mode is a typed failure, never an
//! assumed affine fallback.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::config::Qwen4ExpConfig;
use super::error::Qwen4ExpConfigError;

/// How one tensor's weights are physically stored.
///
/// Block-scaled floating-point modes parse successfully because published
/// artifacts declare them; whether the runtime can execute them is a separate
/// binding decision, so validation never silently downgrades an artifact to
/// affine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen4ExpQuantizationMode {
    Affine,
    Nvfp4,
    Mxfp4,
}

impl Qwen4ExpQuantizationMode {
    fn parse(mode: &str) -> Option<Self> {
        match mode {
            "affine" => Some(Self::Affine),
            "nvfp4" => Some(Self::Nvfp4),
            "mxfp4" => Some(Self::Mxfp4),
            _ => None,
        }
    }

    /// The wire spelling of this mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Affine => "affine",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp4 => "mxfp4",
        }
    }
}

/// One quantization profile: bit width, group size, and mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpQuantizationProfile {
    pub bits: u32,
    pub group_size: u32,
    pub mode: Qwen4ExpQuantizationMode,
}

#[derive(Debug, Deserialize)]
pub(super) struct Qwen4ExpConfigDocument {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub model_type: String,
    pub text_config: Qwen4ExpTextConfigDocument,
}

/// Nested text configuration, before validation.
#[derive(Debug, Deserialize)]
pub(super) struct Qwen4ExpTextConfigDocument {
    pub model_type: String,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    #[serde(default)]
    pub layer_types: Vec<String>,
    pub full_attention_interval: Option<u32>,
    pub head_dim: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u64,
    pub output_gate_type: Option<String>,
    pub rms_norm_eps: Option<f64>,
    #[serde(default, rename = "dtype", alias = "torch_dtype")]
    pub activation_dtype: Option<String>,
    pub linear_num_key_heads: Option<u32>,
    pub linear_num_value_heads: Option<u32>,
    pub linear_key_head_dim: Option<u32>,
    pub linear_value_head_dim: Option<u32>,
    pub linear_conv_kernel_dim: Option<u32>,
    pub mamba_ssm_dtype: Option<String>,
    pub indexer_budget: Option<u32>,
    pub indexer_compress_ratio: Option<u32>,
    pub indexer_head_dim: Option<u32>,
    pub indexer_kv_heads: Option<u32>,
    pub indexer_n_heads: Option<u32>,
    pub hc_count: Option<u32>,
    pub hc_lowrank: Option<u32>,
    pub ngram_size: Option<u32>,
    pub heads_per_ngram: Option<u32>,
    pub ngram_vocab_size_base: Option<u64>,
    pub split_ngram_parts: Option<u32>,
    pub ple_embed_dim: Option<u32>,
    #[serde(default)]
    pub ple_layer_ids: Option<Vec<u32>>,
    pub ple_conv_kernel_size: Option<u32>,
    pub make_ngram_vocab_size_divisible_by: Option<u32>,
    pub mtp_num_hidden_layers: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl Qwen4ExpConfig {
    /// Parses and validates a configuration document from JSON bytes.
    ///
    /// The quantization document is read here because it lives beside the
    /// text configuration: a default profile plus any number of per-tensor
    /// overrides. An unknown mode is a typed failure, never an assumed
    /// affine fallback.
    ///
    /// # Errors
    /// When the bytes are not valid JSON or any value fails validation.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, Qwen4ExpConfigError> {
        let wire: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|_| Qwen4ExpConfigError::MissingField {
                field: "valid JSON",
            })?;
        let document: Qwen4ExpConfigDocument =
            serde_json::from_value(wire.clone()).map_err(|_| {
                Qwen4ExpConfigError::MissingField {
                    field: "the declared field set",
                }
            })?;
        let mut per_tensor = BTreeMap::new();
        let mut default_profile = None;
        if let Some(quantization) = wire
            .get("quantization")
            .or_else(|| wire.get("quantization_config"))
            && let Some(object) = quantization.as_object()
        {
            for (name, value) in object {
                if matches!(
                    name.as_str(),
                    "bits" | "group_size" | "mode" | "quant_method"
                ) {
                    continue;
                }
                let bits = value.get("bits").and_then(serde_json::Value::as_u64);
                let group_size = value.get("group_size").and_then(serde_json::Value::as_u64);
                let mode = value.get("mode").and_then(serde_json::Value::as_str);
                let (Some(bits), Some(group_size)) = (bits, group_size) else {
                    return Err(Qwen4ExpConfigError::MissingField {
                        field: "bits and group_size in every quantization override",
                    });
                };
                let mode = mode.unwrap_or("affine");
                let profile = Qwen4ExpQuantizationProfile {
                    bits: u32::try_from(bits).map_err(|_| Qwen4ExpConfigError::MissingField {
                        field: "a bits value within u32",
                    })?,
                    group_size: u32::try_from(group_size).map_err(|_| {
                        Qwen4ExpConfigError::MissingField {
                            field: "a group_size value within u32",
                        }
                    })?,
                    mode: Qwen4ExpQuantizationMode::parse(mode).ok_or_else(|| {
                        Qwen4ExpConfigError::UnknownQuantizationMode {
                            provided: mode.to_owned(),
                        }
                    })?,
                };
                per_tensor.insert(name.clone(), profile);
            }
            let default_bits = object.get("bits").and_then(serde_json::Value::as_u64);
            let default_group = object.get("group_size").and_then(serde_json::Value::as_u64);
            if let (Some(bits), Some(group_size)) = (default_bits, default_group) {
                let mode = object
                    .get("mode")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("affine");
                default_profile = Some(Qwen4ExpQuantizationProfile {
                    bits: u32::try_from(bits).map_err(|_| Qwen4ExpConfigError::MissingField {
                        field: "a bits value within u32",
                    })?,
                    group_size: u32::try_from(group_size).map_err(|_| {
                        Qwen4ExpConfigError::MissingField {
                            field: "a group_size value within u32",
                        }
                    })?,
                    mode: Qwen4ExpQuantizationMode::parse(mode).ok_or_else(|| {
                        Qwen4ExpConfigError::UnknownQuantizationMode {
                            provided: mode.to_owned(),
                        }
                    })?,
                });
            }
        }
        Self::validate(document, default_profile, per_tensor)
    }
}

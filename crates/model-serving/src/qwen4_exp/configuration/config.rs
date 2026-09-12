//! Validated `qwen4_exp` text configuration.
//!
//! One owner turns the parsed wire document into typed geometry and refuses
//! anything unrecognized instead of assuming a default. Published artifacts
//! of this family differ on expert count, quantization profile, per-tensor
//! overrides, and lookup-table form, so every value here is read from the
//! document; a literal that belongs to one artifact is a defect (see
//! `docs/qwen4-exp-architecture.md`).

use std::collections::BTreeMap;

use super::document::{Qwen4ExpConfigDocument, Qwen4ExpQuantizationProfile};
use super::error::Qwen4ExpConfigError;

/// Validated `qwen4_exp` text configuration.
///
/// Fields the execution path will consume are grouped by subsystem; optional
/// subsystems stay optional so a text-only or head-less variant validates the
/// same way as a full one. A partially declared group is a failure, because a
/// subsystem that silently vanishes while its tensors still ship is worse than
/// a rejected artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpConfig {
    pub hidden_size: u32,
    pub decoder_layers: u32,
    pub layer_schedule: Vec<Qwen4ExpLayerKind>,
    pub full_attention_interval: u32,
    pub head_dim: u32,
    pub attention_heads: u32,
    pub key_value_heads: u32,
    pub vocabulary_size: u32,
    pub context_window_tokens: u64,
    pub rms_norm_epsilon: f64,
    pub activation_dtype: String,
    pub eos_token_id: u32,
    pub linear_attention: Option<Qwen4ExpLinearAttentionConfig>,
    pub sparse_attention: Option<Qwen4ExpSparseAttentionConfig>,
    pub hyper_connections: Option<Qwen4ExpHyperConnectionConfig>,
    pub ngram_embedding: Option<Qwen4ExpNgramConfig>,
    pub multi_token_prediction_layers: Option<u32>,
    pub default_quantization: Option<Qwen4ExpQuantizationProfile>,
    pub per_tensor_quantization: BTreeMap<String, Qwen4ExpQuantizationProfile>,
}

/// The two decoder layer kinds this family alternates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen4ExpLayerKind {
    LinearAttention,
    FullAttention,
}

/// Gated-delta linear-attention geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpLinearAttentionConfig {
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel_dim: u32,
}

/// Sparse-attention indexer geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpSparseAttentionConfig {
    pub budget: u32,
    pub compress_ratio: u32,
    pub head_dim: u32,
    pub key_value_heads: u32,
    pub heads: u32,
}

/// Hyper-connection stream geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpHyperConnectionConfig {
    pub stream_count: u32,
    pub low_rank: u32,
}

/// N-gram embedding table geometry, one-based layer identifiers preserved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen4ExpNgramConfig {
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub vocabulary_base: u64,
    pub split_parts: u32,
    pub embedding_dim: u32,
    pub layer_ids_one_based: Vec<u32>,
    pub conv_kernel_size: u32,
    pub vocabulary_divisor: u32,
}

const EXPECTED_ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";
const EXPECTED_MODEL_TYPE: &str = "qwen4_exp";
const EXPECTED_TEXT_MODEL_TYPE: &str = "qwen4_exp_text";
const SUPPORTED_ACTIVATION_DTYPES: [&str; 3] = ["bfloat16", "float16", "float32"];
const SUPPORTED_OUTPUT_GATE_TYPES: [&str; 1] = ["sigmoid"];
const SUPPORTED_MAMBA_SSM_DTYPES: [&str; 1] = ["float32"];

/// A field group is either fully declared or fully absent; a partial group
/// would let one subsystem silently vanish while its tensors still ship.
fn complete_group(group: &'static str, present: &[bool]) -> Result<bool, Qwen4ExpConfigError> {
    let declared = present.iter().filter(|declared| **declared).count();
    if declared == 0 {
        return Ok(false);
    }
    if declared == present.len() {
        return Ok(true);
    }
    Err(Qwen4ExpConfigError::PartialFieldGroup { group })
}

impl Qwen4ExpConfig {
    /// Validates a parsed document into typed geometry.
    ///
    /// # Errors
    /// When any value is unrecognized, inconsistent, or missing.
    pub(super) fn validate(
        document: Qwen4ExpConfigDocument,
        default_quantization: Option<Qwen4ExpQuantizationProfile>,
        per_tensor_quantization: BTreeMap<String, Qwen4ExpQuantizationProfile>,
    ) -> Result<Self, Qwen4ExpConfigError> {
        if let Some(architectures) = document.architectures.first()
            && architectures != EXPECTED_ARCHITECTURE
        {
            return Err(Qwen4ExpConfigError::UnsupportedArchitecture {
                provided: architectures.clone(),
            });
        }
        if document.model_type != EXPECTED_MODEL_TYPE
            && document.model_type != EXPECTED_TEXT_MODEL_TYPE
        {
            return Err(Qwen4ExpConfigError::UnsupportedModelType {
                provided: document.model_type,
            });
        }
        let text = document.text_config;
        if text.model_type != EXPECTED_TEXT_MODEL_TYPE {
            return Err(Qwen4ExpConfigError::UnsupportedTextModelType {
                provided: text.model_type,
            });
        }
        let activation_dtype = text
            .activation_dtype
            .clone()
            .unwrap_or_else(|| "bfloat16".to_owned());
        if !SUPPORTED_ACTIVATION_DTYPES.contains(&activation_dtype.as_str()) {
            return Err(Qwen4ExpConfigError::UnsupportedActivationDtype {
                provided: activation_dtype,
            });
        }
        let output_gate_type = text
            .output_gate_type
            .clone()
            .unwrap_or_else(|| "sigmoid".to_owned());
        if !SUPPORTED_OUTPUT_GATE_TYPES.contains(&output_gate_type.as_str()) {
            return Err(Qwen4ExpConfigError::UnsupportedOutputGateType {
                provided: output_gate_type,
            });
        }
        if let Some(mamba_ssm_dtype) = &text.mamba_ssm_dtype
            && !SUPPORTED_MAMBA_SSM_DTYPES.contains(&mamba_ssm_dtype.as_str())
        {
            return Err(Qwen4ExpConfigError::UnsupportedMambaSsmDtype {
                provided: mamba_ssm_dtype.clone(),
            });
        }

        let mut layer_schedule = Vec::with_capacity(text.layer_types.len());
        for entry in &text.layer_types {
            let kind = match entry.as_str() {
                "linear_attention" => Qwen4ExpLayerKind::LinearAttention,
                "full_attention" => Qwen4ExpLayerKind::FullAttention,
                other => {
                    return Err(Qwen4ExpConfigError::UnknownLayerKind {
                        provided: other.to_owned(),
                    });
                }
            };
            layer_schedule.push(kind);
        }
        if !layer_schedule.is_empty() && layer_schedule.len() != text.num_hidden_layers as usize {
            return Err(Qwen4ExpConfigError::LayerScheduleLengthMismatch {
                declared_layers: text.num_hidden_layers,
                schedule_length: layer_schedule.len(),
            });
        }
        let full_attention_layers = layer_schedule
            .iter()
            .filter(|kind| **kind == Qwen4ExpLayerKind::FullAttention)
            .count() as u32;
        let full_attention_interval = text
            .full_attention_interval
            .unwrap_or(text.num_hidden_layers.max(1));
        if full_attention_interval > 0
            && full_attention_layers != text.num_hidden_layers / full_attention_interval
        {
            return Err(Qwen4ExpConfigError::LayerScheduleIntervalMismatch {
                interval: full_attention_interval,
                full_attention_layers,
                declared_layers: text.num_hidden_layers,
            });
        }

        let ngram_embedding = match (
            text.ngram_size,
            text.heads_per_ngram,
            text.ngram_vocab_size_base,
            text.split_ngram_parts,
            text.ple_embed_dim,
            text.ple_conv_kernel_size,
            text.make_ngram_vocab_size_divisible_by,
        ) {
            (
                Some(ngram_size),
                Some(heads_per_ngram),
                Some(vocabulary_base),
                Some(split_parts),
                Some(embedding_dim),
                Some(conv_kernel_size),
                Some(vocabulary_divisor),
            ) => {
                if ngram_size < 2 {
                    return Err(Qwen4ExpConfigError::NgramSizeTooSmall {
                        provided: ngram_size,
                    });
                }
                let head_count = (ngram_size - 1) * heads_per_ngram;
                if head_count == 0 || embedding_dim % head_count != 0 {
                    return Err(Qwen4ExpConfigError::NgramHeadDimMismatch {
                        embedding_dim,
                        head_count,
                    });
                }
                let layer_ids_one_based = text.ple_layer_ids.clone().unwrap_or_default();
                for layer_id in &layer_ids_one_based {
                    if *layer_id == 0 || *layer_id > text.num_hidden_layers {
                        return Err(Qwen4ExpConfigError::InvalidPleLayerId {
                            provided: *layer_id,
                            declared_layers: text.num_hidden_layers,
                        });
                    }
                }
                Some(Qwen4ExpNgramConfig {
                    ngram_size,
                    heads_per_ngram,
                    vocabulary_base,
                    split_parts,
                    embedding_dim,
                    layer_ids_one_based,
                    conv_kernel_size,
                    vocabulary_divisor,
                })
            }
            (None, None, None, None, None, None, None) => None,
            _ => {
                return Err(Qwen4ExpConfigError::PartialFieldGroup {
                    group: "n-gram embedding",
                });
            }
        };

        let linear_attention = complete_group(
            "linear attention",
            &[
                text.linear_num_key_heads.is_some(),
                text.linear_num_value_heads.is_some(),
                text.linear_key_head_dim.is_some(),
                text.linear_value_head_dim.is_some(),
                text.linear_conv_kernel_dim.is_some(),
            ],
        )?
        .then(|| Qwen4ExpLinearAttentionConfig {
            key_heads: text
                .linear_num_key_heads
                .expect("group completeness checked"),
            value_heads: text
                .linear_num_value_heads
                .expect("group completeness checked"),
            key_head_dim: text
                .linear_key_head_dim
                .expect("group completeness checked"),
            value_head_dim: text
                .linear_value_head_dim
                .expect("group completeness checked"),
            conv_kernel_dim: text
                .linear_conv_kernel_dim
                .expect("group completeness checked"),
        });
        let sparse_attention = complete_group(
            "sparse-attention indexer",
            &[
                text.indexer_budget.is_some(),
                text.indexer_compress_ratio.is_some(),
                text.indexer_head_dim.is_some(),
                text.indexer_kv_heads.is_some(),
                text.indexer_n_heads.is_some(),
            ],
        )?
        .then(|| Qwen4ExpSparseAttentionConfig {
            budget: text.indexer_budget.expect("group completeness checked"),
            compress_ratio: text
                .indexer_compress_ratio
                .expect("group completeness checked"),
            head_dim: text.indexer_head_dim.expect("group completeness checked"),
            key_value_heads: text.indexer_kv_heads.expect("group completeness checked"),
            heads: text.indexer_n_heads.expect("group completeness checked"),
        });
        let hyper_connections = complete_group(
            "hyper-connection",
            &[text.hc_count.is_some(), text.hc_lowrank.is_some()],
        )?
        .then(|| Qwen4ExpHyperConnectionConfig {
            stream_count: text.hc_count.expect("group completeness checked"),
            low_rank: text.hc_lowrank.expect("group completeness checked"),
        });

        Ok(Self {
            hidden_size: text.hidden_size,
            decoder_layers: text.num_hidden_layers,
            layer_schedule,
            full_attention_interval,
            head_dim: text.head_dim,
            attention_heads: text.num_attention_heads,
            key_value_heads: text.num_key_value_heads,
            vocabulary_size: text.vocab_size,
            context_window_tokens: text.max_position_embeddings,
            rms_norm_epsilon: text.rms_norm_eps.unwrap_or(1.0e-6),
            activation_dtype,
            eos_token_id: text.eos_token_id.unwrap_or_default(),
            linear_attention,
            sparse_attention,
            hyper_connections,
            ngram_embedding,
            multi_token_prediction_layers: text.mtp_num_hidden_layers,
            default_quantization,
            per_tensor_quantization,
        })
    }
}

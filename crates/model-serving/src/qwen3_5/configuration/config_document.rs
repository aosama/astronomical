use serde::Deserialize;

use super::{
    config::Qwen3_5FeedForwardArchitecture,
    config_validation::{
        Qwen3_5ConfigError, deserialize_f32_bits, deserialize_optional_eos_token_ids,
        deserialize_optional_f32_bits, validate_exact_boolean, validate_exact_value,
    },
    quantizations::optiq::{OptiQQuantizationConfig, QuantizationConfigSource},
};

const EXPECTED_MOE_TEXT_MODEL_TYPE: &str = "qwen3_5_moe_text";
const EXPECTED_DENSE_TEXT_MODEL_TYPE: &str = "qwen3_5_text";
const EXPECTED_HIDDEN_ACTIVATION: &str = "silu";

/// Qwen MTP sidecar declaration parsed from `mlx_lm_extra_tensors`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub(super) struct MlxLmExtraTensors {
    /// Path to the MTP sidecar safetensors file, relative to the model directory.
    /// E.g., "mtp.safetensors" or "optiq/mtp.safetensors".
    #[serde(default, rename = "mtp_file")]
    pub(super) mtp_file: Option<String>,
}

/// Global MTP quantization parameters declared in the top-level config.
/// Provides default bit width and group size for MTP modules that lack
/// per-module overrides in the `quantization` dict. Absent when the
/// model does not declare quantized MTP.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(super) struct MtplxMtpQuantization {
    #[serde(default)]
    pub(super) bits: u32,
    #[serde(default)]
    pub(super) group_size: u32,
}

/// Private wire schema retained only while validating one model config document.
#[derive(Debug, Deserialize)]
pub(super) struct Qwen3_5ConfigDocument {
    pub(super) architectures: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_optional_eos_token_ids")]
    pub(super) eos_token_id: Option<Vec<u32>>,
    pub(super) model_type: String,
    #[serde(default)]
    pub(super) pad_token_id: Option<u32>,
    #[serde(default)]
    pub(super) quantization: Option<OptiQQuantizationConfig>,
    #[serde(default)]
    pub(super) quantization_config: Option<OptiQQuantizationConfig>,
    pub(super) text_config: Qwen3_5TextConfig,
    pub(super) tie_word_embeddings: bool,
    #[serde(rename = "dtype", alias = "torch_dtype", default)]
    pub(super) activation_dtype: Option<String>,
    /// Sidecar file declarations from config.json's `mlx_lm_extra_tensors` field.
    /// Absent when models store all tensors in the shard index.
    #[serde(default)]
    pub(super) mlx_lm_extra_tensors: Option<MlxLmExtraTensors>,

    /// Top-level MTP sidecar path declared directly in config.json.
    /// Provides a fallback when `mlx_lm_extra_tensors` is absent.
    #[serde(default, rename = "mtp_file")]
    pub(super) mtp_file: Option<String>,

    /// Global MTP quantization parameters for prequantized MTP sidecars.
    /// E.g. `{"bits": 4, "group_size": 64, "mode": "affine", "prequantized": true}`.
    /// Absent when the model does not declare quantized MTP.
    #[serde(default, rename = "mtplx_mtp_quantization")]
    pub(super) mtxplx_mtp_quantization: Option<MtplxMtpQuantization>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(super) struct Qwen3_5TextConfig {
    pub(super) attention_bias: bool,
    model_type: String,
    #[serde(rename = "dtype", alias = "torch_dtype", default)]
    pub(super) text_config_dtype: Option<String>,
    #[serde(
        rename = "eos_token_id",
        default,
        deserialize_with = "deserialize_optional_eos_token_ids"
    )]
    pub(super) text_config_eos_token_id: Option<Vec<u32>>,
    pub(super) hidden_act: String,
    pub(super) hidden_size: u32,
    pub(super) num_hidden_layers: u32,
    pub(super) num_attention_heads: u32,
    pub(super) num_key_value_heads: u32,
    pub(super) head_dim: u32,
    #[serde(rename = "rms_norm_eps", deserialize_with = "deserialize_f32_bits")]
    pub(super) rms_norm_epsilon_bits: u32,
    #[serde(
        rename = "rope_theta",
        default,
        deserialize_with = "deserialize_optional_f32_bits"
    )]
    pub(super) legacy_rope_theta_bits: Option<u32>,
    #[serde(default)]
    pub(super) rope_parameters: Option<Qwen3_5RopeParameters>,
    #[serde(
        rename = "partial_rotary_factor",
        default,
        deserialize_with = "deserialize_optional_f32_bits"
    )]
    pub(super) partial_rotary_factor_bits: Option<u32>,
    #[serde(default)]
    pub(super) mlp_bias: bool,
    #[serde(default = "default_true")]
    pub(super) norm_topk_prob: bool,
    pub(super) vocab_size: u32,
    pub(super) max_position_embeddings: u32,
    pub(super) layer_types: Vec<String>,
    pub(super) linear_conv_kernel_dim: u32,
    pub(super) linear_num_key_heads: u32,
    pub(super) linear_num_value_heads: u32,
    pub(super) linear_key_head_dim: u32,
    pub(super) linear_value_head_dim: u32,
    #[serde(default)]
    pub(super) num_experts: u32,
    #[serde(default)]
    pub(super) num_experts_per_tok: u32,
    #[serde(default)]
    pub(super) moe_intermediate_size: u32,
    #[serde(default)]
    pub(super) shared_expert_intermediate_size: u32,
    #[serde(default)]
    pub(super) intermediate_size: u32,
    pub(super) mtp_num_hidden_layers: u32,
    #[serde(default)]
    mamba_ssm_dtype: Option<String>,
}

impl QuantizationConfigSource for Qwen3_5TextConfig {
    fn layer_count(&self) -> u32 {
        self.num_hidden_layers
    }

    fn decoder_layer_is_full_attention(&self, decoder_layer_index: usize) -> bool {
        self.layer_types
            .get(decoder_layer_index)
            .is_some_and(|layer_type| layer_type == "full_attention")
    }
}

impl Qwen3_5TextConfig {
    pub(super) fn validate(
        &self,
        feed_forward_architecture: Qwen3_5FeedForwardArchitecture,
    ) -> Result<(), Qwen3_5ConfigError> {
        let expected_text_model_type = match feed_forward_architecture {
            Qwen3_5FeedForwardArchitecture::Dense => EXPECTED_DENSE_TEXT_MODEL_TYPE,
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts => EXPECTED_MOE_TEXT_MODEL_TYPE,
        };
        validate_exact_value(
            "text_config.model_type",
            &self.model_type,
            expected_text_model_type,
        )?;
        validate_exact_value(
            "text_config.hidden_act",
            &self.hidden_act,
            EXPECTED_HIDDEN_ACTIVATION,
        )?;
        if let Some(rope_parameters) = &self.rope_parameters {
            rope_parameters.validate()?;
        }
        validate_exact_boolean("text_config.attention_bias", self.attention_bias, false)?;
        validate_exact_boolean("text_config.mlp_bias", self.mlp_bias, false)?;
        validate_exact_boolean("text_config.norm_topk_prob", self.norm_topk_prob, true)?;
        if let Some(mamba_ssm_dtype) = self.mamba_ssm_dtype.as_deref()
            && !matches!(mamba_ssm_dtype, "bfloat16" | "float32")
        {
            return Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
                description: format!(
                    "text_config.mamba_ssm_dtype contains unsupported dtype '{mamba_ssm_dtype}'"
                ),
            });
        }
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
            || self.linear_conv_kernel_dim == 0
            || self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
            || self.linear_key_head_dim == 0
            || self.linear_value_head_dim == 0
        {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "text_config numeric fields must be positive",
            });
        }
        match feed_forward_architecture {
            Qwen3_5FeedForwardArchitecture::Dense if self.intermediate_size == 0 => {
                return Err(Qwen3_5ConfigError::InvalidConfigValue {
                    description: "text_config numeric fields must be positive",
                });
            }
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts
                if self.num_experts == 0
                    || self.num_experts_per_tok == 0
                    || self.moe_intermediate_size == 0
                    || self.shared_expert_intermediate_size == 0 =>
            {
                return Err(Qwen3_5ConfigError::InvalidConfigValue {
                    description: "text_config numeric fields must be positive",
                });
            }
            _ => {}
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "text_config.num_attention_heads must divide evenly by num_key_value_heads",
            });
        }
        if matches!(
            feed_forward_architecture,
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts
        ) && self.num_experts_per_tok > self.num_experts
        {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "text_config.num_experts_per_tok must not exceed num_experts",
            });
        }
        if matches!(
            feed_forward_architecture,
            Qwen3_5FeedForwardArchitecture::Dense
        ) && (self.num_experts != 0
            || self.num_experts_per_tok != 0
            || self.moe_intermediate_size != 0
            || self.shared_expert_intermediate_size != 0)
        {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "dense Qwen3.5 text config must not declare sparse-expert dimensions",
            });
        }
        let maximum_mlx_shape_dimension = i32::MAX as u64;
        if [
            self.linear_conv_kernel_dim,
            self.linear_num_value_heads,
            self.linear_value_head_dim,
            self.linear_key_head_dim,
        ]
        .into_iter()
        .any(|shape_dimension| u64::from(shape_dimension) > maximum_mlx_shape_dimension)
            || self.linear_convolution_state_dimension() > maximum_mlx_shape_dimension
        {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "linear-attention dimensions must fit the MLX signed 32-bit shape range",
            });
        }
        if self.layer_types.len() != self.num_hidden_layers as usize {
            return Err(Qwen3_5ConfigError::LayerTypeCountMismatch {
                actual_layer_type_count: self.layer_types.len(),
                expected_layer_type_count: self.num_hidden_layers as usize,
            });
        }
        for layer_type in &self.layer_types {
            if !matches!(layer_type.as_str(), "full_attention" | "linear_attention") {
                return Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
                    description: format!(
                        "text_config.layer_types contains unsupported attention type '{layer_type}'"
                    ),
                });
            }
        }
        Ok(())
    }

    pub(super) fn linear_convolution_state_dimension(&self) -> u64 {
        u64::from(self.linear_num_key_heads)
            .saturating_mul(u64::from(self.linear_key_head_dim))
            .saturating_mul(2)
            .saturating_add(
                u64::from(self.linear_num_value_heads)
                    .saturating_mul(u64::from(self.linear_value_head_dim)),
            )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(super) struct Qwen3_5RopeParameters {
    mrope_interleaved: bool,
    mrope_section: [u32; 3],
    #[serde(deserialize_with = "deserialize_f32_bits")]
    pub(super) partial_rotary_factor: u32,
    #[serde(rename = "rope_theta", deserialize_with = "deserialize_f32_bits")]
    pub(super) rope_theta_bits: u32,
    #[serde(rename = "type", alias = "rope_type")]
    rope_type: String,
}

impl Qwen3_5RopeParameters {
    fn validate(&self) -> Result<(), Qwen3_5ConfigError> {
        validate_exact_boolean(
            "text_config.rope_parameters.mrope_interleaved",
            self.mrope_interleaved,
            true,
        )?;
        if self.mrope_section != [11, 11, 10] {
            return Err(Qwen3_5ConfigError::MropeSectionMismatch {
                actual_section: self.mrope_section,
            });
        }
        validate_exact_value(
            "text_config.rope_parameters.type",
            &self.rope_type,
            "default",
        )
    }
}

const fn default_true() -> bool {
    true
}

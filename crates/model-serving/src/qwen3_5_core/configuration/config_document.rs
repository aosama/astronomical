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
    #[serde(default)]
    pub(super) norm_convention: Option<String>,
    pub(super) text_config: Qwen3_5TextConfig,
    pub(super) tie_word_embeddings: bool,
    #[serde(rename = "dtype", alias = "torch_dtype", default)]
    pub(super) activation_dtype: Option<String>,
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
        validate_exact_text_values(self, expected_text_model_type)?;
        // The rope-parameter contract is checked between the pinned wire tokens
        // and the boolean flags so a config reporting both failures still
        // surfaces the rope failure first, matching the original pass order.
        if let Some(rope_parameters) = &self.rope_parameters {
            rope_parameters.validate()?;
        }
        validate_boolean_flags_and_mamba_dtype(self)?;
        validate_numeric_fields_are_positive(self, feed_forward_architecture)?;
        validate_attention_and_expert_head_counts(self, feed_forward_architecture)?;
        validate_linear_attention_shape_bounds(self)?;
        validate_layer_types(self)?;
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

/// Rejects text fields that must exactly equal the wire tokens this executor
/// implements; the expected model type depends on the declared architecture.
fn validate_exact_text_values(
    text_config: &Qwen3_5TextConfig,
    expected_text_model_type: &'static str,
) -> Result<(), Qwen3_5ConfigError> {
    validate_exact_value(
        "text_config.model_type",
        &text_config.model_type,
        expected_text_model_type,
    )?;
    validate_exact_value(
        "text_config.hidden_act",
        &text_config.hidden_act,
        EXPECTED_HIDDEN_ACTIVATION,
    )
}

/// Checks the boolean implementation contracts and the mamba SSM dtype
/// allowlist; both are pinned because the execution kernels support no
/// alternatives.
fn validate_boolean_flags_and_mamba_dtype(
    text_config: &Qwen3_5TextConfig,
) -> Result<(), Qwen3_5ConfigError> {
    validate_exact_boolean(
        "text_config.attention_bias",
        text_config.attention_bias,
        false,
    )?;
    validate_exact_boolean("text_config.mlp_bias", text_config.mlp_bias, false)?;
    validate_exact_boolean(
        "text_config.norm_topk_prob",
        text_config.norm_topk_prob,
        true,
    )?;
    if let Some(mamba_ssm_dtype) = text_config.mamba_ssm_dtype.as_deref()
        && !matches!(mamba_ssm_dtype, "bfloat16" | "float32")
    {
        return Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
            description: format!(
                "text_config.mamba_ssm_dtype contains unsupported dtype '{mamba_ssm_dtype}'"
            ),
        });
    }
    Ok(())
}

/// Rejects zero-valued execution dimensions. The shared text-config fields are
/// checked first, then each feed-forward family adds its own required-positive
/// fields, so the shared failure is reported even when a family field is also
/// zero.
fn validate_numeric_fields_are_positive(
    text_config: &Qwen3_5TextConfig,
    feed_forward_architecture: Qwen3_5FeedForwardArchitecture,
) -> Result<(), Qwen3_5ConfigError> {
    if text_config.hidden_size == 0
        || text_config.num_hidden_layers == 0
        || text_config.num_attention_heads == 0
        || text_config.num_key_value_heads == 0
        || text_config.head_dim == 0
        || text_config.vocab_size == 0
        || text_config.max_position_embeddings == 0
        || text_config.linear_conv_kernel_dim == 0
        || text_config.linear_num_key_heads == 0
        || text_config.linear_num_value_heads == 0
        || text_config.linear_key_head_dim == 0
        || text_config.linear_value_head_dim == 0
    {
        return Err(Qwen3_5ConfigError::InvalidConfigValue {
            description: "text_config numeric fields must be positive",
        });
    }
    match feed_forward_architecture {
        Qwen3_5FeedForwardArchitecture::Dense if text_config.intermediate_size == 0 => {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "text_config numeric fields must be positive",
            });
        }
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts
            if text_config.num_experts == 0
                || text_config.num_experts_per_tok == 0
                || text_config.moe_intermediate_size == 0
                || text_config.shared_expert_intermediate_size == 0 =>
        {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "text_config numeric fields must be positive",
            });
        }
        _ => {}
    }
    Ok(())
}

/// Checks head-count divisibility and the per-family expert-dimension
/// invariants: sparse routing must stay within the expert count, and a dense
/// checkpoint must not carry leftover sparse-expert dimensions.
fn validate_attention_and_expert_head_counts(
    text_config: &Qwen3_5TextConfig,
    feed_forward_architecture: Qwen3_5FeedForwardArchitecture,
) -> Result<(), Qwen3_5ConfigError> {
    if !text_config
        .num_attention_heads
        .is_multiple_of(text_config.num_key_value_heads)
    {
        return Err(Qwen3_5ConfigError::InvalidConfigValue {
            description: "text_config.num_attention_heads must divide evenly by num_key_value_heads",
        });
    }
    if matches!(
        feed_forward_architecture,
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts
    ) && text_config.num_experts_per_tok > text_config.num_experts
    {
        return Err(Qwen3_5ConfigError::InvalidConfigValue {
            description: "text_config.num_experts_per_tok must not exceed num_experts",
        });
    }
    if matches!(
        feed_forward_architecture,
        Qwen3_5FeedForwardArchitecture::Dense
    ) && (text_config.num_experts != 0
        || text_config.num_experts_per_tok != 0
        || text_config.moe_intermediate_size != 0
        || text_config.shared_expert_intermediate_size != 0)
    {
        return Err(Qwen3_5ConfigError::InvalidConfigValue {
            description: "dense Qwen3.5 text config must not declare sparse-expert dimensions",
        });
    }
    Ok(())
}

/// MLX shape dimensions are signed 32-bit, so each individual linear-attention
/// dimension and their derived convolution state width must fit that range;
/// the state width can overflow even when every individual dimension fits.
fn validate_linear_attention_shape_bounds(
    text_config: &Qwen3_5TextConfig,
) -> Result<(), Qwen3_5ConfigError> {
    let maximum_mlx_shape_dimension = i32::MAX as u64;
    if [
        text_config.linear_conv_kernel_dim,
        text_config.linear_num_value_heads,
        text_config.linear_value_head_dim,
        text_config.linear_key_head_dim,
    ]
    .into_iter()
    .any(|shape_dimension| u64::from(shape_dimension) > maximum_mlx_shape_dimension)
        || text_config.linear_convolution_state_dimension() > maximum_mlx_shape_dimension
    {
        return Err(Qwen3_5ConfigError::InvalidConfigValue {
            description: "linear-attention dimensions must fit the MLX signed 32-bit shape range",
        });
    }
    Ok(())
}

/// Every decoder layer must declare exactly one supported attention kind, and
/// the declaration count must match the decoder depth; the count mismatch is
/// reported before any unsupported entry.
fn validate_layer_types(text_config: &Qwen3_5TextConfig) -> Result<(), Qwen3_5ConfigError> {
    if text_config.layer_types.len() != text_config.num_hidden_layers as usize {
        return Err(Qwen3_5ConfigError::LayerTypeCountMismatch {
            actual_layer_type_count: text_config.layer_types.len(),
            expected_layer_type_count: text_config.num_hidden_layers as usize,
        });
    }
    for layer_type in &text_config.layer_types {
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

#[cfg(test)]
#[path = "config_document_tests.rs"]
mod config_document_tests;

use std::collections::BTreeMap;

use super::config_document::{Qwen3_5ConfigDocument, Qwen3_5TextConfig};
use super::config_validation::{
    QWEN_CHAT_EOS_TOKEN_ID, Qwen3_5ConfigError, validate_exact_boolean, validate_exact_value,
};
use super::quantizations::optiq::OptiQQuantizationProfile;

const EXPECTED_MOE_ARCHITECTURE: &str = "Qwen3_5MoeForConditionalGeneration";
const EXPECTED_DENSE_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const EXPECTED_MOE_MODEL_TYPE: &str = "qwen3_5_moe";
const EXPECTED_DENSE_MODEL_TYPE: &str = "qwen3_5";
const EXPECTED_TORCH_DTYPE: &str = "bfloat16";

/// Physical representation of executable model weights.
///
/// Native BF16 tensors use ordinary dense MLX operations. Affine-quantized
/// tensors require packed-weight matrix multiplication with scales and biases.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelWeightStorage {
    NativeBfloat16,
    AffineQuantized,
}

/// Feed-forward implementation declared by a validated Qwen3.5 checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5FeedForwardArchitecture {
    Dense,
    MixtureOfExperts,
}

/// Validated Qwen3.5 text configuration used to derive execution shapes and behavior.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5Config {
    eos_token_ids: Vec<u32>,
    has_tied_embeddings: bool,
    quantized_module_profiles: BTreeMap<String, OptiQQuantizationProfile>,
    mtp_quantized_module_profiles: BTreeMap<String, OptiQQuantizationProfile>,
    model_weight_storage: ModelWeightStorage,
    default_quantization_bits: u32,
    default_quantization_group_size: u32,
    text_config: Qwen3_5TextConfig,
    activation_dtype: String,
    feed_forward_architecture: Qwen3_5FeedForwardArchitecture,
}

impl Qwen3_5Config {
    /// Parses the config bytes retained by validated artifact ownership.
    pub fn from_json_bytes(config_bytes: &[u8]) -> Result<Self, Qwen3_5ConfigError> {
        let config_document = serde_json::from_slice::<Qwen3_5ConfigDocument>(config_bytes)
            .map_err(Qwen3_5ConfigError::DeserializeConfig)?;
        let feed_forward_architecture = match config_document.model_type.as_str() {
            EXPECTED_DENSE_MODEL_TYPE => Qwen3_5FeedForwardArchitecture::Dense,
            EXPECTED_MOE_MODEL_TYPE => Qwen3_5FeedForwardArchitecture::MixtureOfExperts,
            _ => {
                return Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
                    description: format!(
                        "model_type must be '{EXPECTED_DENSE_MODEL_TYPE}' or '{EXPECTED_MOE_MODEL_TYPE}'"
                    ),
                });
            }
        };
        let expected_architecture = match feed_forward_architecture {
            Qwen3_5FeedForwardArchitecture::Dense => EXPECTED_DENSE_ARCHITECTURE,
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts => EXPECTED_MOE_ARCHITECTURE,
        };
        validate_exact_value(
            "architectures",
            &config_document.architectures.join(","),
            expected_architecture,
        )?;
        // Resolve activation dtype: prefer top-level, fall back to text_config.dtype.
        // Qwen3.6 places `dtype` inside `text_config` instead of at the top level.
        let activation_dtype = config_document
            .activation_dtype
            .or(config_document.text_config.text_config_dtype.clone())
            .ok_or(Qwen3_5ConfigError::MissingActivationDtype)?;
        validate_exact_value("dtype", &activation_dtype, EXPECTED_TORCH_DTYPE)?;
        // Source lineage: MLX-VLM's Qwen3.5 configuration compatibility logic
        // (MIT License; see third-party license notices). Resolve eos_token_ids:
        // 1. Use top-level eos_token_id if present
        // 2. Otherwise fall back to text_config.eos_token_id
        // 3. Normalize single integers to arrays
        // 4. Append the Qwen chat EOS token (248046) if not already in the list
        // 5. Retain every declared stop token.
        let resolved_eos_token_ids = match config_document.eos_token_id {
            Some(token_ids) => token_ids,
            None => config_document
                .text_config
                .text_config_eos_token_id
                .clone()
                .ok_or(Qwen3_5ConfigError::MissingEosTokenId)?,
        };
        // Append QWEN_CHAT_EOS_TOKEN_ID if not already present
        let mut resolved_eos_token_ids = resolved_eos_token_ids;
        if !resolved_eos_token_ids.contains(&QWEN_CHAT_EOS_TOKEN_ID) {
            resolved_eos_token_ids.push(QWEN_CHAT_EOS_TOKEN_ID);
        }
        if resolved_eos_token_ids.is_empty() {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "eos_token_id must contain at least one token ID",
            });
        }
        if resolved_eos_token_ids.len() == 1
            && let Some(pad_token_id) = config_document.pad_token_id
            && !resolved_eos_token_ids.contains(&pad_token_id)
        {
            resolved_eos_token_ids.push(pad_token_id);
        }
        let eos_token_ids = resolved_eos_token_ids;
        validate_exact_boolean(
            "tie_word_embeddings",
            config_document.tie_word_embeddings,
            false,
        )?;
        config_document
            .text_config
            .validate(feed_forward_architecture)?;
        let (
            quantized_module_profiles,
            mtp_quantized_module_profiles,
            model_weight_storage,
            default_quantization_bits,
            default_quantization_group_size,
        ) = match (
            config_document.quantization.as_ref(),
            config_document.quantization_config.as_ref(),
        ) {
            (None, None) => (
                BTreeMap::new(),
                BTreeMap::new(),
                ModelWeightStorage::NativeBfloat16,
                0,
                0,
            ),
            (quantization, quantization_config) => {
                if let (Some(quantization), Some(quantization_config)) =
                    (quantization, quantization_config)
                    && quantization != quantization_config
                {
                    return Err(Qwen3_5ConfigError::QuantizationCopiesDiffer);
                }
                let quantization = quantization.or(quantization_config).ok_or(
                    Qwen3_5ConfigError::InvalidConfigValueDynamic {
                        description: "quantization configuration is missing".to_owned(),
                    },
                )?;
                let quantized_module_profiles = quantization
                    .validate(&config_document.text_config, feed_forward_architecture)?;
                let mtp_quantized_module_profiles = quantization.mtp_quantized_module_profiles();
                (
                    quantized_module_profiles,
                    mtp_quantized_module_profiles,
                    ModelWeightStorage::AffineQuantized,
                    quantization.default_bits(),
                    quantization.default_group_size(),
                )
            }
        };
        Ok(Self {
            eos_token_ids,
            has_tied_embeddings: config_document.tie_word_embeddings,
            quantized_module_profiles,
            mtp_quantized_module_profiles,
            model_weight_storage,
            default_quantization_bits,
            default_quantization_group_size,
            text_config: config_document.text_config,
            activation_dtype,
            feed_forward_architecture,
        })
    }

    /// Returns the activation dtype declared by the model config.
    #[must_use]
    pub fn activation_dtype(&self) -> &str {
        &self.activation_dtype
    }

    /// Returns the feed-forward architecture declared by this checkpoint.
    #[must_use]
    pub const fn feed_forward_architecture(&self) -> Qwen3_5FeedForwardArchitecture {
        self.feed_forward_architecture
    }

    /// Returns the intermediate width for a dense Qwen3.5 SwiGLU MLP.
    #[must_use]
    pub const fn dense_intermediate_size(&self) -> u32 {
        self.text_config.intermediate_size
    }

    /// Returns the validated OptiQ quantization profile for every executable quantized module.
    #[must_use]
    pub const fn quantized_module_profiles(&self) -> &BTreeMap<String, OptiQQuantizationProfile> {
        &self.quantized_module_profiles
    }

    /// Returns the artifact-wide executable weight representation.
    #[must_use]
    pub const fn model_weight_storage(&self) -> ModelWeightStorage {
        self.model_weight_storage
    }

    /// Returns the quantization profile for a specific module, falling back to the
    /// default profile if the module is not found in the explicit overrides.
    #[must_use]
    pub fn quantization_profile_for_module(&self, module_name: &str) -> OptiQQuantizationProfile {
        self.mtp_quantized_module_profiles
            .get(module_name)
            .copied()
            .or_else(|| self.quantized_module_profiles.get(module_name).copied())
            .unwrap_or(OptiQQuantizationProfile {
                bits: self.default_quantization_bits,
                group_size: self.default_quantization_group_size,
            })
    }

    /// Returns the default quantization bit width for modules not in the override map.
    #[must_use]
    pub const fn default_quantization_bits(&self) -> u32 {
        self.default_quantization_bits
    }

    /// Resolves modules stored as native floating-point when both affine companion
    /// tensors are absent from the safetensors index. This handles mixed storage
    /// artifacts whose default quantization profile does not describe every module,
    /// including optional MTP modules.
    ///
    /// The `shard_tensor_names` parameter should contain all tensor names from the
    /// safetensors index (the weight_map keys).
    pub fn resolve_unquantized_modules_from_shard_index(
        &mut self,
        shard_tensor_names: &std::collections::BTreeSet<String>,
    ) {
        let native_module_names = shard_tensor_names
            .iter()
            .filter_map(|tensor_name| tensor_name.strip_suffix(".weight"))
            .filter(|module_name| {
                !shard_tensor_names.contains(&format!("{module_name}.scales"))
                    && !shard_tensor_names.contains(&format!("{module_name}.biases"))
            })
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for native_module_name in native_module_names {
            let native_quantization_profile = OptiQQuantizationProfile::unquantized();
            if native_module_name.starts_with("language_model.mtp.") {
                self.mtp_quantized_module_profiles
                    .insert(native_module_name, native_quantization_profile);
            } else {
                self.quantized_module_profiles
                    .insert(native_module_name, native_quantization_profile);
            }
        }
    }

    /// Returns the default quantization group size for modules not in the override map.
    #[must_use]
    pub const fn default_quantization_group_size(&self) -> u32 {
        self.default_quantization_group_size
    }

    /// Returns the activation dtype declared by the model config.
    #[must_use]
    pub fn torch_dtype(&self) -> &str {
        self.activation_dtype()
    }

    /// Returns the declared MLP activation.
    #[must_use]
    pub fn hidden_activation(&self) -> &str {
        &self.text_config.hidden_act
    }

    /// Returns the declared text hidden dimension.
    #[must_use]
    pub const fn hidden_size(&self) -> u32 {
        self.text_config.hidden_size
    }

    /// Returns the declared text decoder-layer count.
    #[must_use]
    pub const fn layer_count(&self) -> u32 {
        self.text_config.num_hidden_layers
    }

    /// Returns the declared tokenizer vocabulary size.
    #[must_use]
    pub const fn vocabulary_size(&self) -> u32 {
        self.text_config.vocab_size
    }

    /// Returns the native combined prompt and generation position count.
    #[must_use]
    pub const fn maximum_position_count(&self) -> u32 {
        self.text_config.max_position_embeddings
    }

    /// Returns the exact RMSNorm epsilon bits.
    #[must_use]
    pub const fn rms_norm_epsilon_bits(&self) -> u32 {
        self.text_config.rms_norm_epsilon_bits
    }

    /// Returns the exact RoPE base bits.
    #[must_use]
    pub fn rope_theta_bits(&self) -> u32 {
        self.text_config
            .rope_parameters
            .as_ref()
            .map(|rope_parameters| rope_parameters.rope_theta_bits)
            .or(self.text_config.legacy_rope_theta_bits)
            .unwrap_or_default()
    }

    /// Returns the exact partial rotary factor bits.
    /// Falls back to `rope_parameters.partial_rotary_factor` when the top-level
    /// `text_config.partial_rotary_factor` is absent, matching the Qwen3.5
    /// configuration compatibility rule above.
    #[must_use]
    pub fn partial_rotary_factor_bits(&self) -> u32 {
        self.text_config
            .partial_rotary_factor_bits
            .or_else(|| {
                self.text_config
                    .rope_parameters
                    .as_ref()
                    .map(|rope_parameters| rope_parameters.partial_rotary_factor)
            })
            .unwrap_or_default()
    }

    /// Returns every stop-token ID declared by the model config.
    #[must_use]
    pub fn end_of_sequence_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Returns whether attention projections use bias terms.
    #[must_use]
    pub const fn has_attention_bias(&self) -> bool {
        self.text_config.attention_bias
    }

    /// Returns whether MLP projections use bias terms.
    #[must_use]
    pub const fn has_mlp_bias(&self) -> bool {
        self.text_config.mlp_bias
    }

    /// Returns whether input and output embeddings are tied.
    #[must_use]
    pub const fn has_tied_embeddings(&self) -> bool {
        self.has_tied_embeddings
    }

    /// Returns whether selected MoE router probabilities are normalized.
    #[must_use]
    pub const fn normalizes_top_k_probabilities(&self) -> bool {
        self.text_config.norm_topk_prob
    }

    /// Returns the decoder-layer indexes that use full attention.
    #[must_use]
    pub fn full_attention_decoder_layer_indexes(&self) -> Vec<usize> {
        self.text_config
            .layer_types
            .iter()
            .enumerate()
            .filter_map(|(decoder_layer_index, decoder_layer_type)| {
                if decoder_layer_type == "full_attention" {
                    Some(decoder_layer_index)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the decoder-layer indexes that use linear attention.
    #[must_use]
    pub fn linear_attention_decoder_layer_indexes(&self) -> Vec<usize> {
        self.text_config
            .layer_types
            .iter()
            .enumerate()
            .filter_map(|(decoder_layer_index, decoder_layer_type)| {
                if decoder_layer_type == "linear_attention" {
                    Some(decoder_layer_index)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns whether the validated decoder layer uses full attention.
    #[must_use]
    pub fn decoder_layer_is_full_attention(&self, decoder_layer_index: usize) -> bool {
        self.text_config
            .layer_types
            .get(decoder_layer_index)
            .is_some_and(|decoder_layer_type| decoder_layer_type == "full_attention")
    }

    /// Returns the full-attention key/value head count.
    #[must_use]
    pub const fn key_value_head_count(&self) -> u32 {
        self.text_config.num_key_value_heads
    }

    /// Returns the full-attention key/value head dimension.
    #[must_use]
    pub const fn head_dimension(&self) -> u32 {
        self.text_config.head_dim
    }

    /// Returns the linear-attention convolution kernel dimension.
    #[must_use]
    pub const fn linear_convolution_kernel_dimension(&self) -> u32 {
        self.text_config.linear_conv_kernel_dim
    }

    /// Returns the linear-attention key head count.
    #[must_use]
    pub const fn linear_key_head_count(&self) -> u32 {
        self.text_config.linear_num_key_heads
    }

    /// Returns the linear-attention value head count.
    #[must_use]
    pub const fn linear_value_head_count(&self) -> u32 {
        self.text_config.linear_num_value_heads
    }

    /// Returns the linear-attention key head dimension.
    #[must_use]
    pub const fn linear_key_head_dimension(&self) -> u32 {
        self.text_config.linear_key_head_dim
    }

    /// Returns the validated convolution-state width used by linear attention.
    #[must_use]
    pub fn linear_convolution_state_dimension(&self) -> i32 {
        self.text_config.linear_convolution_state_dimension() as i32
    }

    /// Returns the linear-attention value head dimension.
    #[must_use]
    pub const fn linear_value_head_dimension(&self) -> u32 {
        self.text_config.linear_value_head_dim
    }

    /// Returns the full-attention query head count.
    #[must_use]
    pub const fn query_head_count(&self) -> u32 {
        self.text_config.num_attention_heads
    }

    /// Returns the total number of sparse experts.
    #[must_use]
    pub const fn expert_count(&self) -> u32 {
        self.text_config.num_experts
    }

    /// Returns the number of experts selected per token.
    #[must_use]
    pub const fn experts_per_token(&self) -> u32 {
        self.text_config.num_experts_per_tok
    }

    /// Returns the per-expert feed-forward intermediate size.
    #[must_use]
    pub const fn expert_intermediate_size(&self) -> u32 {
        self.text_config.moe_intermediate_size
    }

    /// Returns the shared expert feed-forward intermediate size.
    #[must_use]
    pub const fn shared_expert_intermediate_size(&self) -> u32 {
        self.text_config.shared_expert_intermediate_size
    }

    /// Returns the linear-attention key dimension (key heads * key head dim).
    #[must_use]
    pub const fn linear_key_dimension(&self) -> u32 {
        self.text_config
            .linear_num_key_heads
            .saturating_mul(self.text_config.linear_key_head_dim)
    }

    /// Returns the linear-attention value dimension (value heads * value head dim).
    #[must_use]
    pub const fn linear_value_dimension(&self) -> u32 {
        self.text_config
            .linear_num_value_heads
            .saturating_mul(self.text_config.linear_value_head_dim)
    }

    /// Returns the linear-attention convolution dimension
    /// (2 * key dimension + value dimension).
    #[must_use]
    pub const fn linear_convolution_dimension(&self) -> u32 {
        self.linear_key_dimension()
            .saturating_mul(2)
            .saturating_add(self.linear_value_dimension())
    }

    /// Returns the RoPE rotary dimension (head_dim * partial_rotary_factor).
    #[must_use]
    pub fn rotary_dimension(&self) -> u32 {
        let partial_rotary_factor = f32::from_bits(self.partial_rotary_factor_bits());
        (self.text_config.head_dim as f32 * partial_rotary_factor) as u32
    }

    /// Returns the artifact-declared MTP layer count.
    #[must_use]
    pub const fn mtp_layer_count(&self) -> u32 {
        self.text_config.mtp_num_hidden_layers
    }
}

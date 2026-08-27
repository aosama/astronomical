use std::collections::BTreeMap;

use super::config::{ModelWeightStorage, Qwen3_5Config, Qwen3_5FeedForwardArchitecture};
use super::quantizations::optiq::OptiQQuantizationProfile;

impl Qwen3_5Config {
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
    /// `mtplx_mtp_quantization` global default for MTP modules, then the model-wide
    /// default profile if the module is not found in any override map.
    #[must_use]
    pub fn quantization_profile_for_module(&self, module_name: &str) -> OptiQQuantizationProfile {
        self.mtp_quantized_module_profiles
            .get(module_name)
            .copied()
            .or_else(|| self.quantized_module_profiles.get(module_name).copied())
            .or_else(|| {
                // MTP modules without a per-module override get the global
                // mtplx_mtp_quantization fallback when declared.
                if module_name.starts_with("language_model.mtp.") {
                    self.mtxplx_mtp_quantization_fallback
                        .map(|fallback| OptiQQuantizationProfile {
                            bits: fallback.bits,
                            group_size: fallback.group_size,
                        })
                } else {
                    None
                }
            })
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
    /// The shard_tensor_names parameter should contain all tensor names from the
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
    /// Falls back to rope_parameters.partial_rotary_factor when the top-level
    /// text_config.partial_rotary_factor is absent, matching the Qwen3.5
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

    /// Returns the MTP sidecar file path declared in `mlx_lm_extra_tensors.mtp_file`,
    /// or `None` when MTP weights are embedded in the shard index or absent.
    #[must_use]
    pub fn sidecar_mtp_file(&self) -> Option<&str> {
        self.sidecar_mtp_file.as_deref()
    }
}

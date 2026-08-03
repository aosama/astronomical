use std::collections::BTreeMap;

use serde::Deserialize;

use crate::qwen3_5::{
    Qwen3_5FeedForwardArchitecture,
    configuration::config_validation::{Qwen3_5ConfigError, validate_exact_value},
};

const EXPECTED_QUANTIZATION_MODE: &str = "affine";

/// Returns whether MLX supports the affine quantization bit width.
pub(crate) const fn is_mlx_affine_quantization_bit_width_supported(bit_width: u32) -> bool {
    matches!(bit_width, 2 | 3 | 4 | 5 | 6 | 8)
}

/// Returns whether MLX supports the affine quantization group size.
pub(crate) const fn is_mlx_affine_quantization_group_size_supported(group_size: u32) -> bool {
    matches!(group_size, 32 | 64 | 128)
}

/// Per-module quantization profile carrying both bit width and group size.
///
/// Each quantized module in a Qwen3.5 model can have its own bit width
/// and group size. The OptiQ mixed-precision format uses sparse overrides:
/// modules not listed in the config use the default `(bits, group_size)`.
///
/// A profile with `bits = 0` indicates an unquantized module stored as
/// bfloat16 (no scales/biases tensors on disk). This is used for the
/// MoE router gate in some models (e.g., oQ6e where the gate is unquantized).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OptiQQuantizationProfile {
    pub bits: u32,
    pub group_size: u32,
}

impl OptiQQuantizationProfile {
    /// Returns true when the module is stored as plain bfloat16 (no quantization).
    ///
    /// Unquantized modules have `bits = 0` and no `.scales`/`.biases` tensors
    /// on disk. The weight is loaded directly as a bfloat16 tensor.
    #[must_use]
    pub const fn is_unquantized(&self) -> bool {
        self.bits == 0
    }

    /// Creates an unquantized profile (bits=0, group_size=0) indicating bfloat16 storage.
    #[must_use]
    pub const fn unquantized() -> Self {
        Self {
            bits: 0,
            group_size: 0,
        }
    }
}

/// Text config fields needed by the quantization validator.
pub trait QuantizationConfigSource {
    fn layer_count(&self) -> u32;
    fn decoder_layer_is_full_attention(&self, decoder_layer_index: usize) -> bool;
}

/// The two identical quantization documents embedded in the Qwen3.5 config.
///
/// Supports both the original OptiQ-4bit format (explicit overrides for every
/// module) and the newer mixed-precision format (sparse overrides where modules
/// not listed use the default `bits` and `group_size`).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct OptiQQuantizationConfig {
    group_size: u32,
    bits: u32,
    mode: String,
    #[serde(flatten)]
    module_overrides: BTreeMap<String, OptiQQuantizationOverride>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct OptiQQuantizationOverride {
    group_size: u32,
    bits: u32,
}

impl OptiQQuantizationConfig {
    /// Returns the default quantization group size declared by the artifact.
    ///
    /// This is the group size used for modules not listed in the overrides.
    /// Individual overrides may specify a different group size.
    pub const fn default_group_size(&self) -> u32 {
        self.group_size
    }

    /// Returns the default quantization bit width declared by the artifact.
    ///
    /// This is the bit width used for modules not listed in the overrides.
    /// Individual overrides may specify a different bit width.
    pub const fn default_bits(&self) -> u32 {
        self.bits
    }

    /// Validates the quantization config and returns a fully resolved map of
    /// module name to quantization profile (bits + group_size).
    ///
    /// For modules not listed in overrides, the default `(bits, group_size)`
    /// is used. Supports OptiQ-4bit (explicit all-module overrides with bits=4),
    /// and mixed-precision formats (sparse overrides with default bits=6).
    pub fn validate(
        &self,
        config_source: &impl QuantizationConfigSource,
        feed_forward_architecture: Qwen3_5FeedForwardArchitecture,
    ) -> Result<BTreeMap<String, OptiQQuantizationProfile>, Qwen3_5ConfigError> {
        validate_exact_value("quantization.mode", &self.mode, EXPECTED_QUANTIZATION_MODE)?;

        if !is_mlx_affine_quantization_bit_width_supported(self.bits) {
            return Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
                description: format!(
                    "quantization.bits must be supported by MLX affine quantization, got {}",
                    self.bits
                ),
            });
        }

        if !is_mlx_affine_quantization_group_size_supported(self.group_size) {
            return Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
                description: format!(
                    "quantization.group_size must be supported by MLX affine quantization, got {}",
                    self.group_size
                ),
            });
        }

        // Validate each override against the same affine parameters MLX accepts.
        for (module_name, quantization_override) in &self.module_overrides {
            if !is_mlx_affine_quantization_bit_width_supported(quantization_override.bits) {
                return Err(Qwen3_5ConfigError::UnsupportedQuantizationOverrideBits {
                    module_name: module_name.clone(),
                    actual_value: quantization_override.bits,
                });
            }
            if !is_mlx_affine_quantization_group_size_supported(quantization_override.group_size) {
                return Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
                    description: format!(
                        "quantization module override '{module_name}' group_size is not supported by MLX affine quantization: {}",
                        quantization_override.group_size
                    ),
                });
            }
        }

        // Build the fully resolved module profile map.
        // For OptiQ-4bit (bits=4): all modules are explicit overrides.
        // For mixed-precision (bits=6): overrides specify modules that differ from default.
        let default_profile = OptiQQuantizationProfile {
            bits: self.bits,
            group_size: self.group_size,
        };
        let all_module_names =
            expected_quantized_module_names(config_source, feed_forward_architecture);
        let mut quantized_module_profiles = BTreeMap::new();
        for module_name in all_module_names {
            let profile = match self.module_overrides.get(&module_name) {
                Some(quantization_override) => OptiQQuantizationProfile {
                    bits: quantization_override.bits,
                    group_size: quantization_override.group_size,
                },
                None => default_profile,
            };
            quantized_module_profiles.insert(module_name, profile);
        }

        Ok(quantized_module_profiles)
    }

    /// Returns the artifact-declared quantization profiles for optional MTP modules.
    #[must_use]
    pub fn mtp_quantized_module_profiles(&self) -> BTreeMap<String, OptiQQuantizationProfile> {
        self.module_overrides
            .iter()
            .filter(|(module_name, _)| module_name.starts_with("language_model.mtp."))
            .map(|(module_name, quantization_override)| {
                (
                    module_name.clone(),
                    OptiQQuantizationProfile {
                        bits: quantization_override.bits,
                        group_size: quantization_override.group_size,
                    },
                )
            })
            .collect()
    }
}

fn expected_quantized_module_names(
    config_source: &impl QuantizationConfigSource,
    feed_forward_architecture: Qwen3_5FeedForwardArchitecture,
) -> Vec<String> {
    let layer_count = config_source.layer_count() as usize;
    let mut quantized_module_names = Vec::new();
    for decoder_layer_index in 0..layer_count {
        let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}");
        if config_source.decoder_layer_is_full_attention(decoder_layer_index) {
            for projection_name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                quantized_module_names.push(format!("{layer_prefix}.self_attn.{projection_name}"));
            }
        } else {
            for projection_name in [
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            ] {
                quantized_module_names
                    .push(format!("{layer_prefix}.linear_attn.{projection_name}"));
            }
        }
        let mlp_module_suffixes: &[&str] = match feed_forward_architecture {
            Qwen3_5FeedForwardArchitecture::Dense => {
                &["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
            }
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts => &[
                "mlp.gate",
                "mlp.switch_mlp.gate_proj",
                "mlp.switch_mlp.up_proj",
                "mlp.switch_mlp.down_proj",
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.down_proj",
                "mlp.shared_expert_gate",
            ],
        };
        for module_suffix in mlp_module_suffixes {
            quantized_module_names.push(format!("{layer_prefix}.{module_suffix}"));
        }
    }
    quantized_module_names.push("language_model.model.embed_tokens".to_owned());
    quantized_module_names.push("language_model.lm_head".to_owned());
    quantized_module_names
}

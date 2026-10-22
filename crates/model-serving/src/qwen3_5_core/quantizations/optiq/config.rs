use std::collections::BTreeMap;

use serde::Deserialize;

use astronomical_jangtq2_qwen35::configuration::Jangtq2WeightOverrideEntry;
use crate::qwen3_5_core::configuration::Qwen3_5FeedForwardArchitecture;
use crate::qwen3_5_core::configuration::config_validation::{
    Qwen3_5ConfigError, validate_exact_value,
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

/// Physical codec one module's weights are stored with.
///
/// The affine codec is the default and covers everything the stock loader
/// understands today. The codebook codec carries only the descriptor the
/// jangtq2 family needs; every rule about that format lives in the family
/// folder, and consumers branch on this enum exactly once, at the point
/// where storage becomes tensor shapes or decode work — via
/// [`OptiQQuantizationProfile::is_jangtq2_codebook`] plus delegation to the
/// family, never by matching the variant inside consumer logic.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ModuleWeightCodec {
    /// Standard MLX affine quantization: packed `uint32` weights plus
    /// per-group scales and biases.
    #[default]
    MlxAffineQuantized,
    /// JANGTQ v2 codebook storage: expert-stacked packed codes with one
    /// `float16` scale per expert row and an activation-side rotation.
    Jangtq2CodebookQuantized {
        /// The rotation the decode path must apply to activations.
        rotation: astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
    },
}

/// Per-module quantization profile carrying both bit width and group size.
///
/// Each quantized module in a Qwen3.5 model can have its own bit width
/// and group size. The OptiQ mixed-precision format uses sparse overrides:
/// modules not listed in the config use the default `(bits, group_size)`.
///
/// A profile with `bits = 0` indicates an unquantized module stored as
/// bfloat16 (no scales/biases tensors on disk), used for native
/// floating-point modules inside an otherwise affine-quantized model.
/// A profile with the codebook codec carries the jangtq2 bit width and
/// rotation with `group_size = 0` — codebook storage has no group concept.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OptiQQuantizationProfile {
    pub bits: u32,
    pub group_size: u32,
    pub weight_codec: ModuleWeightCodec,
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

    /// Returns true when the module is stored in the jangtq2 codebook format.
    #[must_use]
    pub const fn is_jangtq2_codebook(&self) -> bool {
        matches!(
            self.weight_codec,
            ModuleWeightCodec::Jangtq2CodebookQuantized { .. }
        )
    }

    /// Creates an affine profile, the codec for every storage the stock
    /// affine loader understands (including `bits = 0` native bfloat16).
    ///
    /// Building profiles exclusively through the constructors keeps the
    /// codec/group-size invariant (codebook profiles always carry
    /// `group_size = 0`) in one place instead of four scattered literals.
    #[must_use]
    pub const fn affine(bits: u32, group_size: u32) -> Self {
        Self {
            bits,
            group_size,
            weight_codec: ModuleWeightCodec::MlxAffineQuantized,
        }
    }

    /// Creates a jangtq2 codebook profile with the family's no-group
    /// sentinel (`group_size = 0`).
    #[must_use]
    pub const fn jangtq2_codebook(
        bits: u32,
        rotation: astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
    ) -> Self {
        Self {
            bits,
            group_size: 0,
            weight_codec: ModuleWeightCodec::Jangtq2CodebookQuantized { rotation },
        }
    }

    /// Creates an unquantized profile (bits=0, group_size=0) indicating bfloat16 storage.
    #[must_use]
    pub const fn unquantized() -> Self {
        Self::affine(0, 0)
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
///
/// Per-module override entries parse through the jangtq2 family's entry
/// type, which owns the three wire shapes that coexist in quantized bundles:
/// MLX affine objects, jangtq2 codebook objects, and MLX loader boolean
/// flags. Affine entries resolve into ordinary affine profiles exactly as
/// they always have; codebook entries resolve into profiles whose codec
/// names the jangtq2 family, which owns every format rule downstream
/// (tensor inventory shapes and, once its serving integration lands, the
/// per-expert decode).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct OptiQQuantizationConfig {
    group_size: u32,
    bits: u32,
    mode: String,
    #[serde(flatten)]
    module_overrides: BTreeMap<String, Jangtq2WeightOverrideEntry>,
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

        // Validate each override. Affine entries are checked against the
        // same MLX affine grid as before; codebook and loader-flag shapes
        // delegate their own rules to the jangtq2 family, which owns that
        // format end to end.
        for (module_name, weight_override) in &self.module_overrides {
            let Jangtq2WeightOverrideEntry::Affine { bits, group_size } = weight_override else {
                weight_override.validate().map_err(|jangtq2_error| {
                    Qwen3_5ConfigError::InvalidConfigValueDynamic {
                        description: format!(
                            "quantization module override `{module_name}`: {jangtq2_error}"
                        ),
                    }
                })?;
                continue;
            };
            if !is_mlx_affine_quantization_bit_width_supported(*bits) {
                return Err(Qwen3_5ConfigError::UnsupportedQuantizationOverrideBits {
                    module_name: module_name.clone(),
                    actual_value: *bits,
                });
            }
            if !is_mlx_affine_quantization_group_size_supported(*group_size) {
                return Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
                    description: format!(
                        "quantization module override '{module_name}' group_size is not supported by MLX affine quantization: {}",
                        group_size
                    ),
                });
            }
        }

        // Build the fully resolved module profile map.
        // For OptiQ-4bit (bits=4): all modules are explicit overrides.
        // For mixed-precision (bits=6): overrides specify modules that differ from default.
        let default_profile = OptiQQuantizationProfile::affine(self.bits, self.group_size);
        let mut all_module_names =
            expected_quantized_module_names(config_source, feed_forward_architecture);
        // The OptiQ map may also carry explicitly quantized vision modules.
        // Their geometry belongs to the vision configuration, but their codec
        // descriptors still need to survive text-config parsing for the vision
        // tensor contract and weight loader.
        all_module_names.extend(self.module_overrides.keys().cloned());
        all_module_names.sort_unstable();
        all_module_names.dedup();
        let mut quantized_module_profiles = BTreeMap::new();
        for module_name in all_module_names {
            let module_profile = match self.module_overrides.get(&module_name) {
                Some(Jangtq2WeightOverrideEntry::Affine { bits, group_size }) => {
                    OptiQQuantizationProfile::affine(*bits, *group_size)
                }
                // A `false` flag marks native floating-point storage — the
                // same bits=0 convention the shard-index resolver produces.
                Some(Jangtq2WeightOverrideEntry::LoaderFlag(false)) => {
                    OptiQQuantizationProfile::unquantized()
                }
                // A `true` flag defers to the document defaults, exactly
                // like an unlisted module.
                Some(Jangtq2WeightOverrideEntry::LoaderFlag(true)) | None => default_profile,
                // Codebook storage carries its bit width and rotation; the
                // format's tensor shapes and decode rules belong to the
                // jangtq2 family, reached through the profile's codec.
                Some(Jangtq2WeightOverrideEntry::Codebook { bits, rotation, .. }) => {
                    OptiQQuantizationProfile::jangtq2_codebook(*bits, *rotation)
                }
            };
            quantized_module_profiles.insert(module_name, module_profile);
        }

        Ok(quantized_module_profiles)
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

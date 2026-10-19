//! Wire-schema types for a folded ternary Prism pack (`config.json`).
//!
//! The family owns its document schema outright rather than reaching into the
//! standard qwen3_5 configuration: the folded pack declares a different
//! top-level shape (schema version, packed-module records, component flags),
//! and family-owned parsing keeps the standard family free of folded concepts.

use thiserror::Error;

use super::hadamard;

#[cfg(test)]
mod tests;

mod document;

use document::{
    FoldedPackDocument, FoldedTextConfigDocument, PackedModuleDocument, RopeParametersDocument,
};

/// The only accepted folded-pack schema version.
/// The shipped pack declares schema version 2 (the vision-carrying folded
/// document generation); the module records, sign manifest, and text config
/// fields this family validates were modeled from that shipped document.
pub(crate) const SUPPORTED_FOLDED_SCHEMA_VERSION: u32 = 2;
/// The only accepted discovery model type for this family.
pub(crate) const FOLDED_MODEL_TYPE: &str = "prism_hadamard_qwen35";
/// The only accepted base family. The folded pack is a dense Qwen3.5-derived
/// hybrid; any other base has no parity reference here.
pub(crate) const FOLDED_BASE_MODEL_TYPE: &str = "qwen3_5";
/// The only accepted Hadamard manifest location. The manifest is validated
/// independently; this field only pins where the pack declares it.
pub(crate) const FOLDED_HADAMARD_CONFIG_FILE_NAME: &str = "hadamard.json";
/// The only accepted tensor namespace. It fixes the mapping from packed-module
/// record paths to safetensors tensor names.
pub(crate) const FOLDED_TENSOR_NAMESPACE: &str = "mlx-vlm-qwen3_5";
/// The safetensors prefix every language-side packed-module tensor carries.
pub(crate) const LANGUAGE_MODEL_TENSOR_PREFIX: &str = "language_model.";
/// The only accepted GDN activation layout.
pub(crate) const FOLDED_GDN_ACTIVATION_LAYOUT: &str = "grouped";
/// The only accepted activation dtype for packed modules.
pub(crate) const FOLDED_MODULE_DTYPE: &str = "float16";
/// The only accepted quantization mode for ternary packs.
pub(crate) const FOLDED_QUANTIZATION_MODE: &str = "affine";
/// The only accepted hidden activation, matching the family's kernel contract.
pub(crate) const FOLDED_HIDDEN_ACTIVATION: &str = "silu";
/// The only accepted GDN output gate, matching the family's kernel contract.
pub(crate) const FOLDED_OUTPUT_GATE_TYPE: &str = "swish";
/// The only accepted recurrent-state dtype, matching the GDN decode kernels.
pub(crate) const FOLDED_SSM_DTYPE: &str = "float32";
/// The quantization bit width of the shipped ternary representation.
pub(crate) const FOLDED_QUANTIZATION_BITS: u32 = 2;
/// The quantization group size of the shipped ternary representation.
pub(crate) const FOLDED_QUANTIZATION_GROUP_SIZE: u32 = 128;
/// Bound on the packed-module record count. The shipped 27B pack declares 402.
pub(crate) const MAXIMUM_PACKED_MODULE_COUNT: usize = 65_536;
/// Bound on any packed-module record path.
const MAXIMUM_MODULE_PATH_BYTES: usize = 512;
/// Bound on the declared runtime requirement string. The field is metadata
/// for generic loaders; this family implements the contract natively.
const MAXIMUM_REQUIRES_RUNTIME_BYTES: usize = 512;

/// The validated configuration of one folded ternary pack.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FoldedPackConfig {
    text: FoldedTextDimensions,
    packed_modules: Vec<PackedModule>,
    vision_declared: bool,
}

impl FoldedPackConfig {
    /// Parses and strictly validates a folded pack `config.json`.
    pub fn from_json_bytes(config_bytes: &[u8]) -> Result<Self, FoldedPackConfigError> {
        let document = serde_json::from_slice::<FoldedPackDocument>(config_bytes)
            .map_err(FoldedPackConfigError::DeserializeConfig)?;
        Self::from_validated_document(document)
    }

    fn from_validated_document(
        document: FoldedPackDocument,
    ) -> Result<Self, FoldedPackConfigError> {
        if document.schema_version != SUPPORTED_FOLDED_SCHEMA_VERSION {
            return Err(FoldedPackConfigError::UnsupportedSchemaVersion {
                actual_schema_version: document.schema_version,
            });
        }
        if document.model_type != FOLDED_MODEL_TYPE {
            return Err(FoldedPackConfigError::UnsupportedModelType {
                actual_model_type: document.model_type,
            });
        }
        if document.base_model_type != FOLDED_BASE_MODEL_TYPE {
            return Err(FoldedPackConfigError::UnsupportedBaseModelType {
                actual_base_model_type: document.base_model_type,
            });
        }
        if document.hadamard_config != FOLDED_HADAMARD_CONFIG_FILE_NAME {
            return Err(FoldedPackConfigError::UnsupportedHadamardConfigReference {
                actual_hadamard_config: document.hadamard_config,
            });
        }
        if document.tensor_namespace != FOLDED_TENSOR_NAMESPACE {
            return Err(FoldedPackConfigError::UnsupportedTensorNamespace {
                actual_tensor_namespace: document.tensor_namespace,
            });
        }
        if document.gdn_activation_layout != FOLDED_GDN_ACTIVATION_LAYOUT {
            return Err(FoldedPackConfigError::UnsupportedGdnLayout {
                actual_gdn_activation_layout: document.gdn_activation_layout,
            });
        }
        // Absence is legitimate (text-only exporter variants omit the
        // marker); only an absurdly long value is refused, because it rides
        // inside every load error report.
        if document.requires_runtime.len() > MAXIMUM_REQUIRES_RUNTIME_BYTES {
            return Err(FoldedPackConfigError::InvalidRequiresRuntime);
        }
        if document.components.mtp {
            return Err(FoldedPackConfigError::MultiTokenPredictionUnsupported);
        }
        if !document.components.text {
            return Err(FoldedPackConfigError::TextComponentMissing);
        }
        if document.tie_word_embeddings {
            return Err(FoldedPackConfigError::TiedEmbeddingsUnsupported);
        }
        if document.quantization.bits != FOLDED_QUANTIZATION_BITS
            || document.quantization.group_size != FOLDED_QUANTIZATION_GROUP_SIZE
            || document.quantization.mode != FOLDED_QUANTIZATION_MODE
        {
            return Err(FoldedPackConfigError::UnsupportedQuantization {
                actual_bits: document.quantization.bits,
                actual_group_size: document.quantization.group_size,
                actual_mode: document.quantization.mode,
            });
        }
        let text = FoldedTextDimensions::from_validated_document(document.text_config)?;
        let packed_modules = validated_packed_modules(document.modules)?;
        if document.components.vision {
            if document.vision_config.is_none() {
                return Err(FoldedPackConfigError::VisionConfigMissing);
            }
            if document.image_token_id.is_none()
                || document.video_token_id.is_none()
                || document.vision_start_token_id.is_none()
                || document.vision_end_token_id.is_none()
            {
                return Err(FoldedPackConfigError::VisionTokenIdsMissing);
            }
        }
        Ok(Self {
            text,
            packed_modules,
            vision_declared: document.components.vision,
        })
    }

    /// The validated language-model dimensions and schedule.
    pub const fn text(&self) -> &FoldedTextDimensions {
        &self.text
    }

    /// The validated packed-module records in declared order.
    pub fn packed_modules(&self) -> &[PackedModule] {
        &self.packed_modules
    }

    /// Whether the pack declares a vision component. The tower is validated
    /// with the pack inventory; image serving lands in a later phase.
    pub const fn vision_declared(&self) -> bool {
        self.vision_declared
    }
}

/// One packed-module record: a projection whose weights live in the rotated
/// ternary basis, with the Hadamard block the runtime must honor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedModule {
    /// Module path in pack notation (`lm_head`, `model.embed_tokens`,
    /// `layers.0.mlp.down_proj`), without the language tensor prefix.
    pub(crate) path: String,
    /// The Hadamard block size declared for this module.
    pub(crate) block: u32,
    /// Whether the module is an embedding lookup (inverse transform side).
    pub(crate) embedding: bool,
}

impl PackedModule {
    /// The full safetensors tensor stem for this module.
    pub fn tensor_stem(&self) -> String {
        format!("{LANGUAGE_MODEL_TENSOR_PREFIX}{}", self.path)
    }

    pub const fn block(&self) -> u32 {
        self.block
    }

    pub const fn is_embedding(&self) -> bool {
        self.embedding
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Validated language-model dimensions and decoder schedule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FoldedTextDimensions {
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) num_hidden_layers: u32,
    pub(crate) num_attention_heads: u32,
    pub(crate) num_key_value_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) vocab_size: u32,
    pub(crate) full_attention_interval: u32,
    pub(crate) linear_num_value_heads: u32,
    pub(crate) linear_num_key_heads: u32,
    pub(crate) linear_key_head_dim: u32,
    pub(crate) linear_value_head_dim: u32,
    pub(crate) linear_conv_kernel_dim: u32,
    pub(crate) rms_norm_epsilon_bits: u32,
    pub(crate) rope_theta_bits: u32,
    pub(crate) partial_rotary_factor_bits: u32,
    pub(crate) mrope_interleaved: bool,
    pub(crate) mrope_section: Vec<u32>,
    pub(crate) layer_types: Vec<String>,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) max_position_embeddings: u32,
}

impl FoldedTextDimensions {
    fn from_validated_document(
        text: FoldedTextConfigDocument,
    ) -> Result<Self, FoldedPackConfigError> {
        if text.model_type != FOLDED_BASE_TEXT_MODEL_TYPE {
            return Err(FoldedPackConfigError::UnsupportedTextModelType {
                actual_text_model_type: text.model_type,
            });
        }
        if text.hidden_act != FOLDED_HIDDEN_ACTIVATION {
            return Err(FoldedPackConfigError::UnsupportedHiddenActivation {
                actual_hidden_act: text.hidden_act,
            });
        }
        if text.output_gate_type != FOLDED_OUTPUT_GATE_TYPE {
            return Err(FoldedPackConfigError::UnsupportedOutputGateType {
                actual_output_gate_type: text.output_gate_type,
            });
        }
        if text.mamba_ssm_dtype != FOLDED_SSM_DTYPE {
            return Err(FoldedPackConfigError::UnsupportedSsmDtype {
                actual_ssm_dtype: text.mamba_ssm_dtype,
            });
        }
        if text.attention_bias {
            return Err(FoldedPackConfigError::AttentionBiasUnsupported);
        }
        if !text.attn_output_gate {
            return Err(FoldedPackConfigError::OutputGateMissing);
        }
        if text.tie_word_embeddings {
            return Err(FoldedPackConfigError::TiedEmbeddingsUnsupported);
        }
        validated_dimensionality(&text)?;
        let layer_types = validated_layer_types(&text)?;
        let rope = validated_rope(&text.rope_parameters)?;
        Ok(Self {
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            vocab_size: text.vocab_size,
            full_attention_interval: text.full_attention_interval,
            linear_num_value_heads: text.linear_num_value_heads,
            linear_num_key_heads: text.linear_num_key_heads,
            linear_key_head_dim: text.linear_key_head_dim,
            linear_value_head_dim: text.linear_value_head_dim,
            linear_conv_kernel_dim: text.linear_conv_kernel_dim,
            rms_norm_epsilon_bits: text.rms_norm_epsilon_bits,
            rope_theta_bits: rope.theta_bits,
            partial_rotary_factor_bits: rope.partial_rotary_factor_bits,
            mrope_interleaved: rope.mrope_interleaved,
            mrope_section: rope.mrope_section,
            layer_types,
            eos_token_ids: text.eos_token_id.unwrap_or_default(),
            max_position_embeddings: text.max_position_embeddings,
        })
    }
}

const FOLDED_BASE_TEXT_MODEL_TYPE: &str = "qwen3_5_text";

fn validated_dimensionality(text: &FoldedTextConfigDocument) -> Result<(), FoldedPackConfigError> {
    if text.hidden_size == 0
        || text.intermediate_size == 0
        || text.num_hidden_layers == 0
        || text.num_attention_heads == 0
        || text.head_dim == 0
        || text.vocab_size == 0
        || text.full_attention_interval == 0
        || text.linear_conv_kernel_dim == 0
        || text.max_position_embeddings == 0
    {
        return Err(FoldedPackConfigError::NonPositiveDimension);
    }
    if text.num_key_value_heads == 0 || text.num_key_value_heads > text.num_attention_heads {
        return Err(FoldedPackConfigError::InvalidKeyValueHeadCount {
            actual_key_value_heads: text.num_key_value_heads,
            actual_attention_heads: text.num_attention_heads,
        });
    }
    if text.linear_num_key_heads == 0
        || text.linear_num_value_heads == 0
        || text.linear_num_value_heads % text.linear_num_key_heads != 0
    {
        return Err(FoldedPackConfigError::InvalidLinearHeadCounts {
            actual_value_heads: text.linear_num_value_heads,
            actual_key_heads: text.linear_num_key_heads,
        });
    }
    if text.linear_key_head_dim == 0 || text.linear_value_head_dim == 0 {
        return Err(FoldedPackConfigError::NonPositiveDimension);
    }
    Ok(())
}

fn validated_layer_types(
    text: &FoldedTextConfigDocument,
) -> Result<Vec<String>, FoldedPackConfigError> {
    if text.layer_types.len() != text.num_hidden_layers as usize {
        return Err(FoldedPackConfigError::LayerTypeCountMismatch {
            declared_layers: text.num_hidden_layers,
            actual_layer_type_count: text.layer_types.len(),
        });
    }
    for layer_type in &text.layer_types {
        if layer_type != "linear_attention" && layer_type != "full_attention" {
            return Err(FoldedPackConfigError::UnsupportedLayerType {
                actual_layer_type: layer_type.clone(),
            });
        }
    }
    Ok(text.layer_types.clone())
}

struct ValidatedRope {
    theta_bits: u32,
    partial_rotary_factor_bits: u32,
    mrope_interleaved: bool,
    mrope_section: Vec<u32>,
}

fn validated_rope(rope: &RopeParametersDocument) -> Result<ValidatedRope, FoldedPackConfigError> {
    if rope.rope_type != "default" {
        return Err(FoldedPackConfigError::UnsupportedRopeType {
            actual_rope_type: rope.rope_type.clone(),
        });
    }
    if rope.mrope_section.is_empty() {
        return Err(FoldedPackConfigError::EmptyMropeSection);
    }
    Ok(ValidatedRope {
        theta_bits: rope.rope_theta_bits,
        partial_rotary_factor_bits: rope.partial_rotary_factor_bits,
        mrope_interleaved: rope.mrope_interleaved,
        mrope_section: rope.mrope_section.clone(),
    })
}

fn validated_packed_modules(
    modules: Vec<PackedModuleDocument>,
) -> Result<Vec<PackedModule>, FoldedPackConfigError> {
    if modules.is_empty() {
        return Err(FoldedPackConfigError::NoPackedModules);
    }
    if modules.len() > MAXIMUM_PACKED_MODULE_COUNT {
        return Err(FoldedPackConfigError::TooManyPackedModules {
            actual_module_count: modules.len(),
            maximum_module_count: MAXIMUM_PACKED_MODULE_COUNT,
        });
    }
    let mut validated_modules = Vec::with_capacity(modules.len());
    let mut seen_paths = std::collections::BTreeSet::new();
    for module in modules {
        if module.path.is_empty() || module.path.len() > MAXIMUM_MODULE_PATH_BYTES {
            return Err(FoldedPackConfigError::InvalidModulePath {
                actual_path: module.path,
            });
        }
        if module.dtype != FOLDED_MODULE_DTYPE {
            return Err(FoldedPackConfigError::UnsupportedModuleDtype {
                actual_dtype: module.dtype,
                module_path: module.path,
            });
        }
        if !hadamard::SUPPORTED_BLOCK_SIZES.contains(&module.block) {
            return Err(FoldedPackConfigError::UnsupportedModuleBlock {
                actual_block: module.block,
                module_path: module.path,
            });
        }
        if !seen_paths.insert(module.path.clone()) {
            return Err(FoldedPackConfigError::DuplicateModulePath {
                duplicate_path: module.path,
            });
        }
        validated_modules.push(PackedModule {
            path: module.path,
            block: module.block,
            embedding: module.embedding,
        });
    }
    Ok(validated_modules)
}

/// Typed refusal for a folded-pack configuration this family does not serve.
#[derive(Debug, Error)]
pub enum FoldedPackConfigError {
    #[error("folded pack declares unsupported schema version {actual_schema_version}")]
    UnsupportedSchemaVersion { actual_schema_version: u32 },
    #[error("folded pack declares unsupported model type {actual_model_type:?}")]
    UnsupportedModelType { actual_model_type: String },
    #[error("folded pack declares unsupported base model type {actual_base_model_type:?}")]
    UnsupportedBaseModelType { actual_base_model_type: String },
    #[error("folded pack declares unsupported text model type {actual_text_model_type:?}")]
    UnsupportedTextModelType { actual_text_model_type: String },
    #[error(
        "folded pack declares unsupported quantization (bits={actual_bits}, group_size={actual_group_size}, mode={actual_mode:?})"
    )]
    UnsupportedQuantization {
        actual_bits: u32,
        actual_group_size: u32,
        actual_mode: String,
    },
    #[error(
        "folded pack declares unsupported hadamard manifest reference {actual_hadamard_config:?}"
    )]
    UnsupportedHadamardConfigReference { actual_hadamard_config: String },
    #[error("folded pack declares unsupported tensor namespace {actual_tensor_namespace:?}")]
    UnsupportedTensorNamespace { actual_tensor_namespace: String },
    #[error(
        "folded pack declares unsupported GDN activation layout {actual_gdn_activation_layout:?}"
    )]
    UnsupportedGdnLayout {
        actual_gdn_activation_layout: String,
    },
    #[error("folded pack declares an invalid requires-runtime reference")]
    InvalidRequiresRuntime,
    #[error("folded pack declares the multi-token-prediction component, which is unsupported")]
    MultiTokenPredictionUnsupported,
    #[error("folded pack omits the text component")]
    TextComponentMissing,
    #[error("folded pack declares tied embeddings, which are unsupported")]
    TiedEmbeddingsUnsupported,
    #[error("folded pack declares unsupported hidden activation {actual_hidden_act:?}")]
    UnsupportedHiddenActivation { actual_hidden_act: String },
    #[error("folded pack declares unsupported GDN output gate type {actual_output_gate_type:?}")]
    UnsupportedOutputGateType { actual_output_gate_type: String },
    #[error("folded pack declares unsupported SSM dtype {actual_ssm_dtype:?}")]
    UnsupportedSsmDtype { actual_ssm_dtype: String },
    #[error("folded pack declares attention bias tensors, which are unsupported")]
    AttentionBiasUnsupported,
    #[error("folded pack disables the GDN output gate, which the kernel contract requires")]
    OutputGateMissing,
    #[error("folded pack declares a non-positive dimension")]
    NonPositiveDimension,
    #[error(
        "folded pack declares invalid head counts (kv={actual_key_value_heads}, attention={actual_attention_heads})"
    )]
    InvalidKeyValueHeadCount {
        actual_key_value_heads: u32,
        actual_attention_heads: u32,
    },
    #[error(
        "folded pack declares invalid linear head counts (value={actual_value_heads}, key={actual_key_heads})"
    )]
    InvalidLinearHeadCounts {
        actual_value_heads: u32,
        actual_key_heads: u32,
    },
    #[error(
        "folded pack layer schedule covers {actual_layer_type_count} layers but the config declares {declared_layers}"
    )]
    LayerTypeCountMismatch {
        declared_layers: u32,
        actual_layer_type_count: usize,
    },
    #[error("folded pack declares unsupported layer type {actual_layer_type:?}")]
    UnsupportedLayerType { actual_layer_type: String },
    #[error("folded pack declares unsupported rope type {actual_rope_type:?}")]
    UnsupportedRopeType { actual_rope_type: String },
    #[error("folded pack declares an empty mrope section")]
    EmptyMropeSection,
    #[error("folded pack declares no packed modules")]
    NoPackedModules,
    #[error(
        "folded pack declares {actual_module_count} packed modules above the {maximum_module_count} limit"
    )]
    TooManyPackedModules {
        actual_module_count: usize,
        maximum_module_count: usize,
    },
    #[error("folded pack declares an invalid packed-module path {actual_path:?}")]
    InvalidModulePath { actual_path: String },
    #[error("folded pack module {module_path:?} declares unsupported dtype {actual_dtype:?}")]
    UnsupportedModuleDtype {
        actual_dtype: String,
        module_path: String,
    },
    #[error(
        "folded pack module {module_path:?} declares unsupported hadamard block {actual_block}"
    )]
    UnsupportedModuleBlock {
        actual_block: u32,
        module_path: String,
    },
    #[error("folded pack declares duplicate module path {duplicate_path:?}")]
    DuplicateModulePath { duplicate_path: String },
    #[error("folded pack declares the vision component without a vision config")]
    VisionConfigMissing,
    #[error("folded pack declares the vision component without the vision token ids")]
    VisionTokenIdsMissing,
    #[error("folded pack config is not valid JSON: {0}")]
    DeserializeConfig(#[source] serde_json::Error),
}

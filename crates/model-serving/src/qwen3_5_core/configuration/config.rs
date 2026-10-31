use std::collections::BTreeMap;

use super::config_document::{Qwen3_5ConfigDocument, Qwen3_5TextConfig};
use super::config_validation;
use super::config_validation::{QWEN_CHAT_EOS_TOKEN_ID, Qwen3_5ConfigError};
use super::quantizations::optiq::OptiQQuantizationProfile;
use astronomical_jangtq2_qwen35::configuration::Jangtq2RmsNormConvention;

#[cfg(test)]
mod tests;

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
    pub(super) eos_token_ids: Vec<u32>,
    pub(super) has_tied_embeddings: bool,
    pub(super) quantized_module_profiles: BTreeMap<String, OptiQQuantizationProfile>,
    pub(super) model_weight_storage: ModelWeightStorage,
    pub(super) default_quantization_bits: u32,
    pub(super) default_quantization_group_size: u32,
    pub(super) jangtq2_rms_norm_convention: Option<Jangtq2RmsNormConvention>,
    pub(super) text_config: Qwen3_5TextConfig,
    pub(super) activation_dtype: String,
    pub(super) feed_forward_architecture: Qwen3_5FeedForwardArchitecture,
}

impl Qwen3_5Config {
    /// Parses the config bytes retained by validated artifact ownership.
    pub fn from_json_bytes(config_bytes: &[u8]) -> Result<Self, Qwen3_5ConfigError> {
        let config_document = serde_json::from_slice::<Qwen3_5ConfigDocument>(config_bytes)
            .map_err(Qwen3_5ConfigError::DeserializeConfig)?;
        let jangtq2_rms_norm_convention = config_document
            .norm_convention
            .as_deref()
            .map(Jangtq2RmsNormConvention::from_wire_name)
            .transpose()
            .map_err(
                |norm_convention_error| Qwen3_5ConfigError::InvalidConfigValueDynamic {
                    description: norm_convention_error.to_string(),
                },
            )?;
        let feed_forward_architecture =
            feed_forward_architecture_for_model_type(&config_document.model_type)?;
        let expected_architecture = match feed_forward_architecture {
            Qwen3_5FeedForwardArchitecture::Dense => EXPECTED_DENSE_ARCHITECTURE,
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts => EXPECTED_MOE_ARCHITECTURE,
        };
        config_validation::validate_exact_value(
            "architectures",
            &config_document.architectures.join(","),
            expected_architecture,
        )?;
        // Resolve activation dtype: prefer top-level, fall back to text_config.dtype.
        // Qwen3.6 places dtype inside text_config instead of at the top level.
        let activation_dtype = config_document
            .activation_dtype
            .clone()
            .or(config_document.text_config.text_config_dtype.clone())
            .ok_or(Qwen3_5ConfigError::MissingActivationDtype)?;
        config_validation::validate_exact_value("dtype", &activation_dtype, EXPECTED_TORCH_DTYPE)?;
        let eos_token_ids = resolved_eos_token_ids(&config_document)?;
        config_document
            .text_config
            .validate(feed_forward_architecture)?;
        let ResolvedQuantization {
            quantized_module_profiles,
            model_weight_storage,
            default_quantization_bits,
            default_quantization_group_size,
        } = resolved_quantization(&config_document, feed_forward_architecture)?;
        Ok(Self {
            eos_token_ids,
            has_tied_embeddings: config_document.tie_word_embeddings,
            quantized_module_profiles,
            model_weight_storage,
            default_quantization_bits,
            default_quantization_group_size,
            jangtq2_rms_norm_convention,
            text_config: config_document.text_config,
            activation_dtype,
            feed_forward_architecture,
        })
    }
}

/// Maps the document's `model_type` to the feed-forward architecture this
/// engine implements. Only the two expected Qwen3.5 families parse; anything
/// else is a different model generation wearing the same file name.
fn feed_forward_architecture_for_model_type(
    model_type: &str,
) -> Result<Qwen3_5FeedForwardArchitecture, Qwen3_5ConfigError> {
    match model_type {
        EXPECTED_DENSE_MODEL_TYPE => Ok(Qwen3_5FeedForwardArchitecture::Dense),
        EXPECTED_MOE_MODEL_TYPE => Ok(Qwen3_5FeedForwardArchitecture::MixtureOfExperts),
        _ => Err(Qwen3_5ConfigError::InvalidConfigValueDynamic {
            description: format!(
                "model_type must be '{EXPECTED_DENSE_MODEL_TYPE}' or '{EXPECTED_MOE_MODEL_TYPE}'"
            ),
        }),
    }
}

/// Resolves the effective end-of-sequence token list from the document:
/// 1. Use top-level eos_token_id if present
/// 2. Otherwise fall back to text_config.eos_token_id
/// 3. Append the Qwen chat EOS token (248046) if not already in the list
/// 4. Retain every declared stop token
/// 5. When exactly one stop token remains, adopt the pad token as a second
///    stop so padding never decodes past a sequence end
///
/// An empty resolution (neither field present) fails: generation needs at
/// least one terminator.
fn resolved_eos_token_ids(
    config_document: &Qwen3_5ConfigDocument,
) -> Result<Vec<u32>, Qwen3_5ConfigError> {
    let mut resolved_eos_token_ids = match &config_document.eos_token_id {
        Some(token_ids) => token_ids.clone(),
        None => config_document
            .text_config
            .text_config_eos_token_id
            .clone()
            .ok_or(Qwen3_5ConfigError::MissingEosTokenId)?,
    };
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
    Ok(resolved_eos_token_ids)
}

/// The quantization facts a parsed config carries: per-module profiles, the
/// whole-model weight storage encoding, and the default bits and group size
/// quantized modules fall back to.
struct ResolvedQuantization {
    quantized_module_profiles: BTreeMap<String, OptiQQuantizationProfile>,
    model_weight_storage: ModelWeightStorage,
    default_quantization_bits: u32,
    default_quantization_group_size: u32,
}

/// Resolves quantization from the document's two redundant quantization
/// blocks: absent both means native bfloat16 storage; present, they must
/// agree (a mismatch means the artifact was assembled from two different
/// quantization runs), and whichever copy exists is validated against the
/// text configuration before its profiles and defaults are trusted.
fn resolved_quantization(
    config_document: &Qwen3_5ConfigDocument,
    feed_forward_architecture: Qwen3_5FeedForwardArchitecture,
) -> Result<ResolvedQuantization, Qwen3_5ConfigError> {
    let (
        quantized_module_profiles,
        model_weight_storage,
        default_quantization_bits,
        default_quantization_group_size,
    ) = match (
        config_document.quantization.as_ref(),
        config_document.quantization_config.as_ref(),
    ) {
        (None, None) => (BTreeMap::new(), ModelWeightStorage::NativeBfloat16, 0, 0),
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
            let quantized_module_profiles =
                quantization.validate(&config_document.text_config, feed_forward_architecture)?;
            (
                quantized_module_profiles,
                ModelWeightStorage::AffineQuantized,
                quantization.default_bits(),
                quantization.default_group_size(),
            )
        }
    };
    Ok(ResolvedQuantization {
        quantized_module_profiles,
        model_weight_storage,
        default_quantization_bits,
        default_quantization_group_size,
    })
}

//! The multi-token-prediction head a dense Qwen 3.5 pack ships beside its
//! model, in a folder of its own (`mtp/`, as Ternary Bonsai 2's 9B does).
//!
//! The head is one Qwen 3.5 full-attention decoder layer between a
//! projection that fuses a token's embedding with the model's final hidden
//! state and a final norm; it borrows the model's embedding table and output
//! head. Its weights are quantized apart from the model's, so its
//! configuration names its own quantization. This module reads that
//! configuration, refuses one that does not fit the model it sits beside,
//! and lists every tensor the head must ship.

use serde::Deserialize;

use crate::{TensorDtype, TensorProfile};

use super::Qwen3_5Config;
use super::tensor_spec::{
    append_qwen3_5_full_attention_tensor_profiles, append_qwen3_5_quantized_affine_tensor_profiles,
    qwen3_5_tensor_profile,
};

/// The folder beside the model's own files that holds the head.
pub const DENSE_MTP_HEAD_FOLDER_NAME: &str = "mtp";
/// The head's weights file inside that folder.
pub const DENSE_MTP_HEAD_WEIGHTS_FILE_NAME: &str = "model.safetensors";
/// The head's configuration file inside that folder.
pub const DENSE_MTP_HEAD_CONFIG_FILE_NAME: &str = "config.json";
/// A head configuration is a few kilobytes; anything near this is not one.
pub const MAXIMUM_DENSE_MTP_HEAD_CONFIG_BYTES: u64 = 1024 * 1024;
/// The `model_type` a dense Qwen 3.5 head declares. Other heads in an `mtp/`
/// folder (Bonsai 2.5's `mtp-v2-head`) declare none and are not this kind.
const DENSE_MTP_HEAD_MODEL_TYPE: &str = "qwen3_5_mtp";
/// The head's one decoder layer.
pub const DENSE_MTP_HEAD_LAYER_PREFIX: &str = "layers.0";
/// The projection that fuses the embedding with the hidden state.
pub const DENSE_MTP_HEAD_FUSION_MODULE: &str = "fc";
/// MLX affine quantization's bit widths and group sizes.
const SUPPORTED_QUANTIZATION_BITS: [u32; 6] = [2, 3, 4, 5, 6, 8];
const SUPPORTED_QUANTIZATION_GROUP_SIZES: [u32; 3] = [32, 64, 128];

/// A validated head configuration: how its weights are quantized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseMtpHeadSidecar {
    pub quantization_bits: u32,
    pub quantization_group_size: u32,
}

/// Why a head that declared itself a dense Qwen 3.5 head was refused.
#[derive(Debug, thiserror::Error)]
pub enum DenseMtpHeadSidecarError {
    #[error("the head's config.json is not valid JSON: {0}")]
    Deserialize(#[from] serde_json::Error),
    #[error("the head is quantized {bits}-bit in groups of {group_size}, which MLX cannot serve")]
    UnsupportedQuantization { bits: u32, group_size: u32 },
    #[error("the head's {field} is {head_value}, but the model's is {model_value}")]
    ModelMismatch {
        field: &'static str,
        head_value: u32,
        model_value: u32,
    },
    #[error("the head's configuration does not say how its weights are quantized")]
    QuantizationMissing,
    #[error("the head's configuration has no text_config to check against the model")]
    DimensionsMissing,
    #[error("the head brings embeddings of its own, which this engine does not read")]
    DedicatedEmbeddings,
}

#[derive(Deserialize)]
struct DenseMtpHeadDocument {
    #[serde(default)]
    model_type: Option<String>,
    quantization: Option<QuantizationDocument>,
    text_config: Option<HeadTextConfigDocument>,
}

#[derive(Deserialize)]
struct QuantizationDocument {
    bits: u32,
    group_size: u32,
}

/// The dimensions a head must share with its model. The head's
/// `mtp_num_hidden_layers` is not read: Prism's export writes 0 there while
/// shipping one layer, so the layer count comes from the tensors.
#[derive(Deserialize)]
struct HeadTextConfigDocument {
    hidden_size: u32,
    vocab_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    #[serde(default)]
    mtp_use_dedicated_embeddings: bool,
}

impl DenseMtpHeadSidecar {
    /// Reads a head configuration. `Ok(None)` when the folder holds some
    /// other kind of head, which is not this reader's to judge.
    pub fn from_config_bytes(
        config_bytes: &[u8],
        model_config: &Qwen3_5Config,
    ) -> Result<Option<Self>, DenseMtpHeadSidecarError> {
        let document = serde_json::from_slice::<DenseMtpHeadDocument>(config_bytes)?;
        if document.model_type.as_deref() != Some(DENSE_MTP_HEAD_MODEL_TYPE) {
            return Ok(None);
        }
        let quantization = document
            .quantization
            .ok_or(DenseMtpHeadSidecarError::QuantizationMissing)?;
        if !SUPPORTED_QUANTIZATION_BITS.contains(&quantization.bits)
            || !SUPPORTED_QUANTIZATION_GROUP_SIZES.contains(&quantization.group_size)
        {
            return Err(DenseMtpHeadSidecarError::UnsupportedQuantization {
                bits: quantization.bits,
                group_size: quantization.group_size,
            });
        }
        // Without its dimensions a head cannot be checked against the model
        // it sits beside, so it is refused rather than bound unchecked.
        let text_config = document
            .text_config
            .ok_or(DenseMtpHeadSidecarError::DimensionsMissing)?;
        validate_head_fits_model(&text_config, model_config)?;
        Ok(Some(Self {
            quantization_bits: quantization.bits,
            quantization_group_size: quantization.group_size,
        }))
    }

    /// The configuration the head's weights bind with: the model's, with
    /// the head's own quantization for every module.
    #[must_use]
    pub fn head_config(&self, model_config: &Qwen3_5Config) -> Qwen3_5Config {
        model_config
            .with_uniform_affine_quantization(self.quantization_bits, self.quantization_group_size)
    }
}

fn validate_head_fits_model(
    text_config: &HeadTextConfigDocument,
    model_config: &Qwen3_5Config,
) -> Result<(), DenseMtpHeadSidecarError> {
    if text_config.mtp_use_dedicated_embeddings {
        return Err(DenseMtpHeadSidecarError::DedicatedEmbeddings);
    }
    for (field, head_value, model_value) in [
        (
            "hidden_size",
            text_config.hidden_size,
            model_config.hidden_size(),
        ),
        (
            "vocab_size",
            text_config.vocab_size,
            model_config.vocabulary_size(),
        ),
        (
            "num_attention_heads",
            text_config.num_attention_heads,
            model_config.query_head_count(),
        ),
        (
            "num_key_value_heads",
            text_config.num_key_value_heads,
            model_config.key_value_head_count(),
        ),
        (
            "head_dim",
            text_config.head_dim,
            model_config.head_dimension(),
        ),
        (
            "intermediate_size",
            text_config.intermediate_size,
            model_config.dense_intermediate_size(),
        ),
    ] {
        if head_value != model_value {
            return Err(DenseMtpHeadSidecarError::ModelMismatch {
                field,
                head_value,
                model_value,
            });
        }
    }
    Ok(())
}

/// Every tensor the head ships, for its binding configuration: the fusion
/// projection over twice the hidden width, its two input norms, one
/// full-attention layer with a dense SwiGLU MLP, and the final norm.
pub fn dense_mtp_head_tensor_profiles(
    head_config: &Qwen3_5Config,
) -> Result<Vec<TensorProfile>, super::Qwen3_5ArtifactError> {
    let hidden_size = head_config.hidden_size() as usize;
    let intermediate_size = head_config.dense_intermediate_size() as usize;
    let mut profiles = Vec::new();
    append_qwen3_5_quantized_affine_tensor_profiles(
        &mut profiles,
        DENSE_MTP_HEAD_FUSION_MODULE,
        &[hidden_size],
        hidden_size * 2,
        head_config.quantization_profile_for_module(DENSE_MTP_HEAD_FUSION_MODULE),
    )?;
    append_qwen3_5_full_attention_tensor_profiles(
        &mut profiles,
        DENSE_MTP_HEAD_LAYER_PREFIX,
        hidden_size,
        head_config.query_head_count() as usize,
        head_config.key_value_head_count() as usize,
        head_config.head_dimension() as usize,
        head_config,
    )?;
    for (projection_name, output_dimension, input_dimension) in [
        ("gate_proj", intermediate_size, hidden_size),
        ("up_proj", intermediate_size, hidden_size),
        ("down_proj", hidden_size, intermediate_size),
    ] {
        let module_name = format!("{DENSE_MTP_HEAD_LAYER_PREFIX}.mlp.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            &mut profiles,
            &module_name,
            &[output_dimension],
            input_dimension,
            head_config.quantization_profile_for_module(&module_name),
        )?;
    }
    for norm_name in [
        "pre_fc_norm_embedding.weight".to_owned(),
        "pre_fc_norm_hidden.weight".to_owned(),
        "norm.weight".to_owned(),
        format!("{DENSE_MTP_HEAD_LAYER_PREFIX}.input_layernorm.weight"),
        format!("{DENSE_MTP_HEAD_LAYER_PREFIX}.post_attention_layernorm.weight"),
    ] {
        profiles.push(qwen3_5_tensor_profile(
            norm_name,
            TensorDtype::ModelFloat,
            vec![hidden_size],
        ));
    }
    Ok(profiles)
}

#[cfg(test)]
#[path = "dense_mtp_head_sidecar_tests.rs"]
mod dense_mtp_head_sidecar_tests;

//! The steps that turn a Qwen3.5 checkpoint's loaded shards into typed language
//! weights: deciding how RMSNorm weights are stored, reading every resident
//! tensor profile out of the shards into one named table, then handing the
//! table's tensors to their typed owners.

use std::collections::HashMap;

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxSafetensors};
use text_generation::execution_error::TextExecutionError;

use super::connection_binding::take_model_output_norm;
use super::{
    Qwen3_5AffineWeights, Qwen3_5Config, Qwen3_5DecoderLayerWeights, Qwen3_5DenseProjection,
    WeightBindingDtype, bind_language_tensor, bound_language_tensor_from_model_shards,
    should_apply_raw_hugging_face_norm_offsets, take_decoder_layer_weights,
    take_quantized_affine_weights,
};
use crate::TensorProfile;
use crate::qwen3_5_core::artifacts::Qwen3_5ShardIndex;
use crate::qwen4_exp::hyper_connection::HyperConnectionWeights;

#[cfg(test)]
#[path = "language_tensor_binding_tests.rs"]
mod tests;

/// Every language tensor read from the shards, by canonical name, with the
/// byte total of what was read.
pub(super) struct BoundLanguageTensors {
    pub(super) tensors: HashMap<String, MlxArray>,
    pub(super) payload_bytes: u64,
}

/// The typed owners the table's tensors are handed to.
pub(super) struct TypedLanguageWeights {
    pub(super) embedding_weights: Qwen3_5AffineWeights,
    pub(super) decoder_layer_weights: Vec<Qwen3_5DecoderLayerWeights>,
    pub(super) final_normalization_weight: MlxArray,
    pub(super) final_lane_mixer: Option<Box<HyperConnectionWeights>>,
    pub(super) language_model_head_weights: Qwen3_5DenseProjection,
}

/// Decides whether the shards carry a raw Hugging Face checkpoint. The first
/// linear-attention convolution weight settles it; a checkpoint with no
/// linear-attention layer is treated as not raw.
pub(super) fn raw_hugging_face_norm_offsets_in_shards(
    shard_index: &Qwen3_5ShardIndex,
    model_shards: &[MlxSafetensors],
    qwen3_5_config: &Qwen3_5Config,
    language_tensor_profiles: &[TensorProfile],
) -> Result<bool, TextExecutionError> {
    match first_convolution_profile_name(language_tensor_profiles) {
        Some(convolution_profile_name) => should_apply_raw_hugging_face_norm_offsets(
            shard_index,
            model_shards,
            qwen3_5_config,
            convolution_profile_name,
        ),
        None => Ok(false),
    }
}

/// Name of the first linear-attention convolution weight in the profiles.
fn first_convolution_profile_name(language_tensor_profiles: &[TensorProfile]) -> Option<&str> {
    for tensor_profile in language_tensor_profiles {
        if tensor_profile.name.ends_with(".linear_attn.conv1d.weight") {
            return Some(&tensor_profile.name);
        }
    }
    None
}

/// Reads, validates and binds every profiled tensor from the shards, refusing
/// a name bound twice and a payload total that overflows.
pub(super) fn bind_language_tensors(
    runtime: &MlxRuntime,
    qwen3_5_config: &Qwen3_5Config,
    shard_index: &Qwen3_5ShardIndex,
    model_shards: &[MlxSafetensors],
    language_tensor_profiles: &[TensorProfile],
    raw_hugging_face_norm_offsets_apply: bool,
    weight_binding_dtype: WeightBindingDtype,
) -> Result<BoundLanguageTensors, TextExecutionError> {
    let mut bound_tensors = HashMap::with_capacity(language_tensor_profiles.len());
    let mut actual_payload_bytes = 0_u64;
    for tensor_profile in language_tensor_profiles {
        let tensor = bound_language_tensor_from_model_shards(
            shard_index,
            model_shards,
            &tensor_profile.name,
        )?;
        let (tensor, tensor_payload_bytes) = bind_language_tensor(
            runtime,
            qwen3_5_config,
            raw_hugging_face_norm_offsets_apply,
            tensor_profile,
            tensor,
            weight_binding_dtype,
        )?;
        actual_payload_bytes = add_tensor_payload_bytes(
            actual_payload_bytes,
            tensor_payload_bytes,
            &tensor_profile.name,
        )?;
        insert_tensor_once(&mut bound_tensors, &tensor_profile.name, tensor)?;
    }
    Ok(BoundLanguageTensors {
        tensors: bound_tensors,
        payload_bytes: actual_payload_bytes,
    })
}

/// Adds one tensor's bytes to the running payload total, failing instead of
/// wrapping when the total would pass the `u64` range.
fn add_tensor_payload_bytes(
    running_payload_bytes: u64,
    tensor_payload_bytes: u64,
    tensor_name: &str,
) -> Result<u64, TextExecutionError> {
    match running_payload_bytes.checked_add(tensor_payload_bytes) {
        Some(total_payload_bytes) => Ok(total_payload_bytes),
        None => Err(TextExecutionError::InvalidTensor {
            tensor_name: tensor_name.to_owned(),
            description: "total tensor payload byte count overflowed",
        }),
    }
}

/// Binds a tensor under its canonical name, refusing a second binding of the
/// same name rather than replacing the first.
fn insert_tensor_once<BoundTensor>(
    bound_tensors: &mut HashMap<String, BoundTensor>,
    tensor_name: &str,
    tensor: BoundTensor,
) -> Result<(), TextExecutionError> {
    if bound_tensors
        .insert(tensor_name.to_owned(), tensor)
        .is_some()
    {
        return Err(TextExecutionError::InvalidTensor {
            tensor_name: tensor_name.to_owned(),
            description: "tensor was bound more than once",
        });
    }
    Ok(())
}

/// Moves the bound tensors into the embedding, decoder-layer, final-norm and
/// head owners, then fails if any tensor is left unassigned.
pub(super) fn take_typed_language_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<TypedLanguageWeights, TextExecutionError> {
    let embedding_weights = take_quantized_affine_weights(
        bound_tensors,
        qwen3_5_config,
        "language_model.model.embed_tokens",
    )?;
    let decoder_layer_count = qwen3_5_config.layer_count() as usize;
    let mut decoder_layer_weights = Vec::with_capacity(decoder_layer_count);
    for decoder_layer_index in 0..decoder_layer_count {
        decoder_layer_weights.push(take_decoder_layer_weights(
            bound_tensors,
            qwen3_5_config,
            decoder_layer_index,
        )?);
    }
    let (final_normalization_weight, final_lane_mixer) =
        take_model_output_norm(bound_tensors, qwen3_5_config)?;
    let language_model_head_weights = Qwen3_5DenseProjection::named(
        "language_model.lm_head",
        if qwen3_5_config.has_tied_embeddings() {
            embedding_weights.retained_reference()?
        } else {
            take_quantized_affine_weights(bound_tensors, qwen3_5_config, "language_model.lm_head")?
        },
    );
    if let Some(unassigned_tensor_name) = bound_tensors.keys().next() {
        return Err(TextExecutionError::UnassignedTensor {
            tensor_name: unassigned_tensor_name.clone(),
        });
    }
    Ok(TypedLanguageWeights {
        embedding_weights,
        decoder_layer_weights,
        final_normalization_weight,
        final_lane_mixer,
        language_model_head_weights,
    })
}

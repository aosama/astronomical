//! Binds one dense decoder layer from tensors read outside the shard maps.
//!
//! A streamed dense layer is read per forward through the bounded-range
//! reader, not bound once from the shard maps (issue #1985). Its tensors pass
//! through the same validation, quantization-bit check and RMSNorm storage
//! convention as every resident tensor, then the same typed layer assembly,
//! so a streamed layer computes exactly what a resident one does.

use std::collections::HashMap;

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::MlxArray;

use super::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use super::error::Qwen3_5ExecutionError;
use super::weights::{bind_language_tensor, take_decoder_layer_weights};
use crate::TensorProfile;
use crate::qwen3_5_core::artifacts::qwen3_5_resident_language_tensor_profiles;
use crate::qwen3_5_core::configuration::Qwen3_5Config;

/// The language tensor profiles of each decoder layer, in layer order.
///
/// Computed once at load so a streamed forward does not rebuild the whole
/// model's profile list for every layer it reads.
pub(crate) fn decoder_layer_tensor_profiles(
    qwen3_5_config: &Qwen3_5Config,
) -> Result<Vec<Vec<TensorProfile>>, Qwen3_5ExecutionError> {
    let language_tensor_profiles = qwen3_5_resident_language_tensor_profiles(qwen3_5_config)
        .map_err(|artifact_error| Qwen3_5ExecutionError::ExpertLayerPlan {
            description: artifact_error.to_string(),
        })?;
    let decoder_layer_count = qwen3_5_config.layer_count() as usize;
    let mut layer_tensor_profiles = vec![Vec::new(); decoder_layer_count];
    for tensor_profile in language_tensor_profiles {
        if let Some(layer_index) = decoder_layer_index_of(&tensor_profile.name)
            && let Some(layer_profiles) = layer_tensor_profiles.get_mut(layer_index)
        {
            layer_profiles.push(tensor_profile);
        }
    }
    Ok(layer_tensor_profiles)
}

/// The decoder layer a canonical language tensor belongs to, if any.
pub(crate) fn decoder_layer_index_of(canonical_tensor_name: &str) -> Option<usize> {
    let after_layers_prefix = canonical_tensor_name.strip_prefix("language_model.model.layers.")?;
    let (layer_index_text, _) = after_layers_prefix.split_once('.')?;
    layer_index_text.parse().ok()
}

/// Binds one decoder layer from `layer_tensors`, keyed by canonical name.
///
/// Every profiled tensor must be present and every given tensor must be
/// used, so a read that returned the wrong set fails instead of computing.
pub(crate) fn bind_decoder_layer_from_tensors(
    runtime: &MlxRuntime,
    qwen3_5_config: &Qwen3_5Config,
    layer_index: usize,
    layer_tensor_profiles: &[TensorProfile],
    raw_hugging_face_norm_offsets_apply: bool,
    mut layer_tensors: HashMap<String, MlxArray>,
) -> Result<Qwen3_5DecoderLayerWeights, Qwen3_5ExecutionError> {
    let mut bound_tensors = HashMap::with_capacity(layer_tensor_profiles.len());
    for tensor_profile in layer_tensor_profiles {
        let tensor = layer_tensors.remove(&tensor_profile.name).ok_or_else(|| {
            Qwen3_5ExecutionError::MissingTensor {
                tensor_name: tensor_profile.name.clone(),
            }
        })?;
        let (tensor, _tensor_payload_bytes) = bind_language_tensor(
            runtime,
            qwen3_5_config,
            raw_hugging_face_norm_offsets_apply,
            tensor_profile,
            tensor,
        )?;
        bound_tensors.insert(tensor_profile.name.clone(), tensor);
    }
    if let Some(unrequested_tensor_name) = layer_tensors.keys().next() {
        return Err(Qwen3_5ExecutionError::UnassignedTensor {
            tensor_name: unrequested_tensor_name.clone(),
        });
    }
    let decoder_layer_weights =
        take_decoder_layer_weights(&mut bound_tensors, qwen3_5_config, layer_index)?;
    if let Some(unassigned_tensor_name) = bound_tensors.keys().next() {
        return Err(Qwen3_5ExecutionError::UnassignedTensor {
            tensor_name: unassigned_tensor_name.clone(),
        });
    }
    Ok(decoder_layer_weights)
}

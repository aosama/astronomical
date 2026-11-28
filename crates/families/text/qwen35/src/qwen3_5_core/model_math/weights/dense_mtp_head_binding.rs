//! Binding a dense Qwen 3.5 multi-token-prediction head's tensors into its
//! weights, through the same validation and storage conventions as the
//! model's own language tensors.
//!
//! The head ships as its own weights file with unprefixed names (`fc.*`,
//! `layers.0.*`, `norm.weight`). Every tensor the head must have is bound,
//! and a file with any tensor more or less is refused, so a head of another
//! shape is never half-read.

use std::collections::{BTreeSet, HashMap};

use runtime_integration::{MlxRuntime, MlxSafetensors};
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::artifacts::dense_mtp_head_sidecar::{
    DENSE_MTP_HEAD_FUSION_MODULE, DENSE_MTP_HEAD_LAYER_PREFIX, dense_mtp_head_tensor_profiles,
};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::dense::mlp::bind_qwen3_5_dense_mlp_weights;
use crate::qwen3_5_core::dense::mtp_head_weights::DenseMtpHeadWeights;
use crate::qwen3_5_core::model_math::weight_binding_dtype::WeightBindingDtype;

use super::attention_binding::take_full_attention_weights;
use super::{bind_language_tensor, take_dense_projection, take_tensor};

/// The head's weights from its opened file, bound with `head_config`, which
/// carries the head's own quantization.
pub fn bind_dense_mtp_head_weights(
    runtime: &MlxRuntime,
    head_config: &Qwen3_5Config,
    head_file: MlxSafetensors,
    shipped_tensor_names: &BTreeSet<String>,
) -> Result<DenseMtpHeadWeights, TextExecutionError> {
    let tensor_profiles = dense_mtp_head_tensor_profiles(head_config).map_err(|spec_error| {
        TextExecutionError::ExpertLayerPlan {
            description: spec_error.to_string(),
        }
    })?;
    refuse_unexpected_head_tensors(shipped_tensor_names, &tensor_profiles)?;
    let mut bound_tensors = HashMap::with_capacity(tensor_profiles.len());
    let mut payload_byte_count = 0_u64;
    for tensor_profile in &tensor_profiles {
        let (bound_tensor, tensor_payload_bytes) = bind_language_tensor(
            runtime,
            head_config,
            false,
            tensor_profile,
            head_file.tensor(&tensor_profile.name)?,
            WeightBindingDtype::ForComputeDevice,
        )?;
        payload_byte_count = payload_byte_count.saturating_add(tensor_payload_bytes);
        bound_tensors.insert(tensor_profile.name.clone(), bound_tensor);
    }
    let mut head_weights = take_dense_mtp_head_weights(&mut bound_tensors, head_config, head_file)?;
    head_weights.payload_byte_count = payload_byte_count;
    if let Some(unassigned_tensor_name) = bound_tensors.keys().next() {
        return Err(TextExecutionError::UnassignedTensor {
            tensor_name: unassigned_tensor_name.clone(),
        });
    }
    Ok(head_weights)
}

/// A head file must hold exactly the head's tensors: a second layer, or a
/// projection under another name, means a head this engine would misread.
pub(crate) fn refuse_unexpected_head_tensors(
    shipped_tensor_names: &BTreeSet<String>,
    tensor_profiles: &[crate::TensorProfile],
) -> Result<(), TextExecutionError> {
    let mut expected_names = BTreeSet::new();
    for tensor_profile in tensor_profiles {
        expected_names.insert(tensor_profile.name.as_str());
    }
    for shipped_name in shipped_tensor_names {
        if !expected_names.contains(shipped_name.as_str()) {
            return Err(TextExecutionError::UnassignedTensor {
                tensor_name: shipped_name.clone(),
            });
        }
    }
    Ok(())
}

fn take_dense_mtp_head_weights(
    bound_tensors: &mut HashMap<String, mlx_bridge::MlxArray>,
    head_config: &Qwen3_5Config,
    head_file: MlxSafetensors,
) -> Result<DenseMtpHeadWeights, TextExecutionError> {
    Ok(DenseMtpHeadWeights {
        embedding_norm_scale: take_tensor(
            bound_tensors,
            "pre_fc_norm_embedding.weight".to_owned(),
        )?,
        hidden_norm_scale: take_tensor(bound_tensors, "pre_fc_norm_hidden.weight".to_owned())?,
        fusion_projection: take_dense_projection(
            bound_tensors,
            head_config,
            DENSE_MTP_HEAD_FUSION_MODULE,
        )?,
        input_normalization_weight: take_tensor(
            bound_tensors,
            format!("{DENSE_MTP_HEAD_LAYER_PREFIX}.input_layernorm.weight"),
        )?,
        attention: take_full_attention_weights(
            bound_tensors,
            head_config,
            DENSE_MTP_HEAD_LAYER_PREFIX,
        )?,
        post_attention_normalization_weight: take_tensor(
            bound_tensors,
            format!("{DENSE_MTP_HEAD_LAYER_PREFIX}.post_attention_layernorm.weight"),
        )?,
        mlp: bind_qwen3_5_dense_mlp_weights(
            bound_tensors,
            head_config,
            DENSE_MTP_HEAD_LAYER_PREFIX,
        )?,
        final_norm_scale: take_tensor(bound_tensors, "norm.weight".to_owned())?,
        head_file,
        payload_byte_count: 0,
    })
}

#[cfg(test)]
#[path = "dense_mtp_head_binding_tests.rs"]
mod dense_mtp_head_binding_tests;

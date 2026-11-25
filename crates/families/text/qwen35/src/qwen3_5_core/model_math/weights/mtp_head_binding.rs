//! Binding the `mtp.*` tensors of a Flash-Next checkpoint into the head's
//! weights, through the same validation and storage conventions as the
//! language tensors.

use std::collections::{BTreeSet, HashMap};

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::artifacts::Qwen3_5ShardIndex;
use crate::qwen3_5_core::artifacts::qwen4_exp_mtp_tensor_spec::{
    MTP_HEAD_FINAL_MIXER_PREFIX, MTP_HEAD_LAYER_PREFIX, MTP_HEAD_TENSOR_PREFIX,
    qwen4_exp_mtp_head_tensor_profiles,
};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights, Qwen3_5DecoderLayerWeights,
};
use crate::qwen3_5_core::model_math::feed_forward_weights::bind_qwen3_5_moe_feed_forward_weights;
use crate::qwen3_5_core::model_math::weight_binding_dtype::WeightBindingDtype;
use crate::qwen4_exp::hyper_connection::HyperConnectionWeights;
use crate::qwen4_exp::layer_weights::{
    FlashNextLayerWeights, LayerLaneConnections, SparseAttentionIndexerWeights,
};
use crate::qwen4_exp::mtp_head_weights::{MtpHeadShards, MtpHeadWeights};

use super::attention_binding::take_full_attention_weights;
use super::{
    bind_language_tensor, take_dense_projection, take_quantized_affine_weights, take_tensor,
};

/// The head's weights from its opened shards; `None` when the index names
/// no `mtp.*` tensor.
pub fn bind_mtp_head_weights(
    runtime: &MlxRuntime,
    qwen3_5_config: &Qwen3_5Config,
    shard_index: &Qwen3_5ShardIndex,
    head_shards: MtpHeadShards,
) -> Result<Option<MtpHeadWeights>, TextExecutionError> {
    let Some(architecture) = qwen3_5_config.qwen4_exp_architecture() else {
        return Ok(None);
    };
    let mut head_tensor_names = BTreeSet::new();
    for tensor_name in shard_index
        .auxiliary_tensor_name_to_shard_file_name()
        .keys()
    {
        if tensor_name.starts_with(MTP_HEAD_TENSOR_PREFIX) {
            head_tensor_names.insert(tensor_name.clone());
        }
    }
    if head_tensor_names.is_empty() {
        return Ok(None);
    }
    // The head stores its attention, router and shared expert unquantized
    // while the checkpoint's default is 4-bit; the index says which.
    let mut head_config = qwen3_5_config.clone();
    head_config.resolve_unquantized_modules_from_shard_index(&head_tensor_names);
    let head_tensor_profiles = qwen4_exp_mtp_head_tensor_profiles(&head_config, &architecture)
        .map_err(|artifact_error| TextExecutionError::ExpertLayerPlan {
            description: artifact_error.to_string(),
        })?;
    let mut bound_tensors = HashMap::with_capacity(head_tensor_profiles.len());
    let mut payload_byte_count = 0_u64;
    for tensor_profile in &head_tensor_profiles {
        let stored_tensor = stored_head_tensor(shard_index, &head_shards, &tensor_profile.name)?;
        let (bound_tensor, tensor_payload_bytes) = bind_language_tensor(
            runtime,
            &head_config,
            false,
            tensor_profile,
            stored_tensor,
            WeightBindingDtype::AsStored,
        )?;
        payload_byte_count = payload_byte_count.saturating_add(tensor_payload_bytes);
        bound_tensors.insert(tensor_profile.name.clone(), bound_tensor);
    }
    let mut head_weights = take_mtp_head_weights(&mut bound_tensors, &head_config, head_shards)?;
    head_weights.payload_byte_count = payload_byte_count;
    Ok(Some(head_weights))
}

fn stored_head_tensor(
    shard_index: &Qwen3_5ShardIndex,
    head_shards: &MtpHeadShards,
    canonical_tensor_name: &str,
) -> Result<MlxArray, TextExecutionError> {
    let missing_tensor = || TextExecutionError::MissingTensor {
        tensor_name: canonical_tensor_name.to_owned(),
    };
    let shard_file_name = shard_index
        .auxiliary_tensor_name_to_shard_file_name()
        .get(canonical_tensor_name)
        .ok_or_else(missing_tensor)?;
    let head_shard = head_shards
        .get(shard_file_name)
        .ok_or_else(missing_tensor)?;
    Ok(head_shard.tensor(&shard_index.stored_tensor_name_for(canonical_tensor_name))?)
}

fn take_mtp_head_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    head_config: &Qwen3_5Config,
    head_shards: MtpHeadShards,
) -> Result<MtpHeadWeights, TextExecutionError> {
    let layer = take_mtp_head_layer_weights(bound_tensors, head_config)?;
    let expert_module_prefix = format!("{MTP_HEAD_LAYER_PREFIX}.mlp.switch_mlp");
    Ok(MtpHeadWeights {
        embedding_norm_scale: take_tensor(
            bound_tensors,
            "mtp.pre_fc_norm_embedding.weight".to_owned(),
        )?,
        lane_norm_scale: take_tensor(bound_tensors, "mtp.pre_fc_norm_hidden.weight".to_owned())?,
        embedding_projection: take_dense_projection(
            bound_tensors,
            head_config,
            "mtp.fc_embedding",
        )?,
        lane_projection: take_dense_projection(bound_tensors, head_config, "mtp.fc_hidden")?,
        layer,
        expert_gate_projection: take_quantized_affine_weights(
            bound_tensors,
            head_config,
            &format!("{expert_module_prefix}.gate_proj"),
        )?,
        expert_up_projection: take_quantized_affine_weights(
            bound_tensors,
            head_config,
            &format!("{expert_module_prefix}.up_proj"),
        )?,
        expert_down_projection: take_quantized_affine_weights(
            bound_tensors,
            head_config,
            &format!("{expert_module_prefix}.down_proj"),
        )?,
        final_norm_scale: take_tensor(
            bound_tensors,
            format!("{MTP_HEAD_FINAL_MIXER_PREFIX}.hc_norm.weight"),
        )?,
        final_mixer: HyperConnectionWeights::take(
            bound_tensors,
            head_config,
            MTP_HEAD_FINAL_MIXER_PREFIX,
            false,
        )?,
        head_shards,
        payload_byte_count: 0,
    })
}

/// The head's decoder layer, shaped like a target full-attention layer.
fn take_mtp_head_layer_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    head_config: &Qwen3_5Config,
) -> Result<Qwen3_5DecoderLayerWeights, TextExecutionError> {
    Ok(Qwen3_5DecoderLayerWeights {
        input_normalization_weight: take_tensor(
            bound_tensors,
            format!("{MTP_HEAD_LAYER_PREFIX}.attn_hyper_connection.hc_norm.weight"),
        )?,
        attention_weights: Qwen3_5AttentionWeights::Full(take_full_attention_weights(
            bound_tensors,
            head_config,
            MTP_HEAD_LAYER_PREFIX,
        )?),
        post_attention_normalization_weight: take_tensor(
            bound_tensors,
            format!("{MTP_HEAD_LAYER_PREFIX}.mlp_hyper_connection.hc_norm.weight"),
        )?,
        mlp_weights: Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(
            bind_qwen3_5_moe_feed_forward_weights(
                bound_tensors,
                head_config,
                MTP_HEAD_LAYER_PREFIX,
            )?,
        ),
        flash_next: Some(Box::new(FlashNextLayerWeights {
            lane_connections: LayerLaneConnections::take(
                bound_tensors,
                head_config,
                MTP_HEAD_LAYER_PREFIX,
            )?,
            sparse_attention_indexer: Some(SparseAttentionIndexerWeights::take(
                bound_tensors,
                head_config,
                MTP_HEAD_LAYER_PREFIX,
            )?),
            ngram_layer: None,
        })),
    })
}

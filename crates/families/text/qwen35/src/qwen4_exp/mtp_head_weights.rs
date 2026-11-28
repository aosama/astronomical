//! The weights of Flash-Next's multi-token-prediction head.
//!
//! The head guesses the token after next from the target's own residual
//! lanes: two projections fuse a token's embedding with the lanes, one more
//! Flash-Next decoder layer (full attention with its indexer, routed
//! experts, hyper-connections) refines them, and a final mixer hands the
//! result to the target's output head. Its 512 experts are small enough
//! (about 1.5 GB) to stay in memory whole, so a draft step reads nothing
//! from storage.

use std::collections::{BTreeSet, HashMap};

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxSafetensors};
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::artifacts::qwen4_exp_mtp_tensor_spec::MTP_HEAD_TENSOR_PREFIX;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5DecoderLayerWeights,
};
use crate::qwen3_5_core::model_math::dense_projection::Qwen3_5DenseProjection;
use crate::qwen3_5_core::model_math::weights::bind_flash_next_mtp_head_weights;

use super::hyper_connection::HyperConnectionWeights;

/// Everything the head owns.
#[derive(Debug)]
pub struct FlashNextMtpHeadWeights {
    /// Norm of a token's embedding before `embedding_projection`.
    pub embedding_norm_scale: MlxArray,
    /// One norm across all of the target's lanes before `lane_projection`.
    pub lane_norm_scale: MlxArray,
    pub embedding_projection: Qwen3_5DenseProjection,
    /// Applied to each lane separately.
    pub lane_projection: Qwen3_5DenseProjection,
    /// The head's decoder layer; its norm fields hold the two
    /// hyper-connection norms, as on a target layer.
    pub layer: Qwen3_5DecoderLayerWeights,
    /// All of the layer's experts, stacked along their first axis.
    pub expert_gate_projection: Qwen3_5AffineWeights,
    pub expert_up_projection: Qwen3_5AffineWeights,
    pub expert_down_projection: Qwen3_5AffineWeights,
    pub final_norm_scale: MlxArray,
    pub final_mixer: HyperConnectionWeights,
    /// The opened shards, which the lazy arrays above read from.
    pub head_shards: MtpHeadShards,
    /// What the head occupies once a reply drafts with it.
    pub payload_byte_count: u64,
}

/// Loads the head when the checkpoint ships one; `None` for Qwen 3.5 and
/// for a Flash-Next checkpoint without `mtp.*` tensors. The head has a
/// shard of its own, which binding the language weights leaves untaken, so
/// this runs before or after that.
pub fn load_flash_next_mtp_head_weights(
    runtime: &MlxRuntime,
    validated_artifact: &mut ValidatedQwen3_5Artifact,
) -> Result<Option<FlashNextMtpHeadWeights>, TextExecutionError> {
    if validated_artifact
        .config()
        .qwen4_exp_architecture()
        .is_none()
    {
        return Ok(None);
    }
    let mut head_shard_file_names = BTreeSet::new();
    for (tensor_name, shard_file_name) in validated_artifact
        .shard_index()
        .auxiliary_tensor_name_to_shard_file_name()
    {
        if tensor_name.starts_with(MTP_HEAD_TENSOR_PREFIX) {
            head_shard_file_names.insert(shard_file_name.clone());
        }
    }
    if head_shard_file_names.is_empty() {
        return Ok(None);
    }
    let mut head_shards = HashMap::with_capacity(head_shard_file_names.len());
    for shard_file_name in head_shard_file_names {
        let source_id = validated_artifact.source_id_for_file_name(&shard_file_name)?;
        let shard_file = validated_artifact
            .take_safetensors_source(source_id)?
            .into_file();
        head_shards.insert(shard_file_name, runtime.load_safetensors(shard_file, None)?);
    }
    bind_flash_next_mtp_head_weights(
        runtime,
        validated_artifact.config(),
        validated_artifact.shard_index(),
        head_shards,
    )
}

/// The head's shards by file name, as `load_flash_next_mtp_head_weights` opens them.
pub type MtpHeadShards = HashMap<String, MlxSafetensors>;

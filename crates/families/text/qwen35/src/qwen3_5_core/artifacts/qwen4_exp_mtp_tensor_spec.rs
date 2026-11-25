//! The tensors of Flash-Next's multi-token-prediction head (`mtp.*`).
//!
//! The head is one more Flash-Next decoder layer with full attention, its
//! own indexer, routed experts and hyper-connections, between two small
//! projections that fuse a token's embedding with the target's residual
//! lanes and a final mixer that feeds the target's own output head. A
//! checkpoint may ship without it; these profiles describe it when present.

use crate::qwen3_5_core::configuration::Qwen4ExpArchitecture;
use crate::{TensorDtype, TensorProfile};

use super::qwen4_exp_tensor_spec::{append_hyper_connection_profiles, append_indexer_profiles};
use super::tensor_spec::{
    append_qwen3_5_full_attention_tensor_profiles, append_qwen3_5_moe_feed_forward_tensor_profiles,
    qwen3_5_tensor_profile,
};
use super::{Qwen3_5ArtifactError, Qwen3_5Config};

/// The prefix every tensor of the head shares.
pub const MTP_HEAD_TENSOR_PREFIX: &str = "mtp.";
/// The head's one decoder layer.
pub const MTP_HEAD_LAYER_PREFIX: &str = "mtp.layers.0";
/// The mixer that turns the head's lanes into one hidden state.
pub const MTP_HEAD_FINAL_MIXER_PREFIX: &str = "mtp.hyper_connection_mixer";

/// Every tensor of the head, for a configuration whose `mtp.*` modules
/// already say which are stored unquantized.
pub fn qwen4_exp_mtp_head_tensor_profiles(
    config: &Qwen3_5Config,
    architecture: &Qwen4ExpArchitecture,
) -> Result<Vec<TensorProfile>, Qwen3_5ArtifactError> {
    let hidden_size = config.hidden_size() as usize;
    let lane_width = hidden_size * architecture.residual_lane_count as usize;
    let mut profiles = Vec::new();
    append_qwen3_5_full_attention_tensor_profiles(
        &mut profiles,
        MTP_HEAD_LAYER_PREFIX,
        hidden_size,
        config.query_head_count() as usize,
        config.key_value_head_count() as usize,
        config.head_dimension() as usize,
        config,
    )?;
    append_indexer_profiles(&mut profiles, MTP_HEAD_LAYER_PREFIX, config, architecture)?;
    append_qwen3_5_moe_feed_forward_tensor_profiles(
        &mut profiles,
        MTP_HEAD_LAYER_PREFIX,
        hidden_size,
        config,
    )?;
    for connection_name in ["attn_hyper_connection", "mlp_hyper_connection"] {
        append_hyper_connection_profiles(
            &mut profiles,
            &format!("{MTP_HEAD_LAYER_PREFIX}.{connection_name}"),
            config,
            architecture,
            true,
        )?;
    }
    append_hyper_connection_profiles(
        &mut profiles,
        MTP_HEAD_FINAL_MIXER_PREFIX,
        config,
        architecture,
        false,
    )?;
    for (tensor_name, shape) in [
        ("mtp.fc_embedding.weight", vec![hidden_size, hidden_size]),
        ("mtp.fc_hidden.weight", vec![hidden_size, hidden_size]),
        ("mtp.pre_fc_norm_embedding.weight", vec![hidden_size]),
        ("mtp.pre_fc_norm_hidden.weight", vec![lane_width]),
    ] {
        profiles.push(qwen3_5_tensor_profile(
            tensor_name.to_owned(),
            TensorDtype::ModelFloat,
            shape,
        ));
    }
    Ok(profiles)
}

#[cfg(test)]
#[path = "qwen4_exp_mtp_tensor_spec_tests.rs"]
mod qwen4_exp_mtp_tensor_spec_tests;

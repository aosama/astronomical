//! The multi-token-prediction head: which of its tensors are quantized, how
//! a fused expert tensor is named once split, and whether the head can be
//! written at all. Pure rules, decided from names and header shapes.
//!
//! Releases in the Qwen3.5 family ship a small prediction head as `mtp.*`
//! tensors beside the model. mlx-lm has no module for it and drops it on
//! load, so a conversion that goes through mlx-lm loses the model's own
//! speculation head. OptiQ writes it back as `optiq/mtp.safetensors`, a
//! file no loader globs, and registers it in `config.json`.
//!
//! Ported from OptiQ's `preserve_mtp`, `_is_quantize_target`,
//! `_unfuse_experts` and `_infer_n_mtp_layers` (`optiq/runtime/mtp_convert.py`).

use std::collections::BTreeSet;

use crate::module_inventory::PassThroughTensor;

#[cfg(test)]
#[path = "mtp_sidecar_tests.rs"]
mod mtp_sidecar_tests;

/// Where the head is written, relative to the artifact directory. Under
/// `optiq/` so a `*.safetensors` glob at the root (mlx-vlm, LM Studio)
/// cannot pick it up and fail a strict load: no model has a module for it.
pub const MTP_SIDECAR_RELATIVE_PATH: &str = "optiq/mtp.safetensors";

/// The width the head's projections are quantized at: OptiQ passes the
/// conversion's default `q_bits`, which its pipeline never changes from 4.
pub const MTP_SIDECAR_BITS: u32 = 4;

/// The head's one unquantized projection. OptiQ measured the head fastest
/// with the projections at 4 bits and this fusion layer left at bfloat16.
const UNQUANTIZED_PROJECTION_NAME: &str = "mtp.fc.weight";

/// What the requantizer wrote, for the config registration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MtpSidecarSummary {
    /// Tensors in the file: three per quantized projection, one per other.
    pub tensor_count: usize,
    pub bits: u32,
    pub group_size: u32,
    /// Distinct `mtp.layers.N` blocks in the source, at least one.
    pub layer_count: usize,
}

/// Whether a head tensor is a projection weight to quantize: a `.weight`
/// that is neither the fusion layer nor a norm.
#[must_use]
pub fn is_mtp_quantize_target(tensor_name: &str) -> bool {
    let Some(module_name) = tensor_name.strip_suffix(".weight") else {
        return false;
    };
    if tensor_name == UNQUANTIZED_PROJECTION_NAME {
        return false;
    }
    let leaf_segment = module_name.rsplit('.').next().unwrap_or(module_name);
    !leaf_segment.to_lowercase().contains("norm")
}

/// How a fused routed-expert tensor of the head splits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FusedMtpExpertTensor {
    /// `mlp.experts.gate_up_proj`: gate in the lower half of axis 1, up in
    /// the upper half, one pair per expert.
    GateUp,
    /// `mlp.experts.down_proj`: one matrix per expert.
    Down,
}

/// Recognizes the two fused expert tensors, which carry no `.weight` suffix
/// and would otherwise pass through at bfloat16 beside a quantized head.
#[must_use]
pub fn fused_mtp_expert_tensor(tensor_name: &str, shape: &[usize]) -> Option<FusedMtpExpertTensor> {
    if shape.len() != 3 {
        return None;
    }
    if tensor_name.ends_with("mlp.experts.gate_up_proj") {
        return Some(FusedMtpExpertTensor::GateUp);
    }
    if tensor_name.ends_with("mlp.experts.down_proj") {
        return Some(FusedMtpExpertTensor::Down);
    }
    None
}

/// The per-expert weight name a fused tensor's slice is written under:
/// `...mlp.experts.<expert>.<projection>.weight`.
#[must_use]
pub fn unfused_mtp_expert_weight_name(
    fused_tensor_name: &str,
    fused_projection_suffix: &str,
    expert_index: usize,
    projection: &str,
) -> String {
    let experts_prefix = fused_tensor_name
        .strip_suffix(fused_projection_suffix)
        .unwrap_or(fused_tensor_name);
    format!("{experts_prefix}{expert_index}.{projection}.weight")
}

/// Distinct `mtp.layers.N` blocks among the head's tensors, at least one.
#[must_use]
pub fn mtp_layer_count(mtp_tensors: &[PassThroughTensor]) -> usize {
    let mut layer_indices = BTreeSet::new();
    for mtp_tensor in mtp_tensors {
        let segments: Vec<&str> = mtp_tensor.tensor_name.split('.').collect();
        if segments.len() >= 3
            && segments[0] == "mtp"
            && segments[1] == "layers"
            && let Ok(layer_index) = segments[2].parse::<usize>()
        {
            layer_indices.insert(layer_index);
        }
    }
    layer_indices.len().max(1)
}

/// Whether every projection of the head can be group-quantized at
/// `group_elements`. OptiQ quantizes the whole head or none of it: one
/// projection it cannot encode fails the step and the artifact ships
/// without a head, which is complete, only without speculation.
#[must_use]
pub fn is_mtp_head_encodable(mtp_tensors: &[PassThroughTensor], group_elements: usize) -> bool {
    for mtp_tensor in mtp_tensors {
        let shape = &mtp_tensor.shape;
        let is_fused_expert = fused_mtp_expert_tensor(&mtp_tensor.tensor_name, shape).is_some();
        if !is_fused_expert && !is_mtp_quantize_target(&mtp_tensor.tensor_name) {
            continue;
        }
        let has_matrix_rank = is_fused_expert || shape.len() == 2;
        let last_dimension = shape.last().copied().unwrap_or(0);
        if !has_matrix_rank || last_dimension == 0 || last_dimension % group_elements != 0 {
            return false;
        }
    }
    true
}

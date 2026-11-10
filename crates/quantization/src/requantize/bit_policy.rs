//! Which `(bits, group_size)` a tensor gets, or whether it passes through.

use crate::allocation::QuantizationPredicate;
use crate::allocation::predicate::QuantizationDecisionForModule;
use crate::bit_width::{BitWidth, GroupSize};

#[cfg(test)]
#[path = "bit_policy_tests.rs"]
mod bit_policy_tests;

/// The decision for one tensor key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorEncodingDecision {
    Encode {
        bit_width: BitWidth,
        group_size: GroupSize,
    },
    /// Written at bfloat16 (or as-is for integer tensors).
    PassThrough,
}

/// The structural policy's parameters, OptiQ's `moe_bit_policy` defaults.
///
/// The defaults are not arbitrary. On DeepSeek-V4 a first pass put attention
/// and the shared experts at 4-bit and the result was incoherent, while the
/// same experts at 2-bit with attention at 6 was fine. Sparse or compressed
/// attention has less redundancy to give up than a dense model's, and it is
/// a rounding error in the parameter count either way.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StructuralMixtureOfExpertsPolicy {
    pub routed_expert_bits: BitWidth,
    pub attention_bits: BitWidth,
    pub shared_expert_bits: BitWidth,
    pub embedding_and_head_bits: BitWidth,
    pub dense_bits: BitWidth,
    /// Group size for experts, attention, and dense tensors.
    pub expert_group_size: GroupSize,
    /// Group size for the shared expert and the embedding and head.
    pub high_precision_group_size: GroupSize,
}

impl StructuralMixtureOfExpertsPolicy {
    /// OptiQ's `moe_bit_policy` rules for a non-router key, first match
    /// wins: routed expert, shared expert, embedding or head, attention, then
    /// everything else at the dense width.
    #[must_use]
    pub fn decide(&self, tensor_key: &str) -> TensorEncodingDecision {
        let (bit_width, group_size) = if is_routed_expert_tensor_key(tensor_key) {
            (self.routed_expert_bits, self.expert_group_size)
        } else if tensor_key.contains(".shared_") {
            (self.shared_expert_bits, self.high_precision_group_size)
        } else if is_embedding_or_head_tensor_key(tensor_key) {
            (self.embedding_and_head_bits, self.high_precision_group_size)
        } else if is_attention_tensor_key(tensor_key) {
            (self.attention_bits, self.expert_group_size)
        } else {
            (self.dense_bits, self.expert_group_size)
        };
        TensorEncodingDecision::Encode {
            bit_width,
            group_size,
        }
    }

    /// Experts 2-bit/128, attention 6-bit/128, shared 8-bit/64,
    /// embeddings 8-bit/64, dense 6-bit/128.
    #[must_use]
    pub fn optiq_defaults() -> Self {
        Self {
            routed_expert_bits: BitWidth::new(2).expect("2 is valid"),
            attention_bits: BitWidth::new(6).expect("6 is valid"),
            shared_expert_bits: BitWidth::new(8).expect("8 is valid"),
            embedding_and_head_bits: BitWidth::new(8).expect("8 is valid"),
            dense_bits: BitWidth::new(6).expect("6 is valid"),
            expert_group_size: GroupSize::new(128).expect("128 is valid"),
            high_precision_group_size: GroupSize::new(64).expect("64 is valid"),
        }
    }
}

/// How a tensor key maps to a decision.
#[derive(Clone, Debug)]
pub enum BitPolicy {
    /// The allocation's predicate; `KeepBfloat16` becomes `PassThrough`.
    FromPredicate(QuantizationPredicate),
    /// Every quantizable tensor at one width (the uniform baseline).
    Uniform {
        bit_width: BitWidth,
        group_size: GroupSize,
    },
    /// Name-based structural rules for a large sparse model.
    Structural(StructuralMixtureOfExpertsPolicy),
    /// Nothing is encoded: every module is written at bfloat16 in the
    /// output layout. This turns a source in an original release's layout
    /// into the bfloat16 model the sensitivity probe's engine can load.
    KeepEverythingBfloat16,
}

impl BitPolicy {
    /// Whether a conversion under this policy writes the source's
    /// multi-token-prediction head as a sidecar. Only the final conversion
    /// does: a uniform baseline or a bfloat16 copy exists to be probed, and
    /// a probe never runs the head.
    #[must_use]
    pub const fn writes_mtp_sidecar(&self) -> bool {
        matches!(self, Self::FromPredicate(_) | Self::Structural(_))
    }

    /// Decides for one `.weight` tensor key.
    ///
    /// The allocation's predicate looks the module (the key without
    /// `.weight`) up, routers included: where a router reaches the plan as a
    /// module it is a plain linear layer, and OptiQ quantizes it at the
    /// width its measured sensitivity earned. The uniform policy applies its
    /// one width, except to the two gates mlx-lm's Qwen mixture-of-experts
    /// class pins (the router and the shared expert's output gate: 8 bits,
    /// group size 64), so a uniform baseline is the stock conversion it
    /// stands in for and a sweep over it starts from the same widths. The structural
    /// policy applies OptiQ's `moe_bit_policy` name rules and never encodes
    /// a router: it serves sources whose router is a raw parameter, where
    /// packing the input dimension breaks the `x @ weight.T` shape.
    #[must_use]
    pub fn decide(&self, tensor_key: &str) -> TensorEncodingDecision {
        match self {
            Self::KeepEverythingBfloat16 => TensorEncodingDecision::PassThrough,
            Self::FromPredicate(predicate) => {
                let module_name = tensor_key.strip_suffix(".weight").unwrap_or(tensor_key);
                decision_from_predicate(predicate.decision_for(module_name))
            }
            Self::Uniform {
                bit_width,
                group_size,
            } => uniform_decision(tensor_key, *bit_width, *group_size),
            Self::Structural(structural_policy) => {
                structural_decision(structural_policy, tensor_key)
            }
        }
    }

    /// The dimensional check applied after the name-based decision: only
    /// 2-D and 3-D tensors are group-quantizable and the last dimension has
    /// to divide the group. A tensor that fails at the planned group size
    /// takes the largest supported group not above it that does divide
    /// (Qwen3.8-Flash-Next n-gram shards are 160 wide and land on 32);
    /// anything else passes through.
    #[must_use]
    pub fn adjust_for_shape(
        decision: TensorEncodingDecision,
        shape: &[usize],
    ) -> TensorEncodingDecision {
        let TensorEncodingDecision::Encode {
            bit_width,
            group_size,
        } = decision
        else {
            return decision;
        };
        let Some(last_dimension) = shape.last().copied() else {
            return TensorEncodingDecision::PassThrough;
        };
        if !matches!(shape.len(), 2 | 3) {
            return TensorEncodingDecision::PassThrough;
        }
        match fitting_group_size(last_dimension, group_size) {
            Some(fitting_group_size) => TensorEncodingDecision::Encode {
                bit_width,
                group_size: fitting_group_size,
            },
            None => TensorEncodingDecision::PassThrough,
        }
    }
}

/// The uniform policy's decision: its one width, except for the two gates
/// mlx-lm pins.
fn uniform_decision(
    tensor_key: &str,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> TensorEncodingDecision {
    if is_gate_pinned_by_mlx_lm(tensor_key) {
        return TensorEncodingDecision::Encode {
            bit_width: BitWidth::new(UNIFORM_ROUTER_BITS).expect("8 is a valid width"),
            group_size: GroupSize::new(UNIFORM_ROUTER_GROUP_SIZE_ELEMENTS)
                .expect("64 is a supported group size"),
        };
    }
    TensorEncodingDecision::Encode {
        bit_width,
        group_size,
    }
}

/// The structural policy's decision; a router is never encoded under it.
fn structural_decision(
    structural_policy: &StructuralMixtureOfExpertsPolicy,
    tensor_key: &str,
) -> TensorEncodingDecision {
    if is_router_tensor_key(tensor_key) {
        return TensorEncodingDecision::PassThrough;
    }
    structural_policy.decide(tensor_key)
}

/// The largest fallback group size not above the planned one that divides
/// `last_dimension`, or `None` when none does.
fn fitting_group_size(last_dimension: usize, planned_group_size: GroupSize) -> Option<GroupSize> {
    for fallback_group_elements in FALLBACK_GROUP_SIZES_ELEMENTS {
        let fits_planned_group = fallback_group_elements <= planned_group_size.elements();
        if fits_planned_group && last_dimension.is_multiple_of(fallback_group_elements as usize) {
            return GroupSize::new(fallback_group_elements).ok();
        }
    }
    None
}

/// OptiQ's `is_router`, kept beside the policy because the requantizer's
/// rule is looser than the allocator taxonomy's (`expert_bias` and
/// `.ffn.gate.` included).
#[must_use]
pub fn is_router_tensor_key(tensor_key: &str) -> bool {
    tensor_key.ends_with(".gate.weight")
        || tensor_key.ends_with(".gate")
        || tensor_key.contains(".ffn.gate.")
        || tensor_key.contains("expert_bias")
        || tensor_key.contains(".router.")
}

/// mlx-lm's Qwen3.5 `quant_predicate`: a module path ending in `mlp.gate`
/// (the router) or `shared_expert_gate` is quantized at 8 bits whatever the
/// conversion's width. Both are tiny and decide which experts run and how
/// much the shared expert contributes.
fn is_gate_pinned_by_mlx_lm(tensor_key: &str) -> bool {
    let module_path = tensor_key.strip_suffix(".weight").unwrap_or(tensor_key);
    module_path.ends_with("mlp.gate") || module_path.ends_with("shared_expert_gate")
}

/// OptiQ's shardwise `is_expert`: a routed-expert weight, fused or not,
/// excluding the shared expert.
#[must_use]
pub fn is_routed_expert_tensor_key(tensor_key: &str) -> bool {
    if tensor_key.contains(".shared_") {
        return false;
    }
    tensor_key.contains(".experts.")
        || tensor_key.contains(".switch_mlp.")
        || tensor_key.contains(".switch_glu.")
}

/// The policy the final conversion uses: the allocation's predicate.
#[must_use]
pub fn policy_from_predicate(predicate: QuantizationPredicate) -> BitPolicy {
    BitPolicy::FromPredicate(predicate)
}

/// The group sizes tried, largest first, for a tensor whose last dimension
/// does not divide the planned group: OptiQ's streaming convert order.
const FALLBACK_GROUP_SIZES_ELEMENTS: [u32; 3] = [128, 64, 32];

/// The width and group size mlx-lm's Qwen mixture-of-experts
/// `quant_predicate` pins its two gates at, whatever the uniform width is.
const UNIFORM_ROUTER_BITS: u32 = 8;
const UNIFORM_ROUTER_GROUP_SIZE_ELEMENTS: u32 = 64;

/// Maps an allocation decision onto the requantizer's: the keep-bfloat16
/// tier is written as-is.
fn decision_from_predicate(
    module_decision: QuantizationDecisionForModule,
) -> TensorEncodingDecision {
    match module_decision {
        QuantizationDecisionForModule::Quantize {
            bit_width,
            group_size,
        } => TensorEncodingDecision::Encode {
            bit_width,
            group_size,
        },
        QuantizationDecisionForModule::KeepBfloat16 => TensorEncodingDecision::PassThrough,
    }
}

/// `embed_tokens` anywhere, or `lm_head` as the first or a later segment.
fn is_embedding_or_head_tensor_key(tensor_key: &str) -> bool {
    tensor_key.contains("embed_tokens")
        || tensor_key.starts_with("lm_head")
        || tensor_key.contains(".lm_head")
}

/// OptiQ's attention spellings for the structural policy.
fn is_attention_tensor_key(tensor_key: &str) -> bool {
    tensor_key.contains(".attn.")
        || tensor_key.contains("self_attn")
        || tensor_key.contains("attention")
}

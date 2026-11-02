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
}

impl BitPolicy {
    /// Decides for one `.weight` tensor key.
    ///
    /// Routers are always `PassThrough` regardless of policy: the router
    /// selects experts, and packing its input dimension breaks the
    /// `x @ weight.T` shape. Otherwise the variant decides: the allocation's
    /// predicate looks the module (the key without `.weight`) up, the uniform
    /// policy applies its one width, and the structural policy applies
    /// OptiQ's `moe_bit_policy` name rules.
    #[must_use]
    pub fn decide(&self, tensor_key: &str) -> TensorEncodingDecision {
        if is_router_tensor_key(tensor_key) {
            return TensorEncodingDecision::PassThrough;
        }
        match self {
            Self::FromPredicate(predicate) => {
                let module_name = tensor_key.strip_suffix(".weight").unwrap_or(tensor_key);
                decision_from_predicate(predicate.decision_for(module_name))
            }
            Self::Uniform {
                bit_width,
                group_size,
            } => TensorEncodingDecision::Encode {
                bit_width: *bit_width,
                group_size: *group_size,
            },
            Self::Structural(structural_policy) => structural_policy.decide(tensor_key),
        }
    }

    /// The dimensional check applied after the name-based decision: only
    /// 2-D and 3-D tensors are group-quantizable and the last dimension has
    /// to divide the group. A tensor that fails at the planned group size
    /// but divides 32 is encoded at group size 32 instead (Qwen3.8-Flash-Next
    /// n-gram shards are 160 wide); anything else passes through.
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
        if last_dimension % group_size.elements() as usize == 0 {
            return decision;
        }
        if last_dimension % SMALLEST_GROUP_SIZE_ELEMENTS as usize == 0 {
            return TensorEncodingDecision::Encode {
                bit_width,
                group_size: GroupSize::new(SMALLEST_GROUP_SIZE_ELEMENTS)
                    .expect("32 is a supported group size"),
            };
        }
        TensorEncodingDecision::PassThrough
    }
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

/// The smallest MLX affine group, the fallback for a tensor whose last
/// dimension does not divide the planned group (Qwen3.8-Flash-Next n-gram
/// shards are 160 wide).
const SMALLEST_GROUP_SIZE_ELEMENTS: u32 = 32;

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

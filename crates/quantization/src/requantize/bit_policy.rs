//! Which `(bits, group_size)` a tensor gets, or whether it passes through.

use crate::allocation::QuantizationPredicate;
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
    /// Work to port (OptiQ's `moe_bit_policy` closure and the predicate
    /// path): routers are always `PassThrough` regardless of policy (the
    /// router selects experts; packing its input dimension breaks the
    /// `x @ weight.T` shape). For `Structural`: routed expert → expert bits
    /// and group; `.shared_` → shared bits and high-precision group;
    /// `embed_tokens`, `lm_head` → embedding bits and high-precision group;
    /// `.attn.`, `self_attn`, `attention` → attention bits and expert group;
    /// else dense bits and expert group. For `FromPredicate`: look the module
    /// name (key without `.weight`) up and map the decision. For `Uniform`:
    /// the one width.
    #[must_use]
    pub fn decide(&self, tensor_key: &str) -> TensorEncodingDecision {
        let _ = tensor_key;
        todo!("route by policy variant as documented above")
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
        let _ = (decision, shape);
        todo!(
            "downgrade to group 32 when it divides, else PassThrough, for tensors that fail the planned group"
        )
    }
}

/// OptiQ's `is_router`, kept beside the policy because the requantizer's
/// rule is looser than the allocator taxonomy's (`expert_bias` and
/// `.ffn.gate.` included).
#[must_use]
pub fn is_router_tensor_key(tensor_key: &str) -> bool {
    let _ = tensor_key;
    todo!("ends_with .gate.weight or .gate, contains .ffn.gate. or expert_bias or .router.")
}

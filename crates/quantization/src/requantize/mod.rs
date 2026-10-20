//! Requantize a checkpoint tensor by tensor, without ever building the model.
//!
//! A 300B checkpoint cannot be instantiated on a 24 GB machine, and whole-model
//! quantization over fused `[n_experts, out, in]` expert tensors is fatal at
//! 36 GB just as surely. This walks the source shards instead: for each
//! tensor it dequantizes (if the source is quantized), re-encodes at the
//! width the policy asks for, and writes it straight back out. Nothing is
//! held but the tensor in hand, so peak memory tracks the largest single
//! tensor rather than the model. OptiQ measured about 12 GB peak against a
//! 167 GB DeepSeek-V4 source on a 36 GB Mac.
//!
//! The requantizer serves two callers: the final mixed-precision conversion
//! (its policy is the [`QuantizationPredicate`] from an allocation) and the
//! uniform-4-bit baseline build for the streaming sensitivity reference (its
//! policy is "4 bits everywhere quantizable"). A third, measurement-free
//! policy, [`bit_policy::structural_mixture_of_experts_policy`], encodes the
//! structural rule that works for every large sparse model: routed experts
//! hold almost all the parameters and tolerate the least precision per byte,
//! attention and the shared expert hold few and tolerate almost none.
//!
//! Why this is the stage that most benefits from Rust: the Python tool runs
//! one tensor at a time on one thread with Python-level dispatch between
//! every MLX call. The port can overlap the next shard's header parse and
//! tensor reads with the current tensor's GPU encode, and write shards from
//! a separate thread, while keeping the same single-tensor peak. The
//! acceptance throughput journey measures tensors per second against the
//! Python tool on the same source.

pub mod bit_policy;
pub mod shard_writer;
pub mod tensor_plan;

#[cfg(feature = "direct-mlx")]
pub mod driver;

pub use bit_policy::{BitPolicy, TensorEncodingDecision};
pub use tensor_plan::{RequantizeTensorPlan, TensorPlanEntry};

use crate::allocation::QuantizationPredicate;

/// The files copied verbatim from source to output beside the shards.
pub const DEFAULT_COPY_PATTERNS: &[&str] = &[
    "tokenizer*",
    "*.jinja",
    "generation_config.json",
    "chat_template*",
    "preprocessor_config.json",
    "special_tokens_map.json",
];

/// Target shard size: OptiQ's 4e9 bytes.
pub const DEFAULT_SHARD_BYTES: u64 = 4_000_000_000;

/// What the requantizer produced, for the pipeline's summary and metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequantizeOutcome {
    pub shard_count: usize,
    pub total_output_bytes: u64,
    pub quantized_tensor_count: usize,
    pub passed_through_tensor_count: usize,
    /// Per-module `(bits, group_size)` as actually written, which can differ
    /// from the plan: a module whose last dimension does not divide its
    /// planned group size falls back to 32, and a router is never quantized.
    /// The artifact metadata must describe what was written, not the plan.
    pub written_quantization_by_module: std::collections::BTreeMap<String, (u32, u32)>,
    pub peak_mlx_memory_bytes: u64,
}

/// The policy the final conversion uses: the allocation's predicate.
#[must_use]
pub fn policy_from_predicate(predicate: QuantizationPredicate) -> BitPolicy {
    BitPolicy::FromPredicate(predicate)
}

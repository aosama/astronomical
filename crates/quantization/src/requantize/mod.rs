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
//! (its policy is the [`crate::allocation::QuantizationPredicate`] from an allocation) and the
//! uniform-4-bit baseline build for the streaming sensitivity reference (its
//! policy is "4 bits everywhere quantizable"). A third, measurement-free
//! policy, [`bit_policy::StructuralMixtureOfExpertsPolicy`], encodes the
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
pub mod copied_source_files;
pub mod requantize_outcome;
pub mod shard_writer;
pub mod tensor_plan;

#[cfg(feature = "direct-mlx")]
pub mod driver;

pub use bit_policy::{BitPolicy, TensorEncodingDecision, policy_from_predicate};
pub use copied_source_files::DEFAULT_COPY_PATTERNS;
pub use requantize_outcome::RequantizeOutcome;
pub use shard_writer::DEFAULT_SHARD_BYTES;
pub use tensor_plan::{RequantizeTensorPlan, TensorPlanEntry};

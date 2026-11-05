//! Apple Neural Engine support for prompt prefill (issue #161).
//!
//! This crate proves, from the lowest level upward, that a dense prefill
//! projection can be placed on the Neural Engine through public Core ML, that
//! its numbers match an fp32 reference, and how fast it runs next to the GPU.
//! Serving reaches it only through the opt-in
//! `experimental_neural_engine_prefill_enabled` setting, by way of the whole
//! MLP program and `mlx_hand_off`; everything else is research evidence.
//!
//! The pieces, bottom to top:
//!
//! - [`weights`]: fp16 conversion and the compressed forms weights can take.
//! - `wire`: protobuf, MIL messages, and the weight blob file.
//! - [`documents`]: a projection or a whole MLP as a Core ML model document,
//!   and the `.mlpackage` directory it ships in.
//! - [`core_ml`]: a written model loaded for chosen compute units, with
//!   `MLComputePlan`'s placement verdict, predictions, and which compute devices
//!   Core ML can see. It takes a path and knows nothing of how a document is
//!   encoded.
//! - `mlx_hand_off`: activations into a program's input surface and its
//!   output surface back into MLX.
//! - [`accelerate_reference`]: the fp32 numerical oracle.
//! - [`numerical_comparison`], [`timing_summary`], and [`process_memory`]:
//!   the evidence format.

pub mod accelerate_reference;
pub mod core_ml;
pub mod deterministic_values;
pub mod documents;
#[cfg(feature = "mlx-hand-off")]
pub mod mlx_hand_off;
mod native_bridge;
pub mod numerical_comparison;
pub mod process_memory;
pub mod projection_shape;
pub mod timing_summary;
pub mod weights;
mod wire;

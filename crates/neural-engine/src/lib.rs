//! Apple Neural Engine support for prompt prefill (issue #161), through public
//! Core ML only.
//!
//! Serving reaches this crate only through the opt-in
//! `experimental_neural_engine_prefill_enabled` setting. The evidence that these
//! programs are correct and worth running lives in the research crate beside the
//! acceptance tests, `verification/neural-engine`.
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
//! - `mlx_hand_off`: activations into a program's input surface and its output
//!   surface back into MLX.

pub mod core_ml;
pub mod documents;
#[cfg(feature = "mlx-hand-off")]
pub mod mlx_hand_off;
pub mod projection_shape;
pub mod weights;
mod wire;

//! Apple Neural Engine support for prompt prefill (issue #161).
//!
//! This crate proves, from the lowest level upward, that a dense prefill
//! projection can be placed on the Neural Engine through public Core ML, that
//! its numbers match an fp32 reference, and how fast it runs next to the GPU.
//! Serving reaches it only through the opt-in
//! `experimental_neural_engine_prefill_enabled` setting, by way of the whole
//! MLP program and [`mlx_hand_off`]; everything else is research evidence.
//!
//! The pieces, bottom to top:
//!
//! - [`compute_device_report`]: which compute devices Core ML can see.
//! - [`projection_model_interface`]: the input, output, and errors every
//!   projection model document shares.
//! - [`core_ml_projection_document`]: a projection written as a Core ML
//!   model document directly from Rust.
//! - [`ml_program_projection_document`]: the same projection as an ML
//!   Program whose weights can stay 4-bit ([`affine_quantized_weights`]),
//!   packaged by [`ml_package_layout`].
//! - [`ml_program_mlp_document`]: a whole `silu(gate) · up → down` MLP as one
//!   ML Program, so token-split prefill needs no hand-off inside the chain.
//! - [`core_ml_projection`]: any of these models loaded for chosen compute units,
//!   with `MLComputePlan`'s placement verdict and per-prediction timing.
//! - [`mlx_hand_off`]: activations into a program's input surface and its
//!   output surface back into MLX.
//! - [`accelerate_reference`]: the fp32 numerical oracle.
//! - [`numerical_comparison`], [`timing_summary`], and [`process_memory`]:
//!   the evidence format.

pub mod accelerate_reference;
pub mod affine_quantized_weights;
pub mod compute_device_report;
pub mod core_ml_projection;
pub mod core_ml_projection_document;
pub mod deterministic_values;
pub mod half_precision;
mod mil_program_wire;
pub mod ml_package_layout;
pub mod ml_program_mlp_document;
pub mod ml_program_projection_document;
pub mod ml_program_rotated_mlp_document;
mod ml_program_weight_blob;
#[cfg(feature = "mlx-hand-off")]
pub mod mlx_hand_off;
mod native_bridge;
pub mod numerical_comparison;
pub mod process_memory;
pub mod projection_model_interface;
pub mod projection_shape;
mod protobuf_wire;
mod row_major_block;
pub mod shared_lookup_tables;
pub mod timing_summary;

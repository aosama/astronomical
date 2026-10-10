//! Shared Qwen3.5 family model math and typed weight structures (issue #1132).
//!
//! These are the pieces both engines execute identically and neither may fork:
//! the family execution error, the forward input/output contract, tensor
//! slicing helpers, typed decoder-layer weights (dense and mixture-of-experts),
//! the MoE routing and expert-output combination math, the GDN (gated delta
//! net) sequence kernels, and the shard-to-typed-weights binding. Every module
//! keeps the cfg attribute it carried before the move so the direct-mlx
//! feature matrix is unchanged.

#[cfg(feature = "direct-mlx")]
pub(crate) mod decoder_layer_weights;
#[cfg(feature = "direct-mlx")]
pub(crate) mod error;
#[cfg(feature = "direct-mlx")]
pub(crate) mod feed_forward_weights;
#[cfg(feature = "direct-mlx")]
pub(crate) mod forward_contract;
#[cfg(feature = "direct-mlx")]
pub(crate) mod gated_delta_boundary_checkpoints;
#[cfg(feature = "direct-mlx")]
pub(crate) mod gated_delta_pipelined_kernel;
pub(crate) mod gated_delta_sequence;
pub(crate) mod gated_delta_sequence_contract;
#[cfg(feature = "direct-mlx")]
pub(crate) mod gated_delta_step;
#[cfg(feature = "direct-mlx")]
pub(crate) mod gdn_decode_prework_kernel;
#[cfg(feature = "direct-mlx")]
pub(crate) mod output_combination;
#[cfg(feature = "direct-mlx")]
pub(crate) mod routing;
#[cfg(feature = "direct-mlx")]
pub(crate) mod tensor_slicing;
#[cfg(feature = "direct-mlx")]
pub(crate) mod weights;
#[cfg(feature = "direct-mlx")]
pub(crate) mod weights_validation;

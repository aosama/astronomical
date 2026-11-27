//! Shared Qwen3.5 family model math and typed weight structures (issue #1132).
//!
//! These are the pieces every Qwen3.5 model executes:
//! the family execution error, the forward input/output contract, tensor
//! slicing helpers, typed decoder-layer weights (dense and mixture-of-experts),
//! the MoE routing and expert-output combination math, the GDN (gated delta
//! net) sequence kernels, and the shard-to-typed-weights binding. Every module
//! keeps the cfg attribute it carried before the move so the direct-mlx
//! feature matrix is unchanged.

#[cfg(feature = "direct-mlx")]
pub mod decoder_layer_weights;
#[cfg(feature = "direct-mlx")]
pub mod dense_projection;
#[cfg(feature = "direct-mlx")]
pub mod differentiable_gated_delta;
#[cfg(feature = "direct-mlx")]
pub mod feed_forward_weights;
#[cfg(feature = "direct-mlx")]
pub mod folded_ternary_weights;
#[cfg(feature = "direct-mlx")]
pub mod forward_contract;
#[cfg(feature = "direct-mlx")]
pub mod gated_delta_boundary_checkpoints;
#[cfg(feature = "direct-mlx")]
pub mod gated_delta_input_rules;
#[cfg(feature = "direct-mlx")]
pub mod gated_delta_pipelined_kernel;
#[cfg(feature = "direct-mlx")]
pub mod gated_delta_sequence;
#[cfg(feature = "direct-mlx")]
pub mod gated_delta_sequence_contract;
#[cfg(feature = "direct-mlx")]
pub mod gated_delta_step;
#[cfg(feature = "direct-mlx")]
pub mod gated_delta_tree;
#[cfg(all(test, feature = "direct-mlx"))]
mod gated_delta_tree_tests;
#[cfg(feature = "direct-mlx")]
pub mod gdn_decode_prework_kernel;
#[cfg(feature = "direct-mlx")]
pub mod hadamard_rotation;
#[cfg(feature = "direct-mlx")]
pub mod output_combination;
#[cfg(feature = "direct-mlx")]
pub mod routing;
#[cfg(feature = "direct-mlx")]
pub mod row_exact_tree_attention;
#[cfg(all(test, feature = "direct-mlx"))]
mod row_exact_tree_attention_tests;
#[cfg(feature = "direct-mlx")]
pub mod signed_rotation_kernel;
#[cfg(feature = "direct-mlx")]
pub mod sorted_expert_shared_combination;
#[cfg(all(test, feature = "direct-mlx"))]
mod sorted_expert_shared_combination_tests;
#[cfg(feature = "direct-mlx")]
pub mod streamed_layer_binding;
#[cfg(all(test, feature = "direct-mlx"))]
#[path = "model_math/streamed_layer_binding_tests.rs"]
mod streamed_layer_binding_tests;
#[cfg(feature = "direct-mlx")]
pub mod tensor_slicing;
#[cfg(feature = "direct-mlx")]
pub mod weight_binding_dtype;
#[cfg(feature = "direct-mlx")]
pub mod weights;
#[cfg(feature = "direct-mlx")]
pub mod weights_validation;

#[cfg(all(test, feature = "direct-mlx"))]
#[path = "model_math/gated_delta_boundary_checkpoints_tests.rs"]
mod gated_delta_boundary_checkpoints_tests;

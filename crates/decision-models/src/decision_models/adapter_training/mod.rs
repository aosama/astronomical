//! Training add-on weights inside every layer of a decision model, with
//! gradients from MLX itself, and the parts every family's whole-model
//! training shares: the loss transform, the micro-batch objective, the
//! parameter layout, and recomputing layers for passes too long to keep.

pub mod layer_recomputation;
pub mod loss_gradient;
pub mod micro_batch_objective;
pub mod mlx_buffer_cache_reclaim;
#[cfg(test)]
mod qwen_gated_delta_gradient_tests;
pub mod whole_model_layout;

//! Training add-on weights inside every layer of a decision model, with
//! gradients from MLX itself, and the parts every family's whole-model
//! training shares: the loss transform, the micro-batch objective, the
//! parameter layout, and recomputing layers for passes too long to keep.

pub(crate) mod layer_recomputation;
pub(crate) mod loss_gradient;
pub(crate) mod micro_batch_objective;
pub(crate) mod mlx_buffer_cache_reclaim;
#[cfg(test)]
mod qwen_gated_delta_gradient_tests;
pub(crate) mod whole_model_layout;

//! Training a decision model's final layers on the GPU, inside the worker
//! that serves the model.
//!
//! The worker already holds the model, its MLX runtime and the GPU, so it
//! reads every example once, uploads what the final layers read, and trains
//! there; only the trained tensors and the scores the gate needs go back.

pub(crate) mod gpu_adam;
pub(crate) mod gpu_heads;
pub(crate) mod gpu_math;
pub(crate) mod gpu_training;
pub(crate) mod gpu_upload;

#[cfg(test)]
pub(crate) mod gpu_test_support;
pub(crate) mod question_batch;
pub(crate) mod row_objective;
pub(crate) mod training_session;

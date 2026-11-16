//! Training a decision model on the GPU, inside the worker that serves it:
//! the training job, the loop that moves the parameters, and the final
//! layers' own scorer.
//!
//! The worker already holds the model, its MLX runtime and the GPU, so it
//! reads every example once and trains there; only the trained tensors and
//! the scores the gate needs go back. A final-layer training uploads what
//! the final layers read; a whole-model training (`adapter_training`) runs
//! the model itself through the same loop.

pub(crate) mod gpu_adam;
pub(crate) mod gpu_heads;
pub(crate) mod gpu_math;
pub(crate) mod gpu_training;
pub(crate) mod gpu_upload;

#[cfg(test)]
pub(crate) mod gpu_test_support;
pub(crate) mod question_batch;
pub(crate) mod question_selection;
pub(crate) mod row_objective;
pub(crate) mod stepped_scoring;
pub(crate) mod trainable_scorer;
pub(crate) mod training_phase;
pub(crate) mod training_session;

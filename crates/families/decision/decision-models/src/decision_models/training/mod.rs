//! Training a decision model on the GPU, inside the worker that serves it:
//! the training job, the loop that moves the parameters, and the final
//! layers' own scorer.
//!
//! The worker already holds the model, its MLX runtime and the GPU, so it
//! reads every example once and trains there; only the trained tensors and
//! the scores the gate needs go back. A final-layer training uploads what
//! the final layers read; a whole-model training (`adapter_training`) runs
//! the model itself through the same loop.

// A diagnostic for acceptance journeys, not for serving: built only for
// tests and with the `gradient-check` feature.
#[cfg(any(test, feature = "gradient-check"))]
pub mod directional_gradient_check;
pub mod gpu_adam;
pub mod gpu_heads;
pub mod gpu_math;
pub mod gpu_training;
pub mod gpu_upload;

#[cfg(test)]
pub mod gpu_test_support;
pub mod question_batch;
pub mod question_selection;
pub mod row_objective;
pub mod stepped_scoring;
pub mod trainable_scorer;
pub mod training_phase;
pub mod training_session;

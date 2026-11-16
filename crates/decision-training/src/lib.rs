//! Calibrating a decision model on a person's own labelled examples.
//!
//! A decision model returns a probability for every allowed answer, and a
//! fitted temperature is what makes "0.9" mean right about nine times in
//! ten. The temperatures a checkpoint ships with were fitted on its
//! publisher's data; this crate fits them again on the person's data and
//! writes a calibrated copy of the model that the daemon serves like any
//! other.
//!
//! The steps, one module each, all on the CPU:
//!
//! 1. [`labelled_example`]: read the examples file.
//! 2. The command-line tool sends every example to the model and keeps the
//!    raw scores; [`scored_question`] checks each answer before it is used.
//! 3. [`holdout_split`]: hold out a fifth of the states, never splitting one.
//! 4. [`calibration_plan`]: fit per kind ([`temperature_fit`]) and keep a fit
//!    only when it beats the shipped temperatures on held-out questions
//!    ([`calibration_metrics`]).
//! 5. [`calibrated_model_folder`]: copy the model, prove the copy byte for
//!    byte, and add the calibration file.
//!
//! # Why so many guards
//!
//! Tuning tools are known for making a model quietly worse: writing into
//! the weights they read, fitting on the data they then report on, letting
//! an old setting override a new fit, or mixing up option order. Each of
//! those has a guard here, and a calibration that cannot pass all of them
//! writes nothing. Calibration also never changes a weight: only the
//! temperature that turns scores into probabilities moves, so which answer a
//! model picks stays exactly the same.

#![forbid(unsafe_code)]

pub mod anchored_adam;
pub mod calibrated_model_folder;
pub mod calibration_metrics;
pub mod calibration_plan;
pub mod clef_trainable;
pub mod dense_math;
pub mod gelu;
pub mod head_parameters;
pub mod head_training;
pub mod holdout_split;
pub mod kev_trainable;
pub mod labelled_example;
pub mod laya_trainable;
pub mod progress_line;
pub mod safetensors_slices;
pub mod scored_question;
pub mod shipped_head;
pub mod temperature_fit;
pub mod temperature_softmax;
pub mod trainable_head;
pub mod training_losses;
pub mod training_split;
pub mod training_verdict;

#[cfg(test)]
mod test_support;

//! The settings and shared pieces of head training, which runs on the GPU
//! in the inference worker.
//!
//! Each run starts from the weights the folder serves and walks the
//! training questions in shuffled mini-batches. Every step minimizes the
//! label-smoothed cross-entropy on a batch plus the disagreement with the
//! shipped model on a slice of the built-in check set; the optimizer pulls
//! toward the shipped weights as it goes. After each pass the choosing
//! questions are scored, and the run keeps the weights from its best pass,
//! stopping once several passes in a row bring no improvement.
//!
//! Runs differ only in the order they see the batches. Laya's own training
//! found that whether a head learns at all can depend on that luck (6 of 15
//! runs never left the plateau on identical data), so several runs are made
//! and the choosing questions pick the best.

use crate::anchored_adam::{AdamSettings, HEAD_TRAINING_ADAM_SETTINGS};

/// The settings of one training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingSettings {
    pub adam: AdamSettings,
    /// Labelled questions per optimizer step.
    pub batch_question_count: usize,
    /// Check-set questions mixed into each step.
    pub check_questions_per_batch: usize,
    /// How much keeping the check set's answers counts against fitting the labels.
    pub check_weight: f64,
    pub maximum_epochs: usize,
    /// Passes without a better choosing score before a run stops.
    pub patience_epochs: usize,
    pub seeds: &'static [u64],
}

/// The stage 2 settings: small batches for a few hundred examples, at most
/// 40 passes, three differently shuffled runs.
pub const HEAD_TRAINING_SETTINGS: TrainingSettings = TrainingSettings {
    adam: HEAD_TRAINING_ADAM_SETTINGS,
    batch_question_count: 16,
    check_questions_per_batch: 16,
    check_weight: 1.0,
    maximum_epochs: 40,
    patience_epochs: 6,
    seeds: &[1, 2, 3],
};

/// The whole-model settings. Each pass runs the whole model backwards, so
/// time is the cost to hold down: one run instead of three, and half the
/// check-set questions in each step.
///
/// Measured on Laya with 1,200 questions: three runs of 12 passes took 29.5
/// minutes, every run was still improving at its last pass, and all three
/// reached the same answer, so the extra runs bought nothing. The final
/// layers keep three runs because on a few hundred questions a run can
/// stall on the starting plateau; add-ons in every layer on a thousand or
/// more did not. A run that goes wrong is still caught: the gate judges it
/// on questions it never saw and writes nothing unless it wins.
pub const WHOLE_MODEL_TRAINING_SETTINGS: TrainingSettings = TrainingSettings {
    adam: HEAD_TRAINING_ADAM_SETTINGS,
    batch_question_count: 16,
    check_questions_per_batch: 8,
    check_weight: 1.0,
    maximum_epochs: 12,
    patience_epochs: 3,
    seeds: &[1],
};

/// How one run ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeedEnding {
    StoppedImproving,
    ReachedEpochLimit,
    /// A step's gradient held a non-finite number; the run kept its best
    /// weights from before that pass.
    NonFiniteStep {
        epoch: usize,
    },
}

/// Told how many optimizer steps are done, out of the most there can be.
pub trait TrainingProgress {
    fn report(&mut self, completed_batches: u64, total_batches: u64);
}

/// A Fisher-Yates shuffle of `0..length`, driven by a linear congruential
/// sequence seeded from the run's seed and the pass number, so every run is
/// reproducible.
#[must_use]
pub fn shuffled_order(length: usize, seed: u64, pass_number: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..length).collect();
    let mut sequence_state = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(pass_number.wrapping_mul(0xbf58_476d_1ce4_e5b9))
        | 1;
    for position in (1..length).rev() {
        sequence_state = sequence_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let swap_position = ((sequence_state >> 33) % (position as u64 + 1)) as usize;
        order.swap(position, swap_position);
    }
    order
}

#[cfg(test)]
#[path = "head_training_tests.rs"]
mod tests;

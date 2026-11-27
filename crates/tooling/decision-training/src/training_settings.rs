//! The settings and shared pieces of training, which runs on the GPU
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

use crate::anchored_adam::{AdamSettings, TRAINING_ADAM_SETTINGS};
use decision_contract::head_tuning_file::TunableHeadFamily;

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
pub const FINAL_LAYERS_TRAINING_SETTINGS: TrainingSettings = TrainingSettings {
    adam: TRAINING_ADAM_SETTINGS,
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
///
/// The check set weighs twice what it does for the final layers. Add-ons in
/// every layer can move answers the examples never asked about: on 200
/// coded questions Kev 0.8B learned the code completely at weight 1 but
/// changed 9 of the 120 check-set answers, and the gate refused it; at
/// weight 2 it learned the code just as completely and changed 4.
pub const WHOLE_MODEL_TRAINING_SETTINGS: TrainingSettings = TrainingSettings {
    adam: TRAINING_ADAM_SETTINGS,
    batch_question_count: 16,
    check_questions_per_batch: 8,
    check_weight: 2.0,
    maximum_epochs: 12,
    patience_epochs: 3,
    seeds: &[1],
};

/// Laya's whole-model settings: the shared ones with a faster, longer run.
///
/// The shared settings were measured on Kev, which learns the 200-question
/// coded control (`whole_model_positive_control`) within 12 passes at a
/// learning rate of 1e-4. Laya does not: at those settings its add-ons moved
/// (the largest `up` value after 45 steps was 4.1e-3) and its gradient is
/// the loss's own slope on MLX's CUDA backend (`whole_model_gradient`), yet
/// the run stopped after 8 passes with the unseen lines 60% right (issue
/// #2705). Chosen on the choosing lines, as a run chooses its own stopping
/// point, then judged on the unseen lines once (CUDA, 140 training, 30
/// choosing and 30 unseen lines):
///
/// | Learning rate | Passes (limit, patience) | Choosing loss | Choosing right | Unseen right |
/// |---|---|---|---|---|
/// | 1e-4 (shared) | 12, 3 | 0.658 | 63.3% | 60.0% |
/// | 3e-4 | 30, 12 | 0.301 | 90.0% | 83.3% |
/// | 5e-4 | 30, 8 | 0.400 | 83.3% | 86.7% |
/// | 1e-3 | 30, 8 | 0.028 | 100.0% | 100.0% |
///
/// At 1e-3 the run stops by itself after 18 passes, about four minutes. The
/// check set keeps its weight, which the gate tests: a larger step moves
/// answers nobody asked about more, and a run that does is refused.
pub const LAYA_WHOLE_MODEL_TRAINING_SETTINGS: TrainingSettings = TrainingSettings {
    adam: AdamSettings {
        learning_rate: 1e-3,
        ..TRAINING_ADAM_SETTINGS
    },
    maximum_epochs: 30,
    patience_epochs: 8,
    ..WHOLE_MODEL_TRAINING_SETTINGS
};

// A patience as long as the pass limit would never end a run early.
const _: () = assert!(
    LAYA_WHOLE_MODEL_TRAINING_SETTINGS.patience_epochs
        < LAYA_WHOLE_MODEL_TRAINING_SETTINGS.maximum_epochs
);

/// The whole-model settings of `family`.
#[must_use]
pub const fn whole_model_training_settings(family: TunableHeadFamily) -> TrainingSettings {
    match family {
        TunableHeadFamily::Laya => LAYA_WHOLE_MODEL_TRAINING_SETTINGS,
        TunableHeadFamily::Kev | TunableHeadFamily::Clef => WHOLE_MODEL_TRAINING_SETTINGS,
    }
}

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
#[path = "training_settings_tests.rs"]
mod tests;

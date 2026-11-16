//! A training job as it crosses the process boundaries: what the
//! command-line tool asks the worker to train on, and what comes back.
//!
//! The worker owns the model and the GPU, so it reads every example and
//! trains; the command-line tool owns the person's labels and the cheap,
//! pure decisions (which part each example is in, the never-worse gate, the
//! report, the new folder). The job therefore carries each question's role
//! and right answer in, and the served answers, the trained scores and the
//! trained tensors out; nothing about the split or the gate is decided in
//! the worker.

use serde::{Deserialize, Serialize};

use crate::head_inputs::FloatVector;
use crate::ordered_json::OrderedJsonValue;
use crate::outcome::QuestionOutcome;

/// What one question is for in a training.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "role", rename_all = "snake_case", deny_unknown_fields)]
pub enum QuestionRole {
    /// Its label moves weights.
    Training { correct_option: usize },
    /// Its label picks when to stop and which run is best.
    Choosing { correct_option: usize },
    /// Held back for the gate; the worker only scores it.
    Judging { correct_option: usize },
    /// A check-set question: its shipped answer is what training must keep.
    Check,
}

impl QuestionRole {
    /// The right option, for a labelled role.
    #[must_use]
    pub const fn correct_option(self) -> Option<usize> {
        match self {
            Self::Training { correct_option }
            | Self::Choosing { correct_option }
            | Self::Judging { correct_option } => Some(correct_option),
            Self::Check => None,
        }
    }
}

/// One example: a state, its questions, and each question's role in order.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingExample {
    pub state: OrderedJsonValue,
    /// The System One `questions` object, exactly as the caller wrote it.
    pub questions: OrderedJsonValue,
    pub question_roles: Vec<QuestionRole>,
}

/// How much of the model a training changes.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingScope {
    /// Only the final scoring layers, on what the rest of the model reads
    /// once: fast, and limited to what the model already notices.
    #[default]
    FinalLayers,
    /// Small add-on weights inside every layer plus the final scoring
    /// layers, so the whole model learns to read the examples differently.
    WholeModel,
}

/// Everything one training reads.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionTrainingJob {
    pub examples: Vec<TrainingExample>,
    pub scope: TrainingScope,
}

/// Which stage a training is in, for progress lines.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionTrainingStage {
    /// The model reads the examples: `completed` of `total` examples.
    Reading,
    /// The trainable copy is checked against the model's own scores before
    /// any weight moves: `completed` of `total` questions.
    Checking,
    /// Training: `completed` of at most `total` optimizer steps over all
    /// runs; a run that stops improving early jumps to the end of its steps.
    Training,
    /// The best run scores every question: `completed` of `total` questions.
    Scoring,
}

/// One trained tensor under its shipped name, in 32-bit floats.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: FloatVector,
}

/// How one run ended.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "ending", rename_all = "snake_case")]
pub enum RunEnding {
    StoppedImproving,
    ReachedEpochLimit,
    NonFiniteStep { epoch: usize },
}

/// One run's story, for the report.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunSummary {
    pub seed: u64,
    pub best_choosing_loss: f64,
    pub best_epoch: usize,
    pub epochs_run: usize,
    pub ending: RunEnding,
}

/// One question as the shipped model answered it and as the trained head
/// scores it; `trained_logits` is empty for a question training left out.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainedQuestion {
    pub served: QuestionOutcome,
    pub trained_logits: Vec<f32>,
}

/// What the worker returns: every example's questions in order, the best
/// run's tensors (the final layers, and for a whole-model training the
/// add-ons too), and the runs.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionTrainingOutcome {
    pub examples: Vec<Vec<TrainedQuestion>>,
    pub trained_tensors: Vec<TrainedTensor>,
    /// The add-on weights, each named `<target>.adapter_down` or
    /// `<target>.adapter_up`; empty unless the whole model was trained.
    pub adapter_tensors: Vec<TrainedTensor>,
    pub runs: Vec<RunSummary>,
    /// The seed of the run whose tensors were returned.
    pub best_seed: u64,
    /// The largest gap between the model's scores and the trainable copy's,
    /// with the shipped weights, before training.
    pub largest_parity_gap: f32,
    /// Questions left out because some of their options read the same.
    pub indistinguishable_question_count: usize,
}

#[cfg(test)]
#[path = "decision_training_job_tests.rs"]
mod tests;

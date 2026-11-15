//! A head-training job as it crosses the process boundaries: what the
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

/// Everything one training reads.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HeadTrainingJob {
    pub examples: Vec<TrainingExample>,
}

/// Which stage a training is in, for progress lines.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadTrainingStage {
    /// The model reads the examples: `completed` of `total` examples.
    Reading,
    /// Training passes: `completed` of at most `total` passes over all runs.
    Training,
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
/// run's tensors, and the runs.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HeadTrainingOutcome {
    pub examples: Vec<Vec<TrainedQuestion>>,
    pub trained_tensors: Vec<TrainedTensor>,
    pub runs: Vec<RunSummary>,
    /// The seed of the run whose tensors were returned.
    pub best_seed: u64,
    /// The largest gap between the model's scores and the GPU copy of its
    /// final layers, with the shipped weights, before training.
    pub largest_parity_gap: f32,
    /// Questions left out because some of their options read the same.
    pub indistinguishable_question_count: usize,
}

#[cfg(test)]
#[path = "head_training_job_tests.rs"]
mod tests;

//! Scoring every question of a training a little at a time.
//!
//! A final-layer training scores a thousand questions in milliseconds. A
//! whole-model training runs the model once per served pass, about ten
//! milliseconds each, so scoring every question of a few thousand examples
//! takes most of a minute. Done in one step, that would hold the worker
//! past the daemon's patience and leave a cancelled training running until
//! it finished. So scoring stops when its step's time is up and continues
//! on the next step.

use std::time::Instant;

use mlx_bridge::MlxArray;

use super::gpu_math::{GpuMath, GpuResult};
use super::gpu_training::{GpuTrainingSet, question_scores};

/// Questions scored between two looks at the clock: few enough that one
/// call is well under a step, enough to keep final-layer scoring batched.
const QUESTIONS_PER_SCORING_CALL: usize = 32;

/// Every question's scores under one set of parameters, gathered in row order.
pub(crate) struct SteppedScoring {
    question_count: usize,
    scores: Vec<Vec<f32>>,
}

impl SteppedScoring {
    pub(crate) fn new(question_count: usize) -> Self {
        Self {
            question_count,
            scores: Vec::with_capacity(question_count),
        }
    }

    pub(crate) fn question_count(&self) -> usize {
        self.question_count
    }

    pub(crate) fn scored_question_count(&self) -> usize {
        self.scores.len()
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.scores.len() >= self.question_count
    }

    /// Scores more questions until all are scored or `step_deadline` passes.
    pub(crate) fn advance(
        &mut self,
        math: GpuMath<'_>,
        training_set: &GpuTrainingSet,
        parameters: &[MlxArray],
        step_deadline: Instant,
    ) -> GpuResult<()> {
        while !self.is_finished() {
            let first_question = self.scores.len();
            let next_question =
                (first_question + QUESTIONS_PER_SCORING_CALL).min(self.question_count);
            let question_rows: Vec<usize> = (first_question..next_question).collect();
            self.scores.extend(question_scores(
                math,
                training_set,
                parameters,
                &question_rows,
            )?);
            if Instant::now() >= step_deadline {
                break;
            }
        }
        Ok(())
    }

    /// The scores, one list per question in row order.
    pub(crate) fn into_scores(self) -> Vec<Vec<f32>> {
        self.scores
    }
}

#[cfg(test)]
#[path = "stepped_scoring_tests.rs"]
mod tests;

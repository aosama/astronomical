//! Scoring a training's questions a little at a time.
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

/// Chosen questions' scores under one set of parameters, by row.
pub(crate) struct SteppedScoring {
    /// The rows to score, in the order they are scored.
    scored_rows: Vec<usize>,
    /// Every row's scores; empty for a row not scored (yet).
    scores: Vec<Vec<f32>>,
    scored_question_count: usize,
}

impl SteppedScoring {
    /// Scores `scored_rows` of a training set with `row_count` rows.
    pub(crate) fn new(scored_rows: Vec<usize>, row_count: usize) -> Self {
        Self {
            scored_rows,
            scores: vec![Vec::new(); row_count],
            scored_question_count: 0,
        }
    }

    pub(crate) fn question_count(&self) -> usize {
        self.scored_rows.len()
    }

    pub(crate) fn scored_question_count(&self) -> usize {
        self.scored_question_count
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.scored_question_count >= self.scored_rows.len()
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
            let first_question = self.scored_question_count;
            let next_question =
                (first_question + QUESTIONS_PER_SCORING_CALL).min(self.scored_rows.len());
            let question_rows = &self.scored_rows[first_question..next_question];
            let row_scores = question_scores(math, training_set, parameters, question_rows)?;
            for (question_row, scores) in question_rows.iter().zip(row_scores) {
                self.scores[*question_row] = scores;
            }
            self.scored_question_count = next_question;
            if Instant::now() >= step_deadline {
                break;
            }
        }
        Ok(())
    }

    /// Every row's scores in row order, empty for a row not scored.
    pub(crate) fn into_scores(self) -> Vec<Vec<f32>> {
        self.scores
    }
}

#[cfg(test)]
#[path = "stepped_scoring_tests.rs"]
mod tests;

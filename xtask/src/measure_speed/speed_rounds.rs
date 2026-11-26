//! Every timed reply of one measured case, by task, and the figures drawn from
//! them: each task's median, lowest and highest for prompt reading, the first
//! token and the reply. The catalog's whole-number rates are the rounded reply
//! medians of the same spreads, so the catalog and the ledger cannot disagree
//! by rounding.

use std::collections::BTreeMap;

use super::speed_prompts::SpeedTask;
use super::stream_timing::ReplyMeasurement;
use crate::release_speed::speed_ledger::{FigureSpread, TaskFigures};

/// Rounds a task needs before its figures are kept anywhere: one, since a
/// case's time budget may allow no more.
pub const REQUIRED_ROUND_COUNT: usize = 1;

#[derive(Debug, Default)]
pub struct ModelSpeedRounds {
    replies_by_task: BTreeMap<&'static str, Vec<ReplyMeasurement>>,
}

impl ModelSpeedRounds {
    /// Keeps one timed reply of a task. The caller keeps only replies long
    /// enough to have a reply rate; one without a rate adds no reply figure.
    pub fn record(&mut self, speed_task: SpeedTask, reply_measurement: ReplyMeasurement) {
        self.replies_by_task
            .entry(speed_task.catalog_field())
            .or_default()
            .push(reply_measurement);
    }

    /// A task's figures once it has the required rounds; `None` before.
    #[must_use]
    pub fn task_figures(&self, speed_task: SpeedTask) -> Option<TaskFigures> {
        let task_replies = self.replies_by_task.get(speed_task.catalog_field())?;
        if task_replies.len() < REQUIRED_ROUND_COUNT {
            return None;
        }
        let mut prefill_rates = Vec::new();
        let mut first_token_seconds = Vec::new();
        let mut reply_rates = Vec::new();
        let mut prompt_token_counts = Vec::new();
        for task_reply in task_replies {
            prefill_rates.push(task_reply.prefill_tokens_per_second());
            first_token_seconds.push(task_reply.seconds_to_first_token);
            reply_rates.extend(task_reply.reply_tokens_per_second);
            prompt_token_counts.push(task_reply.prompt_tokens);
        }
        prompt_token_counts.sort_unstable();
        Some(TaskFigures {
            round_count: task_replies.len(),
            prompt_tokens: prompt_token_counts[prompt_token_counts.len() / 2],
            prefill_tokens_per_second: FigureSpread::of_rounds(&prefill_rates)?,
            seconds_to_first_token: FigureSpread::of_rounds(&first_token_seconds)?,
            reply_tokens_per_second: FigureSpread::of_rounds(&reply_rates)?,
        })
    }

    /// Whole-token reply medians for every task, or `None` until each task
    /// has the required rounds; a partial set is never written.
    #[must_use]
    pub fn medians(&self) -> Option<BTreeMap<&'static str, u64>> {
        SpeedTask::ALL
            .iter()
            .map(|speed_task| self.task_median(*speed_task))
            .collect()
    }

    fn task_median(&self, speed_task: SpeedTask) -> Option<(&'static str, u64)> {
        let task_figures = self.task_figures(speed_task)?;
        Some((
            speed_task.catalog_field(),
            whole_rate(task_figures.reply_tokens_per_second.median),
        ))
    }
}

/// The catalog's whole tokens a second for a median.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a rounded positive rate of tokens a second"
)]
#[must_use]
pub fn whole_rate(median_rate: f64) -> u64 {
    median_rate.round() as u64
}

#[cfg(test)]
#[path = "speed_rounds_tests.rs"]
mod speed_rounds_tests;

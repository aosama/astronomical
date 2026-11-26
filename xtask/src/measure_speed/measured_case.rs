//! One measured case (a model under one memory setting) and the ledger
//! measurements it becomes: one a task, each carrying the conditions the
//! engine served under.

use super::served_conditions::ServedConditions;
use super::speed_prompts::SpeedTask;
use super::speed_rounds::ModelSpeedRounds;
use crate::release_speed::speed_ledger::SpeedMeasurement;

pub struct MeasuredCase {
    pub model_id: String,
    pub served_conditions: ServedConditions,
    pub speed_rounds: ModelSpeedRounds,
    pub temperature_override: Option<f64>,
    pub reply_tokens_requested: u32,
    pub long_prompt_tokens: u32,
}

impl MeasuredCase {
    #[must_use]
    pub fn task_temperature(&self, speed_task: SpeedTask) -> f64 {
        self.temperature_override
            .unwrap_or_else(|| speed_task.default_temperature())
    }

    /// A measurement for every task that reached its required rounds; a task
    /// that did not is left out, never written from fewer rounds.
    #[must_use]
    pub fn ledger_measurements(&self, model_revision: Option<&str>) -> Vec<SpeedMeasurement> {
        let mut ledger_measurements = Vec::new();
        for speed_task in SpeedTask::ALL {
            let Some(task_figures) = self.speed_rounds.task_figures(speed_task) else {
                continue;
            };
            let served_conditions = &self.served_conditions;
            ledger_measurements.push(SpeedMeasurement {
                model_id: self.model_id.clone(),
                model_revision: model_revision.map(str::to_owned),
                maximum_mlx_memory_gb: served_conditions.maximum_mlx_memory_gb,
                mlx_memory_limit_bytes: served_conditions.mlx_memory_limit_bytes,
                weights_placement: served_conditions.weights_placement,
                drafter: served_conditions.drafter,
                multi_token_prediction_draft_tokens: served_conditions
                    .multi_token_prediction_draft_tokens,
                task: speed_task.catalog_field().to_owned(),
                temperature: self.task_temperature(speed_task),
                reply_tokens_requested: self.reply_tokens_requested,
                figures: task_figures,
            });
        }
        ledger_measurements
    }
}

#[cfg(test)]
#[path = "measured_case_tests.rs"]
mod measured_case_tests;

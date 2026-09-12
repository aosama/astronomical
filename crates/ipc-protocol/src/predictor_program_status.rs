//! Bounded predictor-program snapshot for worker status and the menu.
//!
//! Percentages are stored as tenths of a percent so the IPC type stays
//! integer `Eq`/`Copy`. HTTP status converts tenths to one decimal.

use serde::{Deserialize, Serialize};

/// Where the predictor is running. The first path is CPU; Neural Engine is
/// reserved for a later bake-off and is never implied by this snapshot.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictorRuntime {
    #[default]
    Cpu,
    NeuralEngine,
}

/// Session-level predictor evidence published after a generation finalizes.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PredictorProgramStatus {
    pub runtime: PredictorRuntime,
    pub training_active: bool,
    pub top_k_accuracy_tenths: u16,
    pub pages_avoided_tenths: u16,
}

impl PredictorProgramStatus {
    /// Builds a CPU snapshot from the trainer counters. Zero denominators
    /// publish `0.0` rather than inventing a rate.
    #[must_use]
    pub fn from_cpu_counts(
        training_active: bool,
        top_k_hit_count: u64,
        evaluated_expert_count: u64,
        pages_avoided_hit_count: u64,
        pages_avoided_opportunity_count: u64,
    ) -> Self {
        Self {
            runtime: PredictorRuntime::Cpu,
            training_active,
            top_k_accuracy_tenths: one_decimal_percent_tenths(
                top_k_hit_count,
                evaluated_expert_count,
            ),
            pages_avoided_tenths: one_decimal_percent_tenths(
                pages_avoided_hit_count,
                pages_avoided_opportunity_count,
            ),
        }
    }

    #[must_use]
    pub fn top_k_accuracy_percent(self) -> f64 {
        f64::from(self.top_k_accuracy_tenths) / 10.0
    }

    #[must_use]
    pub fn pages_avoided_percent(self) -> f64 {
        f64::from(self.pages_avoided_tenths) / 10.0
    }
}

/// Rounds `numerator / denominator * 100` to one decimal, stored as tenths.
#[must_use]
pub fn one_decimal_percent_tenths(numerator: u64, denominator: u64) -> u16 {
    if denominator == 0 {
        return 0;
    }
    let tenths = ((numerator as f64 / denominator as f64) * 1_000.0).round();
    tenths.clamp(0.0, 1_000.0) as u16
}

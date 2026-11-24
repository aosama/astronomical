//! The three figures an evaluation reports, computed the way OptiQ's
//! `evaluate_kl` computes them with NumPy.

use serde::{Deserialize, Serialize};

#[cfg(test)]
#[path = "kl_statistics_tests.rs"]
mod kl_statistics_tests;

/// KL(reference ‖ candidate) over the evaluation prompts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KlEvaluationSummary {
    /// The mean over prompts of each prompt's mean per-token divergence: the
    /// headline figure. A long prompt weighs the same as a short one.
    pub mean_kl_divergence: f64,
    /// The median over every token of every prompt, robust to the few
    /// positions where a small logit shift flips a near-tie.
    pub median_kl_divergence: f64,
    /// The 95th percentile over every token: the tail, where a quant that
    /// is fine on average goes wrong on particular content.
    pub ninety_fifth_percentile_kl_divergence: f64,
    pub prompt_count: usize,
    pub token_count: usize,
}

impl KlEvaluationSummary {
    /// Summarizes per-token divergences grouped by prompt. `None` when no
    /// token was measured.
    #[must_use]
    pub fn from_per_prompt_divergences(per_prompt_divergences: &[Vec<f64>]) -> Option<Self> {
        let mut per_prompt_means = Vec::new();
        let mut every_token_divergence = Vec::new();
        for prompt_divergences in per_prompt_divergences {
            if prompt_divergences.is_empty() {
                continue;
            }
            per_prompt_means.push(mean_of(prompt_divergences));
            every_token_divergence.extend_from_slice(prompt_divergences);
        }
        if every_token_divergence.is_empty() {
            return None;
        }
        every_token_divergence.sort_by(f64::total_cmp);
        Some(Self {
            mean_kl_divergence: mean_of(&per_prompt_means),
            median_kl_divergence: percentile_of_sorted(&every_token_divergence, 50.0),
            ninety_fifth_percentile_kl_divergence: percentile_of_sorted(
                &every_token_divergence,
                95.0,
            ),
            prompt_count: per_prompt_means.len(),
            token_count: every_token_divergence.len(),
        })
    }
}

fn mean_of(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// NumPy's default percentile: linear interpolation between the two order
/// statistics around rank `(count - 1) * percentile / 100`.
fn percentile_of_sorted(sorted_values: &[f64], percentile: f64) -> f64 {
    let fractional_rank = (sorted_values.len() - 1) as f64 * percentile / 100.0;
    let lower_rank = fractional_rank.floor() as usize;
    let upper_rank = fractional_rank.ceil() as usize;
    let interpolation_weight = fractional_rank - lower_rank as f64;
    sorted_values[lower_rank]
        + (sorted_values[upper_rank] - sorted_values[lower_rank]) * interpolation_weight
}

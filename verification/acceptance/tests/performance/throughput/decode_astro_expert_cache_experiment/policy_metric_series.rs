//! One metric's value in every paired trial, under each decode
//! `astro_expert_cache` policy, ready for a paired confidence interval.

use super::{
    PairedConfidenceInterval95, PairedDecodeAstroExpertCacheTrial,
    paired_relative_improvement_confidence_interval_95,
};

/// One metric's value in every paired trial: the baseline policy's and the
/// candidate policy's, in trial order.
pub(super) struct PolicyMetricSeries {
    baseline_values: Vec<f64>,
    candidate_values: Vec<f64>,
}

impl PolicyMetricSeries {
    pub(super) fn relative_improvement_confidence_interval_95(
        &self,
        lower_is_better: bool,
    ) -> Option<PairedConfidenceInterval95> {
        paired_relative_improvement_confidence_interval_95(
            &self.baseline_values,
            &self.candidate_values,
            lower_is_better,
        )
    }
}

pub(super) fn decode_forward_p95_nanoseconds_series(
    paired_trials: &[PairedDecodeAstroExpertCacheTrial],
) -> PolicyMetricSeries {
    let baseline_decode_p95_nanoseconds = paired_trials
        .iter()
        .map(|paired_trial| {
            paired_trial
                .baseline
                .decode_astro_expert_cache_diagnostic_evidence
                .as_ref()
                .and_then(|diagnostic_evidence| {
                    diagnostic_evidence.decode_token_forward_latency_p95_nanoseconds
                })
                .expect("baseline decode-forward p95 should be present") as f64
        })
        .collect::<Vec<_>>();
    let candidate_decode_p95_nanoseconds = paired_trials
        .iter()
        .map(|paired_trial| {
            paired_trial
                .candidate
                .decode_astro_expert_cache_diagnostic_evidence
                .as_ref()
                .and_then(|diagnostic_evidence| {
                    diagnostic_evidence.decode_token_forward_latency_p95_nanoseconds
                })
                .expect("candidate decode-forward p95 should be present") as f64
        })
        .collect::<Vec<_>>();
    PolicyMetricSeries {
        baseline_values: baseline_decode_p95_nanoseconds,
        candidate_values: candidate_decode_p95_nanoseconds,
    }
}

pub(super) fn decode_source_bytes_per_output_token_series(
    paired_trials: &[PairedDecodeAstroExpertCacheTrial],
) -> PolicyMetricSeries {
    let baseline_decode_source_bytes_per_output_token = paired_trials
        .iter()
        .map(|paired_trial| {
            let diagnostic_evidence = paired_trial
                .baseline
                .decode_astro_expert_cache_diagnostic_evidence
                .as_ref()
                .expect("baseline decode attribution should be present");
            diagnostic_evidence.decode_expert_source_payload_bytes as f64
                / f64::from(paired_trial.baseline.total_output_tokens)
        })
        .collect::<Vec<_>>();
    let candidate_decode_source_bytes_per_output_token = paired_trials
        .iter()
        .map(|paired_trial| {
            let diagnostic_evidence = paired_trial
                .candidate
                .decode_astro_expert_cache_diagnostic_evidence
                .as_ref()
                .expect("candidate decode attribution should be present");
            diagnostic_evidence.decode_expert_source_payload_bytes as f64
                / f64::from(paired_trial.candidate.total_output_tokens)
        })
        .collect::<Vec<_>>();
    PolicyMetricSeries {
        baseline_values: baseline_decode_source_bytes_per_output_token,
        candidate_values: candidate_decode_source_bytes_per_output_token,
    }
}

pub(super) fn decode_tokens_per_second_series(
    paired_trials: &[PairedDecodeAstroExpertCacheTrial],
) -> PolicyMetricSeries {
    let baseline_decode_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.baseline.decode_tokens_per_second)
        .collect::<Vec<_>>();
    let candidate_decode_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.candidate.decode_tokens_per_second)
        .collect::<Vec<_>>();
    PolicyMetricSeries {
        baseline_values: baseline_decode_tokens_per_second,
        candidate_values: candidate_decode_tokens_per_second,
    }
}

pub(super) fn prefill_tokens_per_second_series(
    paired_trials: &[PairedDecodeAstroExpertCacheTrial],
) -> PolicyMetricSeries {
    let baseline_prefill_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.baseline.prefill_tokens_per_second)
        .collect::<Vec<_>>();
    let candidate_prefill_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.candidate.prefill_tokens_per_second)
        .collect::<Vec<_>>();
    PolicyMetricSeries {
        baseline_values: baseline_prefill_tokens_per_second,
        candidate_values: candidate_prefill_tokens_per_second,
    }
}

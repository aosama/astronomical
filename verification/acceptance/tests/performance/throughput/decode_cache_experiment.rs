use astronomical_ipc_protocol::WorkerDecodeExpertCachePolicy;
use std::time::Instant;

use crate::performance_throughput::completion::ThroughputMeasurement;
use crate::performance_throughput::completion::{
    DecodeCacheDiagnosticEvidence, PerformanceOperationEvidence,
};
use crate::performance_throughput::support::{self as throughput_support, ThroughputJourney};

pub(crate) const PAIRED_TRIAL_COUNT: usize = 3;

/// The measured token-count window one paired lane accepts. The standard
/// window mirrors the full throughput journey's shape contract; the
/// single-chunk lane narrows it to the shape that fits the journey budget.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasuredShapeWindow {
    pub(crate) minimum_input_tokens: u32,
    pub(crate) maximum_input_tokens: u32,
    pub(crate) minimum_output_tokens: u16,
    pub(crate) maximum_output_tokens: u16,
}

impl MeasuredShapeWindow {
    pub(crate) fn assert_measurement_fits(&self, measured_throughput: &ThroughputMeasurement) {
        assert!(
            (self.minimum_input_tokens..=self.maximum_input_tokens)
                .contains(&measured_throughput.total_input_tokens),
            "the measured Romeo and Juliet prompt should contain {} to {} input tokens, got {}",
            self.minimum_input_tokens,
            self.maximum_input_tokens,
            measured_throughput.total_input_tokens,
        );
        assert!(
            (self.minimum_output_tokens..=self.maximum_output_tokens)
                .contains(&measured_throughput.total_output_tokens),
            "the measured completion should generate {} to {} output tokens, got {}",
            self.minimum_output_tokens,
            self.maximum_output_tokens,
            measured_throughput.total_output_tokens,
        );
        assert_eq!(
            measured_throughput.cached_tokens, 0,
            "the measured completion must not reuse the persistent prompt cache",
        );
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PairedDecodeCacheTrial {
    pub(crate) baseline: ThroughputMeasurement,
    pub(crate) candidate: ThroughputMeasurement,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PairedConfidenceInterval95 {
    pub(crate) mean_improvement_percent: f64,
    pub(crate) lower_bound_percent: f64,
    pub(crate) upper_bound_percent: f64,
}

pub(crate) fn run_paired_decode_cache_trials(
    model_id: &'static str,
    journey_template: &ThroughputJourney,
    paired_trial_count: usize,
    measured_shape_window: &MeasuredShapeWindow,
) -> Vec<PairedDecodeCacheTrial> {
    assert!(
        paired_trial_count >= 2,
        "paired confidence intervals need at least two baseline-candidate trials"
    );
    let mut paired_trials = Vec::with_capacity(paired_trial_count);
    let total_policy_run_count = paired_trial_count.saturating_mul(2);
    let experiment_started_at = Instant::now();
    let mut completed_policy_run_count = 0_usize;
    for pair_index in 0..paired_trial_count {
        let mut baseline_measurement = None;
        let mut candidate_measurement = None;
        for decode_expert_cache_policy in policy_order_for_trial(pair_index) {
            let current_progress_percent = completed_policy_run_count
                .saturating_mul(100)
                .checked_div(total_policy_run_count)
                .unwrap_or(0);
            eprintln!(
                "[decode-cache-experiment] progress_percent={current_progress_percent} completed_policy_runs={completed_policy_run_count} total_policy_runs={total_policy_run_count} pair_index={} policy={decode_expert_cache_policy:?} status=starting",
                pair_index + 1
            );
            let mut journey = journey_template.clone();
            journey.decode_expert_cache_policy = decode_expert_cache_policy;
            let measured_throughput =
                throughput_support::run_journey_with_timeout(model_id, journey.clone());
            completed_policy_run_count += 1;
            let completed_progress_percent = completed_policy_run_count
                .saturating_mul(100)
                .checked_div(total_policy_run_count)
                .unwrap_or(100)
                .min(100);
            let average_policy_run_seconds =
                experiment_started_at.elapsed().as_secs_f64() / completed_policy_run_count as f64;
            let estimated_remaining_seconds = average_policy_run_seconds
                * total_policy_run_count.saturating_sub(completed_policy_run_count) as f64;
            eprintln!(
                "[decode-cache-experiment] progress_percent={completed_progress_percent} completed_policy_runs={completed_policy_run_count} total_policy_runs={total_policy_run_count} eta_seconds={estimated_remaining_seconds:.0} pair_index={} policy={decode_expert_cache_policy:?} status=completed",
                pair_index + 1
            );
            measured_shape_window.assert_measurement_fits(&measured_throughput);
            assert_eq!(
                measured_throughput.decode_expert_cache_policy, decode_expert_cache_policy,
                "the worker measurement must retain the selected cache policy"
            );
            assert!(
                measured_throughput.measured_output_text.is_some(),
                "the completed measurement must preserve streamed output for parity checks"
            );
            assert_peak_memory_within_limit(&journey, &measured_throughput);
            if journey.attribution_enabled {
                let diagnostic_evidence = measured_throughput
                    .decode_cache_diagnostic_evidence
                    .as_ref()
                    .expect("the attributed run must produce measured decode-cache evidence");
                assert!(
                    diagnostic_evidence
                        .decode_token_forward_latency_p95_nanoseconds
                        .is_some(),
                    "the generation report must include decode-forward p95"
                );
                assert!(
                    diagnostic_evidence
                        .decode_token_forward_latency_sample_count
                        .is_some_and(|sample_count| sample_count > 0),
                    "the generation report must include decode-forward histogram samples"
                );
                assert_eq!(
                    diagnostic_evidence.decode_expert_source_payload_bytes,
                    diagnostic_evidence.decode_expert_source_summary_payload_bytes,
                    "the decode source counter must agree with its phase summaries"
                );
            }
            match decode_expert_cache_policy {
                WorkerDecodeExpertCachePolicy::Baseline => {
                    baseline_measurement = Some(measured_throughput);
                }
                WorkerDecodeExpertCachePolicy::Candidate => {
                    candidate_measurement = Some(measured_throughput);
                }
            }
        }
        let baseline = baseline_measurement
            .expect("the alternating pair must include one baseline measurement");
        let candidate = candidate_measurement
            .expect("the alternating pair must include one candidate measurement");
        assert_eq!(
            baseline.measured_output_text, candidate.measured_output_text,
            "baseline and candidate must generate identical Romeo and Juliet output"
        );
        if journey_template.attribution_enabled {
            assert_eq!(
                baseline
                    .decode_cache_diagnostic_evidence
                    .as_ref()
                    .expect("the baseline attribution must be present")
                    .prefill_expert_source_payload_bytes,
                candidate
                    .decode_cache_diagnostic_evidence
                    .as_ref()
                    .expect("the candidate attribution must be present")
                    .prefill_expert_source_payload_bytes,
                "prefill source bytes must remain identical across cache policies"
            );
        }
        paired_trials.push(PairedDecodeCacheTrial {
            baseline,
            candidate,
        });
    }
    paired_trials
}

pub(crate) fn paired_relative_improvement_confidence_interval_95(
    baseline_values: &[f64],
    candidate_values: &[f64],
    lower_is_better: bool,
) -> Option<PairedConfidenceInterval95> {
    if baseline_values.len() < 2 || baseline_values.len() != candidate_values.len() {
        return None;
    }
    let mut paired_improvement_percentages = Vec::with_capacity(baseline_values.len());
    for (baseline_value, candidate_value) in baseline_values.iter().zip(candidate_values) {
        if !baseline_value.is_finite()
            || !candidate_value.is_finite()
            || *baseline_value < 0.0
            || *candidate_value < 0.0
        {
            return None;
        }
        let paired_improvement_percent = if *baseline_value == 0.0 {
            if *candidate_value == 0.0 {
                0.0
            } else {
                return None;
            }
        } else if lower_is_better {
            (baseline_value - candidate_value) / baseline_value * 100.0
        } else {
            (candidate_value - baseline_value) / baseline_value * 100.0
        };
        paired_improvement_percentages.push(paired_improvement_percent);
    }
    paired_mean_confidence_interval_95(&paired_improvement_percentages)
}

fn paired_mean_confidence_interval_95(
    paired_improvement_percentages: &[f64],
) -> Option<PairedConfidenceInterval95> {
    if paired_improvement_percentages.len() < 2 {
        return None;
    }
    let paired_trial_count = paired_improvement_percentages.len() as f64;
    let mean_improvement_percent =
        paired_improvement_percentages.iter().sum::<f64>() / paired_trial_count;
    let sample_variance = paired_improvement_percentages
        .iter()
        .map(|paired_improvement_percent| {
            (paired_improvement_percent - mean_improvement_percent).powi(2)
        })
        .sum::<f64>()
        / (paired_trial_count - 1.0);
    let standard_error = (sample_variance / paired_trial_count).sqrt();
    let degrees_of_freedom = paired_improvement_percentages.len() - 1;
    let confidence_multiplier = student_t_critical_value_975(degrees_of_freedom);
    let confidence_margin = confidence_multiplier * standard_error;
    Some(PairedConfidenceInterval95 {
        mean_improvement_percent,
        lower_bound_percent: mean_improvement_percent - confidence_margin,
        upper_bound_percent: mean_improvement_percent + confidence_margin,
    })
}

fn student_t_critical_value_975(degrees_of_freedom: usize) -> f64 {
    const CRITICAL_VALUES_BY_DEGREES_OF_FREEDOM: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    degrees_of_freedom
        .checked_sub(1)
        .and_then(|degree_index| {
            CRITICAL_VALUES_BY_DEGREES_OF_FREEDOM
                .get(degree_index)
                .copied()
        })
        .unwrap_or(1.96)
}

fn policy_order_for_trial(pair_index: usize) -> [WorkerDecodeExpertCachePolicy; 2] {
    if pair_index.is_multiple_of(2) {
        [
            WorkerDecodeExpertCachePolicy::Baseline,
            WorkerDecodeExpertCachePolicy::Candidate,
        ]
    } else {
        [
            WorkerDecodeExpertCachePolicy::Candidate,
            WorkerDecodeExpertCachePolicy::Baseline,
        ]
    }
}

fn assert_peak_memory_within_limit(
    journey: &ThroughputJourney,
    measured_throughput: &ThroughputMeasurement,
) {
    let diagnostic_peak_memory_bytes = measured_throughput
        .decode_cache_diagnostic_evidence
        .as_ref()
        .and_then(|diagnostic_evidence| diagnostic_evidence.mlx_peak_memory_bytes);
    let observed_peak_memory_bytes = [
        measured_throughput.mlx_peak_memory_bytes,
        diagnostic_peak_memory_bytes,
    ]
    .into_iter()
    .flatten()
    .max()
    .expect("the worker must report a measured peak memory value");
    let maximum_mlx_memory_bytes = journey
        .maximum_mlx_memory_bytes
        .unwrap_or(throughput_support::MAXIMUM_MODEL_MEMORY_BYTES);
    assert!(
        observed_peak_memory_bytes <= maximum_mlx_memory_bytes,
        "the {observed_peak_memory_bytes}-byte observed peak exceeded the {maximum_mlx_memory_bytes}-byte configured memory ceiling"
    );
}

#[cfg(test)]
#[path = "decode_cache_experiment_tests.rs"]
mod decode_cache_experiment_tests;

pub(super) fn report_attributed_policy_comparison(paired_trials: &[PairedDecodeCacheTrial]) {
    let baseline_decode_p95_nanoseconds = paired_trials
        .iter()
        .map(|paired_trial| {
            paired_trial
                .baseline
                .decode_cache_diagnostic_evidence
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
                .decode_cache_diagnostic_evidence
                .as_ref()
                .and_then(|diagnostic_evidence| {
                    diagnostic_evidence.decode_token_forward_latency_p95_nanoseconds
                })
                .expect("candidate decode-forward p95 should be present") as f64
        })
        .collect::<Vec<_>>();
    let baseline_decode_source_bytes_per_output_token = paired_trials
        .iter()
        .map(|paired_trial| {
            let diagnostic_evidence = paired_trial
                .baseline
                .decode_cache_diagnostic_evidence
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
                .decode_cache_diagnostic_evidence
                .as_ref()
                .expect("candidate decode attribution should be present");
            diagnostic_evidence.decode_expert_source_payload_bytes as f64
                / f64::from(paired_trial.candidate.total_output_tokens)
        })
        .collect::<Vec<_>>();
    let baseline_decode_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.baseline.decode_tokens_per_second)
        .collect::<Vec<_>>();
    let candidate_decode_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.candidate.decode_tokens_per_second)
        .collect::<Vec<_>>();
    let baseline_prefill_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.baseline.prefill_tokens_per_second)
        .collect::<Vec<_>>();
    let candidate_prefill_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.candidate.prefill_tokens_per_second)
        .collect::<Vec<_>>();

    let decode_p95_confidence_interval = paired_relative_improvement_confidence_interval_95(
        &baseline_decode_p95_nanoseconds,
        &candidate_decode_p95_nanoseconds,
        true,
    );
    let source_bytes_confidence_interval = paired_relative_improvement_confidence_interval_95(
        &baseline_decode_source_bytes_per_output_token,
        &candidate_decode_source_bytes_per_output_token,
        true,
    );
    let decode_throughput_confidence_interval = paired_relative_improvement_confidence_interval_95(
        &baseline_decode_tokens_per_second,
        &candidate_decode_tokens_per_second,
        false,
    );
    let prefill_throughput_confidence_interval = paired_relative_improvement_confidence_interval_95(
        &baseline_prefill_tokens_per_second,
        &candidate_prefill_tokens_per_second,
        false,
    );
    let has_primary_improvement = decode_p95_confidence_interval
        .is_some_and(|confidence_interval| confidence_interval.lower_bound_percent >= 5.0)
        || source_bytes_confidence_interval
            .is_some_and(|confidence_interval| confidence_interval.lower_bound_percent >= 10.0);
    let has_no_material_regression = [
        decode_p95_confidence_interval,
        source_bytes_confidence_interval,
        decode_throughput_confidence_interval,
        prefill_throughput_confidence_interval,
    ]
    .into_iter()
    .all(|confidence_interval| {
        confidence_interval
            .is_some_and(|confidence_interval| confidence_interval.lower_bound_percent >= -2.0)
    });
    eprintln!(
        "[decode-cache-experiment] trials={} diagnostic=true decode_p95_improvement={} source_bytes_per_output_token_improvement={} decode_throughput_improvement={} prefill_throughput_improvement={} baseline_evidence={} candidate_evidence={} decision={}",
        paired_trials.len(),
        format_confidence_interval(decode_p95_confidence_interval),
        format_confidence_interval(source_bytes_confidence_interval),
        format_confidence_interval(decode_throughput_confidence_interval),
        format_confidence_interval(prefill_throughput_confidence_interval),
        aggregate_cache_evidence(paired_trials, WorkerDecodeExpertCachePolicy::Baseline),
        aggregate_cache_evidence(paired_trials, WorkerDecodeExpertCachePolicy::Candidate),
        if has_primary_improvement && has_no_material_regression {
            "candidate_win"
        } else {
            "inconclusive_or_regression"
        }
    );
}

pub(super) fn report_streaming_decode_stage_evidence(
    measured_throughput: &ThroughputMeasurement,
    diagnostic_evidence: &DecodeCacheDiagnosticEvidence,
) {
    eprintln!(
        "[decode-streaming-stage-evidence] model_id={} input_tokens={} output_tokens={} decode_tokens_per_second={:.2} decode_forward_p95_milliseconds={:?} decode_forward_sample_count={:?} decode_advance_span={} route_host_materialization={} route_array_construction={} route_array_evaluation={} route_host_copy={} streaming_page_setup={} router_graph={} paged_moe_graph={} request_forward_completion_wait={} generated_token_synchronization_wait={} positional_read_call_count={} positional_read_bytes={} positional_read_callback_elapsed_milliseconds={:.3} positional_read_maximum_elapsed_milliseconds={:.3} positional_read_maximum_concurrent_count={} process_physical_disk_read_bytes={:?}",
        measured_throughput.model_id,
        measured_throughput.total_input_tokens,
        measured_throughput.total_output_tokens,
        measured_throughput.decode_tokens_per_second,
        diagnostic_evidence
            .decode_token_forward_latency_p95_nanoseconds
            .map(nanoseconds_to_milliseconds),
        diagnostic_evidence.decode_token_forward_latency_sample_count,
        format_operation_timing_evidence(diagnostic_evidence.decode_advance_span.as_ref()),
        format_operation_timing_evidence(
            diagnostic_evidence
                .decode_expert_route_host_materialization
                .as_ref()
        ),
        format_operation_timing_evidence(
            diagnostic_evidence
                .decode_expert_route_contiguous_array_construction
                .as_ref()
        ),
        format_operation_timing_evidence(
            diagnostic_evidence
                .decode_expert_route_array_evaluation
                .as_ref()
        ),
        format_operation_timing_evidence(
            diagnostic_evidence.decode_expert_route_host_copy.as_ref()
        ),
        format_operation_timing_evidence(
            diagnostic_evidence
                .rust_expert_streaming_layer_preparation
                .as_ref()
        ),
        format_operation_timing_evidence(
            diagnostic_evidence.paged_router_graph_construction.as_ref()
        ),
        format_operation_timing_evidence(diagnostic_evidence.paged_moe_graph_construction.as_ref()),
        format_operation_timing_evidence(diagnostic_evidence.forward_completion_wait.as_ref()),
        format_operation_timing_evidence(
            diagnostic_evidence
                .generated_token_item_synchronization_wait
                .as_ref()
        ),
        diagnostic_evidence.positional_file_read_call_count,
        diagnostic_evidence.positional_file_read_byte_count,
        nanoseconds_to_milliseconds(diagnostic_evidence.positional_file_read_elapsed_nanoseconds,),
        nanoseconds_to_milliseconds(
            diagnostic_evidence.positional_file_read_maximum_elapsed_nanoseconds,
        ),
        diagnostic_evidence.positional_file_read_maximum_concurrent_count,
        diagnostic_evidence.process_physical_disk_read_bytes,
    );
}

pub(super) fn format_operation_timing_evidence(
    operation_evidence: Option<&PerformanceOperationEvidence>,
) -> String {
    operation_evidence.map_or_else(
        || "not_recorded".to_owned(),
        |operation_evidence| {
            let mean_elapsed_nanoseconds = operation_evidence
                .total_elapsed_nanoseconds
                .checked_div(operation_evidence.occurrence_count.max(1))
                .unwrap_or(0);
            format!(
                "count={},total_milliseconds={:.3},mean_milliseconds={:.3},minimum_milliseconds={:.3},maximum_milliseconds={:.3}",
                operation_evidence.occurrence_count,
                nanoseconds_to_milliseconds(operation_evidence.total_elapsed_nanoseconds),
                nanoseconds_to_milliseconds(mean_elapsed_nanoseconds),
                nanoseconds_to_milliseconds(operation_evidence.minimum_elapsed_nanoseconds),
                nanoseconds_to_milliseconds(operation_evidence.maximum_elapsed_nanoseconds),
            )
        },
    )
}

pub(super) fn nanoseconds_to_milliseconds(elapsed_nanoseconds: u64) -> f64 {
    elapsed_nanoseconds as f64 / 1_000_000.0
}

pub(super) fn aggregate_cache_evidence(
    paired_trials: &[PairedDecodeCacheTrial],
    decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
) -> String {
    let mut retained_route_assignment_hit_count = 0_u64;
    let mut retained_route_assignment_miss_count = 0_u64;
    let mut hot_expert_route_fully_covered_count = 0_u64;
    let mut hot_expert_route_partially_covered_count = 0_u64;
    let mut hot_expert_route_fully_missed_count = 0_u64;
    let mut decode_expert_cache_slot_eviction_count = 0_u64;
    let mut positional_file_read_byte_count = 0_u64;
    let mut process_physical_disk_read_bytes = Some(0_u64);
    let mut peak_mlx_memory_bytes = None;
    for paired_trial in paired_trials {
        let measured_throughput = match decode_expert_cache_policy {
            WorkerDecodeExpertCachePolicy::Baseline => &paired_trial.baseline,
            WorkerDecodeExpertCachePolicy::Candidate => &paired_trial.candidate,
        };
        let diagnostic_evidence = measured_throughput
            .decode_cache_diagnostic_evidence
            .as_ref()
            .expect("each attribution trial must retain its diagnostics");
        retained_route_assignment_hit_count = retained_route_assignment_hit_count
            .saturating_add(diagnostic_evidence.retained_route_assignment_hit_count);
        retained_route_assignment_miss_count = retained_route_assignment_miss_count
            .saturating_add(diagnostic_evidence.retained_route_assignment_miss_count);
        hot_expert_route_fully_covered_count = hot_expert_route_fully_covered_count
            .saturating_add(diagnostic_evidence.hot_expert_route_fully_covered_count);
        hot_expert_route_partially_covered_count = hot_expert_route_partially_covered_count
            .saturating_add(diagnostic_evidence.hot_expert_route_partially_covered_count);
        hot_expert_route_fully_missed_count = hot_expert_route_fully_missed_count
            .saturating_add(diagnostic_evidence.hot_expert_route_fully_missed_count);
        decode_expert_cache_slot_eviction_count = decode_expert_cache_slot_eviction_count
            .saturating_add(diagnostic_evidence.decode_expert_cache_slot_eviction_count);
        positional_file_read_byte_count = positional_file_read_byte_count
            .saturating_add(diagnostic_evidence.positional_file_read_byte_count);
        process_physical_disk_read_bytes =
            process_physical_disk_read_bytes.and_then(|total_physical_disk_read_bytes| {
                diagnostic_evidence.process_physical_disk_read_bytes.map(
                    |physical_disk_read_bytes| {
                        total_physical_disk_read_bytes.saturating_add(physical_disk_read_bytes)
                    },
                )
            });
        let trial_peak_memory_bytes = measured_throughput
            .mlx_peak_memory_bytes
            .or(diagnostic_evidence.mlx_peak_memory_bytes);
        if let Some(trial_peak_memory_bytes) = trial_peak_memory_bytes {
            peak_mlx_memory_bytes = Some(peak_mlx_memory_bytes.map_or(
                trial_peak_memory_bytes,
                |current_peak_memory_bytes: u64| {
                    current_peak_memory_bytes.max(trial_peak_memory_bytes)
                },
            ));
        }
    }
    format!(
        "retained_route_assignment_hit_count={retained_route_assignment_hit_count}, retained_route_assignment_miss_count={retained_route_assignment_miss_count}, hot_expert_route_fully_covered_count={hot_expert_route_fully_covered_count}, hot_expert_route_partially_covered_count={hot_expert_route_partially_covered_count}, hot_expert_route_fully_missed_count={hot_expert_route_fully_missed_count}, decode_expert_cache_slot_eviction_count={decode_expert_cache_slot_eviction_count}, positional_file_read_byte_count={positional_file_read_byte_count}, process_physical_disk_read_bytes={process_physical_disk_read_bytes:?}, peak_mlx_memory_bytes={peak_mlx_memory_bytes:?}"
    )
}

pub(super) fn report_clean_policy_comparison(paired_trials: &[PairedDecodeCacheTrial]) {
    let baseline_decode_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.baseline.decode_tokens_per_second)
        .collect::<Vec<_>>();
    let candidate_decode_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.candidate.decode_tokens_per_second)
        .collect::<Vec<_>>();
    let baseline_prefill_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.baseline.prefill_tokens_per_second)
        .collect::<Vec<_>>();
    let candidate_prefill_tokens_per_second = paired_trials
        .iter()
        .map(|paired_trial| paired_trial.candidate.prefill_tokens_per_second)
        .collect::<Vec<_>>();
    let decode_throughput_confidence_interval = paired_relative_improvement_confidence_interval_95(
        &baseline_decode_tokens_per_second,
        &candidate_decode_tokens_per_second,
        false,
    );
    let prefill_throughput_confidence_interval = paired_relative_improvement_confidence_interval_95(
        &baseline_prefill_tokens_per_second,
        &candidate_prefill_tokens_per_second,
        false,
    );
    eprintln!(
        "[decode-cache-experiment] trials={} diagnostic=false decode_throughput_improvement={} prefill_throughput_improvement={}",
        paired_trials.len(),
        format_confidence_interval(decode_throughput_confidence_interval),
        format_confidence_interval(prefill_throughput_confidence_interval)
    );
}

pub(super) fn format_confidence_interval(
    confidence_interval: Option<PairedConfidenceInterval95>,
) -> String {
    confidence_interval.map_or_else(
        || "unavailable".to_owned(),
        |confidence_interval| {
            format!(
                "{:.2}% (95% confidence interval {:.2}% to {:.2}%)",
                confidence_interval.mean_improvement_percent,
                confidence_interval.lower_bound_percent,
                confidence_interval.upper_bound_percent
            )
        },
    )
}

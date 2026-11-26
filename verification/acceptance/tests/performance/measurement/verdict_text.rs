//! How a profiler comparison's metric is printed by the verdict journeys.

use mlx_performance_profiler::{MetricComparison, Verdict};

/// `fused faster -3.1% (12.03 against 12.40 ms, faster in 11 of 12 rounds)`,
/// with `candidate_name` naming the candidate job.
pub(super) fn metric_text(metric: &MetricComparison, candidate_name: &str) -> String {
    let verdict_text = match metric.verdict {
        Verdict::CandidateFaster => format!("{candidate_name} faster"),
        Verdict::WithinSpread => "within spread".to_owned(),
        Verdict::CandidateSlower => format!("{candidate_name} slower"),
    };
    format!(
        "{verdict_text} {:+.1}% ({:.2} against {:.2} ms, faster in {} of {} rounds)",
        metric.relative_difference * 100.0,
        metric.candidate_median_microseconds / 1_000.0,
        metric.baseline_median_microseconds / 1_000.0,
        metric.rounds_candidate_faster,
        metric.round_count
    )
}

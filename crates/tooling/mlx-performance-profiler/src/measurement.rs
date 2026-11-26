//! The safe measurement types and entry points.

use std::path::Path;

use mlx_bridge::MlxArray;

use crate::native_interface;

/// A job the profiler can measure: something that builds its graph afresh.
///
/// The profiler measures a job over several passes (a warm-up, uninstrumented
/// runs for the job's duration, an instrumented timeline, a serial pass), and
/// an evaluated array cannot be evaluated again, so it asks for a new graph
/// each time. The inputs a graph reads may be built once and shared; the
/// arrays returned here must be new and unevaluated on every call.
pub trait MeasuredJob {
    /// A name for the reports. It becomes part of file names: not empty, no
    /// `/`, and different from the job it is compared with.
    fn name(&self) -> &str;

    /// The outputs of a fresh, unevaluated graph of this job.
    ///
    /// # Errors
    /// A description of why the graph could not be built; the measurement
    /// stops and reports it.
    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String>;
}

/// Where the profiler writes: `<directory>/<file_base>.html` and `.json`,
/// and for a comparison one more pair a job.
#[derive(Debug, Clone, Copy)]
pub struct ReportLocation<'report> {
    pub title: &'report str,
    pub directory: &'report Path,
    pub file_base: &'report str,
}

/// One job's headline figures, in microseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JobSummary {
    /// The job's duration over the uninstrumented runs.
    pub job_wall_median_microseconds: f64,
    pub job_wall_lowest_microseconds: f64,
    pub job_wall_highest_microseconds: f64,
    /// GPU execution summed over the nodes Metal timed.
    pub gpu_execution_microseconds: f64,
    pub executed_node_count: i32,
    pub gpu_measured_node_count: i32,
}

/// Whether the candidate differs from the baseline on one metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    CandidateFaster,
    /// The interval for the median difference includes zero: the two are
    /// not told apart by these rounds.
    WithinSpread,
    CandidateSlower,
}

/// One metric over the paired rounds, in microseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricComparison {
    pub baseline_median_microseconds: f64,
    pub candidate_median_microseconds: f64,
    /// The median of the per-round differences, candidate minus baseline:
    /// negative is faster.
    pub median_difference_microseconds: f64,
    pub difference_interval_low_microseconds: f64,
    pub difference_interval_high_microseconds: f64,
    /// The median difference over the baseline's median.
    pub relative_difference: f64,
    pub verdict: Verdict,
    pub rounds_candidate_faster: i32,
    pub round_count: i32,
}

/// Two jobs measured with their rounds interleaved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JobComparison {
    pub baseline: JobSummary,
    pub candidate: JobSummary,
    /// The whole job's duration, uninstrumented.
    pub job_wall: MetricComparison,
    /// The whole job as one timed GPU unit.
    pub gpu_span: MetricComparison,
    /// The host's time inside the evaluation call that submits the job.
    pub host_submission: MetricComparison,
    /// The largest absolute difference between the two jobs' outputs.
    pub largest_output_difference: f64,
}

/// Why a measurement did not happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilerError {
    pub description: String,
}

impl std::fmt::Display for ProfilerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MLX performance profiler: {}", self.description)
    }
}

impl std::error::Error for ProfilerError {}

/// Measures `job` and writes its page and its JSON record of every node.
///
/// # Errors
/// When the job cannot build its graph, a name or path is not usable, or the
/// profiler fails; nothing is written then.
pub fn measure_job(
    job: &mut dyn MeasuredJob,
    report: ReportLocation<'_>,
) -> Result<JobSummary, ProfilerError> {
    native_interface::measure_job(job, report)
}

/// Measures `baseline` and `candidate` over interleaved rounds and writes the
/// comparison, opening with its verdict, and each job's own page and JSON.
///
/// # Errors
/// When either job cannot build its graph, when their outputs differ in
/// count, shape or dtype, when a name or path is not usable, or when the
/// profiler fails; nothing is written then.
pub fn compare_jobs(
    baseline: &mut dyn MeasuredJob,
    candidate: &mut dyn MeasuredJob,
    report: ReportLocation<'_>,
) -> Result<JobComparison, ProfilerError> {
    native_interface::compare_jobs(baseline, candidate, report)
}

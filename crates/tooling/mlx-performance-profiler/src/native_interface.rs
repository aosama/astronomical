//! The safe owner of the native profiler library: its C declarations, the
//! callback it builds graphs through, and the conversions on the way back.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use mlx_bridge::raw;

use crate::measurement::{
    JobComparison, JobSummary, MeasuredJob, MetricComparison, ProfilerError, ReportLocation,
    Verdict,
};

/// `mlx_perf_job_summary` of `mlx_perf_c_api.h`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeJobSummary {
    job_wall_median_us: f64,
    job_wall_min_us: f64,
    job_wall_max_us: f64,
    timeline_wall_us: f64,
    sum_of_stages_us: f64,
    measured_gpu_execution_us: f64,
    cross_stream_stall_us: f64,
    executed_node_count: c_int,
    gpu_measured_node_count: c_int,
    stream_count: c_int,
}

/// `mlx_perf_metric_comparison`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeMetricComparison {
    baseline_median_us: f64,
    baseline_min_us: f64,
    baseline_max_us: f64,
    baseline_lower_quartile_us: f64,
    baseline_upper_quartile_us: f64,
    candidate_median_us: f64,
    candidate_min_us: f64,
    candidate_max_us: f64,
    candidate_lower_quartile_us: f64,
    candidate_upper_quartile_us: f64,
    median_difference_us: f64,
    difference_interval_low_us: f64,
    difference_interval_high_us: f64,
    difference_interval_coverage: f64,
    sign_test_p_value: f64,
    relative_difference: f64,
    verdict: c_int,
    rounds_candidate_faster: c_int,
    round_count: c_int,
}

/// `mlx_perf_comparison_summary`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeComparisonSummary {
    baseline: NativeJobSummary,
    candidate: NativeJobSummary,
    job_wall: NativeMetricComparison,
    gpu_span: NativeMetricComparison,
    host_submission: NativeMetricComparison,
    max_absolute_output_difference: f64,
}

type BuildGraphFunction = unsafe extern "C" fn(raw::mlx_vector_array, *mut c_void) -> c_int;

/// `mlx_perf_variant`, field for field.
#[repr(C)]
struct NativeVariant {
    variant_name: *const c_char,
    build_graph: BuildGraphFunction,
    builder_payload: *mut c_void,
}

unsafe extern "C" {
    fn mlx_perf_profile_to_report(
        build_graph: BuildGraphFunction,
        builder_payload: *mut c_void,
        variant_name: *const c_char,
        report_title: *const c_char,
        output_directory: *const c_char,
        report_file_base: *const c_char,
        summary: *mut NativeJobSummary,
    ) -> c_int;
    fn mlx_perf_compare_to_report(
        baseline: *const NativeVariant,
        candidate: *const NativeVariant,
        report_title: *const c_char,
        output_directory: *const c_char,
        report_file_base: *const c_char,
        summary: *mut NativeComparisonSummary,
    ) -> c_int;
    fn mlx_perf_set_console_output(is_enabled: c_int);
    fn mlx_perf_last_error() -> *const c_char;
}

/// What the callback needs: the caller's job, and where to leave the reason
/// when the job cannot build its graph.
struct JobCall<'job> {
    job: &'job mut dyn MeasuredJob,
    failure: Option<String>,
}

/// The report's strings as the C interface takes them.
struct NativeReportLocation {
    title: CString,
    directory: CString,
    file_base: CString,
}

impl NativeReportLocation {
    fn new(report: ReportLocation<'_>) -> Result<Self, ProfilerError> {
        let directory = report.directory.to_str().ok_or_else(|| ProfilerError {
            description: "the report directory is not UTF-8".to_owned(),
        })?;
        Ok(Self {
            title: c_string(report.title, "the report title")?,
            directory: c_string(directory, "the report directory")?,
            file_base: c_string(report.file_base, "the report file name")?,
        })
    }
}

fn c_string(text: &str, described_as: &str) -> Result<CString, ProfilerError> {
    CString::new(text).map_err(|_| ProfilerError {
        description: format!("{described_as} holds a NUL byte"),
    })
}

pub(crate) fn measure_job(
    job: &mut dyn MeasuredJob,
    report: ReportLocation<'_>,
) -> Result<JobSummary, ProfilerError> {
    let native_report = NativeReportLocation::new(report)?;
    let job_name = c_string(job.name(), "the job name")?;
    let mut job_call = JobCall { job, failure: None };
    let mut native_summary = NativeJobSummary::default();
    // SAFETY: every pointer is valid for the call, which keeps none of them:
    // the strings and the summary outlive it, and `job_call` is borrowed
    // exclusively until it returns, so the callback is its only user.
    let status = unsafe {
        mlx_perf_set_console_output(0);
        mlx_perf_profile_to_report(
            build_fresh_graph,
            std::ptr::from_mut(&mut job_call).cast(),
            job_name.as_ptr(),
            native_report.title.as_ptr(),
            native_report.directory.as_ptr(),
            native_report.file_base.as_ptr(),
            &raw mut native_summary,
        )
    };
    measurement_outcome(status, [job_call.failure, None])?;
    Ok(job_summary(&native_summary))
}

pub(crate) fn compare_jobs(
    baseline: &mut dyn MeasuredJob,
    candidate: &mut dyn MeasuredJob,
    report: ReportLocation<'_>,
) -> Result<JobComparison, ProfilerError> {
    let native_report = NativeReportLocation::new(report)?;
    let baseline_name = c_string(baseline.name(), "the baseline job name")?;
    let candidate_name = c_string(candidate.name(), "the candidate job name")?;
    let mut baseline_call = JobCall {
        job: baseline,
        failure: None,
    };
    let mut candidate_call = JobCall {
        job: candidate,
        failure: None,
    };
    let native_baseline = NativeVariant {
        variant_name: baseline_name.as_ptr(),
        build_graph: build_fresh_graph,
        builder_payload: std::ptr::from_mut(&mut baseline_call).cast(),
    };
    let native_candidate = NativeVariant {
        variant_name: candidate_name.as_ptr(),
        build_graph: build_fresh_graph,
        builder_payload: std::ptr::from_mut(&mut candidate_call).cast(),
    };
    let mut native_summary = NativeComparisonSummary::default();
    // SAFETY: as in `measure_job`; the two job calls are distinct objects,
    // each reached only through its own variant's payload, and the library
    // calls the callbacks on this thread, one at a time.
    let status = unsafe {
        mlx_perf_set_console_output(0);
        mlx_perf_compare_to_report(
            &raw const native_baseline,
            &raw const native_candidate,
            native_report.title.as_ptr(),
            native_report.directory.as_ptr(),
            native_report.file_base.as_ptr(),
            &raw mut native_summary,
        )
    };
    measurement_outcome(status, [baseline_call.failure, candidate_call.failure])?;
    Ok(JobComparison {
        baseline: job_summary(&native_summary.baseline),
        candidate: job_summary(&native_summary.candidate),
        job_wall: metric_comparison(&native_summary.job_wall),
        gpu_span: metric_comparison(&native_summary.gpu_span),
        host_submission: metric_comparison(&native_summary.host_submission),
        largest_output_difference: native_summary.max_absolute_output_difference,
    })
}

/// A job's own reason when it could not build its graph, else the library's.
fn measurement_outcome(
    status: c_int,
    job_failures: [Option<String>; 2],
) -> Result<(), ProfilerError> {
    let [baseline_failure, candidate_failure] = job_failures;
    if let Some(job_failure) = baseline_failure.or(candidate_failure) {
        return Err(ProfilerError {
            description: job_failure,
        });
    }
    if status == 0 {
        return Ok(());
    }
    // SAFETY: the library returns a NUL-terminated string in static or
    // thread-local storage, valid until this thread's next call into it.
    let description = unsafe { CStr::from_ptr(mlx_perf_last_error()) }
        .to_string_lossy()
        .into_owned();
    Err(ProfilerError { description })
}

/// Builds the job's fresh graph for the library and appends its outputs. A
/// panic becomes a failed build, because unwinding into C++ would abort the
/// process.
unsafe extern "C" fn build_fresh_graph(
    fresh_outputs: raw::mlx_vector_array,
    builder_payload: *mut c_void,
) -> c_int {
    // SAFETY: the payload is the `JobCall` the entry point lent for the
    // whole measurement, and the library calls this on the thread that
    // entered it, one call at a time, so this is the only reference.
    let job_call = unsafe { &mut *builder_payload.cast::<JobCall<'_>>() };
    let built_outputs = catch_unwind(AssertUnwindSafe(|| job_call.job.fresh_outputs()));
    let output_arrays = match built_outputs {
        Ok(Ok(output_arrays)) => output_arrays,
        Ok(Err(build_failure)) => {
            job_call.failure = Some(build_failure);
            return 1;
        }
        Err(_) => {
            job_call.failure = Some("the job panicked while building its graph".to_owned());
            return 1;
        }
    };
    for output_array in &output_arrays {
        // SAFETY: the vector is the library's, valid for this call, and
        // MLX-C copies the array handle it appends.
        let append_status =
            unsafe { raw::mlx_vector_array_append_value(fresh_outputs, output_array.raw()) };
        if append_status != 0 {
            job_call.failure = Some("MLX could not take one of the job's outputs".to_owned());
            return 1;
        }
    }
    0
}

fn job_summary(native_summary: &NativeJobSummary) -> JobSummary {
    JobSummary {
        job_wall_median_microseconds: native_summary.job_wall_median_us,
        job_wall_lowest_microseconds: native_summary.job_wall_min_us,
        job_wall_highest_microseconds: native_summary.job_wall_max_us,
        gpu_execution_microseconds: native_summary.measured_gpu_execution_us,
        executed_node_count: native_summary.executed_node_count,
        gpu_measured_node_count: native_summary.gpu_measured_node_count,
    }
}

fn metric_comparison(native_metric: &NativeMetricComparison) -> MetricComparison {
    let verdict = match native_metric.verdict {
        verdict_value if verdict_value < 0 => Verdict::CandidateFaster,
        0 => Verdict::WithinSpread,
        _ => Verdict::CandidateSlower,
    };
    MetricComparison {
        baseline_median_microseconds: native_metric.baseline_median_us,
        candidate_median_microseconds: native_metric.candidate_median_us,
        median_difference_microseconds: native_metric.median_difference_us,
        difference_interval_low_microseconds: native_metric.difference_interval_low_us,
        difference_interval_high_microseconds: native_metric.difference_interval_high_us,
        relative_difference: native_metric.relative_difference,
        verdict,
        rounds_candidate_faster: native_metric.rounds_candidate_faster,
        round_count: native_metric.round_count,
    }
}

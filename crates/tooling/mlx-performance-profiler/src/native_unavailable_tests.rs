//! Off macOS every measurement is refused with the reason, through the
//! crate's public entry points, and never reaches the caller's code.

use std::path::Path;

use mlx_bridge::MlxArray;

use crate::native_unavailable::UNAVAILABLE_PLATFORM_DESCRIPTION;
use crate::{
    MeasuredJob, MeasuredProcedure, ProfilerError, ReportLocation, compare_jobs,
    compare_procedures, measure_job, measure_wire_release_lag, read_wired_memory,
};

/// A job or procedure that counts how often the profiler called into it.
#[derive(Default)]
struct CountingWork {
    call_count: usize,
}

impl MeasuredJob for CountingWork {
    fn name(&self) -> &str {
        "counting"
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        self.call_count += 1;
        Ok(Vec::new())
    }
}

impl MeasuredProcedure for CountingWork {
    fn name(&self) -> &str {
        "counting"
    }

    fn prepare(&mut self) -> Result<(), String> {
        self.call_count += 1;
        Ok(())
    }

    fn run(&mut self) -> Result<(), String> {
        self.call_count += 1;
        Ok(())
    }
}

fn never_written_report() -> ReportLocation<'static> {
    ReportLocation {
        title: "unused",
        directory: Path::new("unused-report-directory"),
        file_base: "unused",
    }
}

fn assert_refused_off_macos(profiler_error: &ProfilerError) {
    assert_eq!(profiler_error.description, UNAVAILABLE_PLATFORM_DESCRIPTION);
    assert!(
        profiler_error.to_string().contains("runs on macOS only"),
        "the message says why: {profiler_error}"
    );
}

#[test]
fn should_refuse_to_measure_a_job_without_building_its_graph() {
    let mut job = CountingWork::default();
    let profiler_error =
        measure_job(&mut job, never_written_report()).expect_err("off macOS nothing is measured");
    assert_refused_off_macos(&profiler_error);
    assert_eq!(job.call_count, 0, "the job's graph is never built");
}

#[test]
fn should_refuse_to_compare_jobs_without_building_either_graph() {
    let mut baseline = CountingWork::default();
    let mut candidate = CountingWork::default();
    let profiler_error = compare_jobs(&mut baseline, &mut candidate, never_written_report())
        .expect_err("off macOS nothing is compared");
    assert_refused_off_macos(&profiler_error);
    assert_eq!(baseline.call_count + candidate.call_count, 0);
}

#[test]
fn should_refuse_to_compare_procedures_without_preparing_or_running_them() {
    let mut baseline = CountingWork::default();
    let mut candidate = CountingWork::default();
    let profiler_error = compare_procedures(&mut baseline, &mut candidate, never_written_report())
        .expect_err("off macOS nothing is compared");
    assert_refused_off_macos(&profiler_error);
    assert_eq!(baseline.call_count + candidate.call_count, 0);
}

#[test]
fn should_refuse_to_read_wired_memory() {
    let profiler_error = read_wired_memory().expect_err("off macOS there is no wired memory");
    assert_refused_off_macos(&profiler_error);
}

#[test]
fn should_refuse_the_wire_release_probe_even_for_zero_repetitions() {
    let profiler_error =
        measure_wire_release_lag(0, 0).expect_err("off macOS the probe does not run");
    assert_refused_off_macos(&profiler_error);
}

//! The stand-in for the native profiler library on every platform but
//! macOS, where the build script compiles nothing (issue #2683).
//!
//! The library times Metal command buffers and reads macOS's wired memory,
//! so it has nothing to measure elsewhere. Each entry point has the
//! signature of its native counterpart, so callers name one module whatever
//! the platform, and each answers with the same refusal before touching the
//! caller's job or procedure, as Core ML and the Neural Engine do off macOS:
//! compiled everywhere, and unavailable at run time.

use crate::measurement::{JobComparison, JobSummary, MeasuredJob, ProfilerError, ReportLocation};
use crate::procedure_measurement::{MeasuredProcedure, ProcedureComparison};
use crate::wired_memory::{WireReleaseLag, WiredMemoryReading};

pub(crate) const UNAVAILABLE_PLATFORM_DESCRIPTION: &str = "runs on macOS only: it times Metal command buffers and reads macOS's wired memory, which this platform does not have";

fn unavailable_on_this_platform() -> ProfilerError {
    ProfilerError {
        description: UNAVAILABLE_PLATFORM_DESCRIPTION.to_owned(),
    }
}

pub(crate) fn measure_job(
    _job: &mut dyn MeasuredJob,
    _report: ReportLocation<'_>,
) -> Result<JobSummary, ProfilerError> {
    Err(unavailable_on_this_platform())
}

pub(crate) fn compare_jobs(
    _baseline: &mut dyn MeasuredJob,
    _candidate: &mut dyn MeasuredJob,
    _report: ReportLocation<'_>,
) -> Result<JobComparison, ProfilerError> {
    Err(unavailable_on_this_platform())
}

pub(crate) fn compare_procedures(
    _baseline: &mut dyn MeasuredProcedure,
    _candidate: &mut dyn MeasuredProcedure,
    _report: ReportLocation<'_>,
) -> Result<ProcedureComparison, ProfilerError> {
    Err(unavailable_on_this_platform())
}

pub(crate) fn read_wired_memory() -> Result<WiredMemoryReading, ProfilerError> {
    Err(unavailable_on_this_platform())
}

pub(crate) fn measure_wire_release_lag(
    _allocation_bytes: u64,
    _repetition_count: i32,
) -> Result<WireReleaseLag, ProfilerError> {
    Err(unavailable_on_this_platform())
}

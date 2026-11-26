//! Measures an MLX job, and compares two implementations of one job, through
//! the pinned MLX performance profiler library.
//!
//! A speed question about a kernel here is nearly always "is this one faster
//! than that one at this shape", and a difference of a few percent between
//! two single runs is as often the machine as the code. The profiler answers
//! with paired rounds: twelve, with the order flipped every round, and a
//! verdict only when the interval for the median difference excludes zero.
//! It also checks that the two jobs agree on their outputs, and writes what
//! each node of each job cost (issue on the host, GPU execution from Metal's
//! own command buffer times, waits) as a page and as JSON.
//!
//! [`compare_jobs`] is that comparison and [`measure_job`] the one-job
//! measurement. A job is anything that can build a fresh, unevaluated graph
//! and hand over its outputs ([`MeasuredJob`]); the profiler builds it once
//! for every pass it runs.
//!
//! The library is native code compiled by this crate's build script against
//! the patched MLX sources; `native_interface` is its safe owner. Everything
//! here needs MLX, so the crate is empty without the `direct-mlx` feature.

// The only module that needs unsafe code: it hands the native library a
// pointer to the caller's job and receives it back while a pass builds its
// graph. Every block carries its SAFETY argument.
#[cfg(feature = "direct-mlx")]
#[allow(unsafe_code)]
mod native_interface;

#[cfg(feature = "direct-mlx")]
mod measurement;

#[cfg(feature = "direct-mlx")]
#[allow(unsafe_code)]
mod native_procedures;

#[cfg(feature = "direct-mlx")]
mod procedure_measurement;

// Two synchronous calls into the library, each writing one local struct.
#[cfg(feature = "direct-mlx")]
#[allow(unsafe_code)]
mod wired_memory;

#[cfg(feature = "direct-mlx")]
pub use wired_memory::{
    WireReleaseLag, WiredMemoryReading, measure_wire_release_lag, read_wired_memory,
};

#[cfg(feature = "direct-mlx")]
pub use procedure_measurement::{
    MeasuredProcedure, ProcedureComparison, ProcedureHandOffs, compare_procedures,
};

#[cfg(feature = "direct-mlx")]
pub use measurement::{
    JobComparison, JobSummary, MeasuredJob, MetricComparison, ProfilerError, ReportLocation,
    Verdict, compare_jobs, measure_job,
};

#[cfg(all(test, feature = "direct-mlx"))]
mod measurement_tests;

#[cfg(all(test, feature = "direct-mlx"))]
mod procedure_measurement_tests;

#[cfg(all(test, feature = "direct-mlx"))]
mod wired_memory_tests;

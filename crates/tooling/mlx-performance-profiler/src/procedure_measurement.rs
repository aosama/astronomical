//! Comparing two procedures: code that makes several MLX evaluations of its
//! own, such as an engine reading a prompt or writing a reply.
//!
//! A job (`MeasuredJob`) is one unevaluated graph the profiler evaluates. A
//! procedure runs however many graphs and evaluations it likes; the profiler
//! times each run whole, from its start until every command buffer it
//! committed has completed, over the same twelve interleaved, order-flipped
//! rounds, and gives the same verdicts. It also says how long the GPU was
//! busy and how long it waited on the host, and counts the hand-offs.

use crate::measurement::{MetricComparison, ProfilerError, ReportLocation};
#[cfg(target_os = "macos")]
use crate::native_procedures as native_library;
#[cfg(not(target_os = "macos"))]
use crate::native_unavailable as native_library;

/// Code the profiler can run again and again and time whole.
pub trait MeasuredProcedure {
    /// A name for the reports. It becomes part of file names: not empty, no
    /// `/`, and different from the procedure it is compared with.
    fn name(&self) -> &str;

    /// Puts back whatever a run starts from. Untimed, and called before
    /// every run; evaluate what it creates so that work is not left to the
    /// timed run.
    ///
    /// # Errors
    /// A description of why the state could not be restored; the comparison
    /// stops and reports it.
    fn prepare(&mut self) -> Result<(), String>;

    /// The procedure itself, timed.
    ///
    /// # Errors
    /// A description of why it could not finish; the comparison stops and
    /// reports it.
    fn run(&mut self) -> Result<(), String>;
}

/// One procedure's hand-offs: medians over the rounds, and its kernel
/// launches from one extra observed run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcedureHandOffs {
    /// MLX's submissions to the GPU.
    pub median_command_buffer_count: f64,
    /// Evaluations and synchronizes that stopped the calling thread until
    /// the GPU finished. Work a procedure hands to a thread of its own is
    /// not counted here.
    pub median_blocking_wait_count: f64,
    pub median_blocking_wait_microseconds: f64,
    pub kernel_launch_count: i32,
    /// Launches with fewer threadgroups than the GPU has cores; `None` when
    /// the core count is unknown.
    pub launches_unable_to_fill_the_gpu: Option<i32>,
}

/// Two procedures measured with their rounds interleaved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcedureComparison {
    pub baseline: ProcedureHandOffs,
    pub candidate: ProcedureHandOffs,
    /// A run from its start until its GPU work completed.
    pub duration: MetricComparison,
    /// Time at least one of the run's command buffers was on the GPU.
    pub gpu_busy: MetricComparison,
    /// The rest of the run: the GPU waiting on the host.
    pub gpu_idle: MetricComparison,
}

/// Measures `baseline` and `candidate` over interleaved rounds and writes
/// the comparison page and its JSON record.
///
/// # Errors
/// When either procedure fails to prepare or run, a name or path is not
/// usable, the profiler fails, or off macOS; nothing is written then.
pub fn compare_procedures(
    baseline: &mut dyn MeasuredProcedure,
    candidate: &mut dyn MeasuredProcedure,
    report: ReportLocation<'_>,
) -> Result<ProcedureComparison, ProfilerError> {
    native_library::compare_procedures(baseline, candidate, report)
}

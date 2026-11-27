//! The native side of procedure comparisons: the C declarations of
//! `mlx_perf_compare_procedures_to_report` and the callbacks it steps a
//! procedure through.

use std::ffi::{CString, c_char, c_int, c_void};

use crate::measurement::{ProfilerError, ReportLocation};
use crate::native_interface::{
    NativeMetricComparison, NativeReportLocation, c_string, guarded_step, measurement_outcome,
    metric_comparison, mlx_perf_set_console_output,
};
use crate::procedure_measurement::{MeasuredProcedure, ProcedureComparison, ProcedureHandOffs};

type ProcedureStepFunction = unsafe extern "C" fn(*mut c_void) -> c_int;

/// `mlx_perf_procedure`, field for field.
#[repr(C)]
struct NativeProcedure {
    procedure_name: *const c_char,
    prepare: ProcedureStepFunction,
    run: ProcedureStepFunction,
    procedure_payload: *mut c_void,
}

/// `mlx_perf_procedure_summary`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeProcedureSummary {
    median_command_buffer_count: f64,
    median_blocking_wait_count: f64,
    median_blocking_wait_us: f64,
    kernel_launch_count: c_int,
    launches_unable_to_fill_gpu: c_int,
}

/// `mlx_perf_memory_comparison`, field for field: one memory figure judged
/// over the paired rounds. Mirrored only so the summary below has the size
/// the library writes; the measurement journeys read memory from the JSON.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeMemoryComparison {
    baseline_median_bytes: f64,
    candidate_median_bytes: f64,
    median_difference_bytes: f64,
    difference_interval_low_bytes: f64,
    difference_interval_high_bytes: f64,
    relative_difference: f64,
    verdict: c_int,
    rounds_candidate_less: c_int,
    round_count: c_int,
}

/// `mlx_perf_procedure_comparison_summary`, field for field. The library
/// writes the whole struct, so every field it has must be here, in order:
/// a shorter mirror would let it write past the end of this one.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeProcedureComparisonSummary {
    baseline: NativeProcedureSummary,
    candidate: NativeProcedureSummary,
    duration: NativeMetricComparison,
    gpu_busy: NativeMetricComparison,
    gpu_idle: NativeMetricComparison,
    peak_mlx_active_added: NativeMemoryComparison,
    mlx_held_change: NativeMemoryComparison,
    system_wired_at_end: NativeMemoryComparison,
    baseline_rounds_crossing_wire_limit: c_int,
    candidate_rounds_crossing_wire_limit: c_int,
    baseline_rounds_with_compression: c_int,
    candidate_rounds_with_compression: c_int,
}

// The sizes the pinned release's C API reference gives (64 and 704 bytes):
// a mismatch is a mirror out of step with the header, caught at compile time
// rather than as a write past the end of the struct at run time.
const _: () = assert!(std::mem::size_of::<NativeMemoryComparison>() == 64);
const _: () = assert!(std::mem::size_of::<NativeProcedureComparisonSummary>() == 704);

unsafe extern "C" {
    fn mlx_perf_compare_procedures_to_report(
        baseline: *const NativeProcedure,
        candidate: *const NativeProcedure,
        report_title: *const c_char,
        output_directory: *const c_char,
        report_file_base: *const c_char,
        summary: *mut NativeProcedureComparisonSummary,
    ) -> c_int;
}

/// What the callbacks need: the caller's procedure, and where to leave the
/// reason when a step fails.
struct ProcedureCall<'procedure> {
    procedure: &'procedure mut dyn MeasuredProcedure,
    failure: Option<String>,
}

impl<'procedure> ProcedureCall<'procedure> {
    fn lending(procedure: &'procedure mut dyn MeasuredProcedure) -> Self {
        Self {
            procedure,
            failure: None,
        }
    }

    /// The C view of this call. The name and this call must outlive every
    /// use of it, which the one caller's scope guarantees.
    fn as_native(&mut self, procedure_name: &CString) -> NativeProcedure {
        NativeProcedure {
            procedure_name: procedure_name.as_ptr(),
            prepare: prepare_procedure,
            run: run_procedure,
            procedure_payload: std::ptr::from_mut(self).cast(),
        }
    }
}

#[derive(Clone, Copy)]
enum ProcedureStep {
    Prepare,
    Run,
}

pub(crate) fn compare_procedures(
    baseline: &mut dyn MeasuredProcedure,
    candidate: &mut dyn MeasuredProcedure,
    report: ReportLocation<'_>,
) -> Result<ProcedureComparison, ProfilerError> {
    let report_strings = NativeReportLocation::new(report)?;
    let procedure_names = [
        c_string(baseline.name(), "the baseline procedure name")?,
        c_string(candidate.name(), "the candidate procedure name")?,
    ];
    let mut baseline_call = ProcedureCall::lending(baseline);
    let mut candidate_call = ProcedureCall::lending(candidate);
    let baseline_procedure = baseline_call.as_native(&procedure_names[0]);
    let candidate_procedure = candidate_call.as_native(&procedure_names[1]);
    let mut procedure_summary = NativeProcedureComparisonSummary::default();
    // SAFETY: every pointer is valid for the call, which keeps none of them:
    // the strings and the summary outlive it. The two procedure calls are
    // distinct objects, each reached only through its own payload, and the
    // library steps them on this thread, one at a time.
    let status = unsafe {
        mlx_perf_set_console_output(0);
        mlx_perf_compare_procedures_to_report(
            &raw const baseline_procedure,
            &raw const candidate_procedure,
            report_strings.title.as_ptr(),
            report_strings.directory.as_ptr(),
            report_strings.file_base.as_ptr(),
            &raw mut procedure_summary,
        )
    };
    measurement_outcome(status, [baseline_call.failure, candidate_call.failure])?;
    Ok(ProcedureComparison {
        baseline: procedure_hand_offs(&procedure_summary.baseline),
        candidate: procedure_hand_offs(&procedure_summary.candidate),
        duration: metric_comparison(&procedure_summary.duration),
        gpu_busy: metric_comparison(&procedure_summary.gpu_busy),
        gpu_idle: metric_comparison(&procedure_summary.gpu_idle),
    })
}

unsafe extern "C" fn prepare_procedure(procedure_payload: *mut c_void) -> c_int {
    // SAFETY: see `step_procedure`.
    unsafe { step_procedure(procedure_payload, ProcedureStep::Prepare) }
}

unsafe extern "C" fn run_procedure(procedure_payload: *mut c_void) -> c_int {
    // SAFETY: see `step_procedure`.
    unsafe { step_procedure(procedure_payload, ProcedureStep::Run) }
}

/// Takes one step of the caller's procedure. A failure or a panic becomes a
/// non-zero return, which ends the comparison.
///
/// # Safety
/// `procedure_payload` must be the `ProcedureCall` the entry point lent for
/// the whole comparison. The library calls this on the thread that entered
/// it, one call at a time, so this is the only reference.
unsafe fn step_procedure(procedure_payload: *mut c_void, procedure_step: ProcedureStep) -> c_int {
    // SAFETY: the caller's contract above.
    let ProcedureCall { procedure, failure } =
        unsafe { &mut *procedure_payload.cast::<ProcedureCall<'_>>() };
    let step_outcome = guarded_step(failure, "the procedure panicked", || match procedure_step {
        ProcedureStep::Prepare => procedure.prepare(),
        ProcedureStep::Run => procedure.run(),
    });
    c_int::from(step_outcome.is_none())
}

fn procedure_hand_offs(procedure_summary: &NativeProcedureSummary) -> ProcedureHandOffs {
    ProcedureHandOffs {
        median_command_buffer_count: procedure_summary.median_command_buffer_count,
        median_blocking_wait_count: procedure_summary.median_blocking_wait_count,
        median_blocking_wait_microseconds: procedure_summary.median_blocking_wait_us,
        kernel_launch_count: procedure_summary.kernel_launch_count,
        launches_unable_to_fill_the_gpu: (procedure_summary.launches_unable_to_fill_gpu >= 0)
            .then_some(procedure_summary.launches_unable_to_fill_gpu),
    }
}

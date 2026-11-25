//! The safe owner of the native host step.

use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use mlx_bridge::error::check_status;
use mlx_bridge::{MlxArray, MlxCError, MlxDtype, raw};

/// What a host step does with the arrays it reads.
///
/// `run` is called on MLX's CPU stream thread, not on the thread that built
/// the graph, and the value is dropped there too; hence `Send`. State a step
/// shares with the thread that evaluates needs its own synchronization.
pub trait HostStep: Send {
    /// Fills `output` from `inputs`. The inputs hold what the GPU steps
    /// before this one computed; the GPU steps after it read `output` once
    /// this returns.
    ///
    /// # Errors
    /// A failure fails the evaluation that reached this step. MLX has by then
    /// marked the output and everything computed from it as evaluated, with
    /// contents nobody wrote, so the caller discards that graph.
    fn run(&mut self, inputs: HostStepInputs<'_>, output: &mut [u8]) -> Result<(), HostStepFailed>;
}

/// A host step could not write its output. The evaluation that reached the
/// step fails with MLX's description; a step that has more to say about why
/// keeps it in its own state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostStepFailed;

/// The evaluated inputs of one run, each as its elements' bytes in row-major
/// order.
///
/// A view over the native step's own list, so a run allocates nothing to
/// read its inputs: a streamed reply runs a step a layer, on the path whose
/// cost this crate exists to remove.
#[derive(Clone, Copy)]
pub struct HostStepInputs<'run> {
    native_inputs: &'run [NativeHostStepInput],
}

impl<'run> HostStepInputs<'run> {
    pub fn count(&self) -> usize {
        self.native_inputs.len()
    }

    /// The bytes of input `input_index`, or `None` past the last input.
    pub fn bytes(&self, input_index: usize) -> Option<&'run [u8]> {
        let native_input = self.native_inputs.get(input_index)?;
        if native_input.byte_count == 0 {
            return Some(&[]);
        }
        // SAFETY: MLX keeps every input's buffer alive until the run that
        // reads it has finished, the native step refuses an input that is
        // not in row-major order, and `byte_count` is that array's own size.
        Some(unsafe {
            std::slice::from_raw_parts(native_input.bytes.cast::<u8>(), native_input.byte_count)
        })
    }
}

/// The array a host step writes: MLX allocates it when the evaluator reaches
/// the step, so its shape is fixed when the graph is built.
#[derive(Debug, Clone, Copy)]
pub struct HostStepOutput<'shape> {
    pub shape: &'shape [i32],
    pub dtype: MlxDtype,
}

#[repr(C)]
struct NativeHostStepInput {
    bytes: *const c_void,
    byte_count: usize,
}

#[repr(C)]
struct NativeHostStepRequest {
    input_arrays: *const raw::mlx_array,
    input_count: usize,
    output_shape: *const c_int,
    output_dimension_count: usize,
    output_dtype: raw::mlx_dtype,
    run: unsafe extern "C" fn(
        *mut c_void,
        *const NativeHostStepInput,
        usize,
        *mut c_void,
        usize,
    ) -> c_int,
    release: unsafe extern "C" fn(*mut c_void),
    host_state: *mut c_void,
}

unsafe extern "C" {
    fn astro_graph_host_step_append(
        request: *const NativeHostStepRequest,
        output_array: *mut raw::mlx_array,
    ) -> c_int;
    fn astro_graph_host_step_set_fast_synch(is_enabled: bool);
}

/// Chooses how MLX orders a host step among GPU steps, from the next
/// evaluation on.
///
/// MLX has two ways. Its default is Metal shared events, which cost a step
/// about as much as ending the evaluation would: 173 microseconds a layer
/// against 163 for a host stop. Its fast synch is a spin-wait on memory the
/// GPU and the host share: 9 microseconds a layer
/// (`cargo measure-graph-host-step`). Fast synch keeps one processor core
/// and one GPU kernel spinning while each waits for the other, so a caller
/// turns it on for work that stops at the host often and leaves MLX's
/// default otherwise. MLX itself falls back to events on a GPU without
/// Metal 3.
///
/// The setting is process-wide and a plain flag inside MLX: change it only
/// from the thread that evaluates, between evaluations.
pub fn set_fast_synch(is_enabled: bool) {
    // SAFETY: the native side stores one flag that MLX reads when an
    // evaluation makes its fences; the caller's contract keeps the store
    // apart from any evaluation.
    unsafe { astro_graph_host_step_set_fast_synch(is_enabled) };
}

/// Adds `host_step` to the graph after `inputs` and returns its lazy output.
///
/// Nothing runs here. When the output, or anything computed from it, is
/// evaluated, MLX runs the step once on its CPU stream, after the steps that
/// produce `inputs` and before the steps that read the output.
///
/// The step is queued on the calling thread's CPU stream, so the graph is
/// evaluated on the thread that built it, as the worker's GPU graphs are.
///
/// # Errors
/// Returns MLX's description when the node cannot be added, as for a negative
/// output dimension; `host_step` has been dropped by then.
pub fn append_host_step(
    inputs: &[&MlxArray],
    output: HostStepOutput<'_>,
    host_step: Box<dyn HostStep>,
) -> Result<MlxArray, MlxCError> {
    let mut input_handles = Vec::with_capacity(inputs.len());
    for input in inputs {
        input_handles.push(input.raw());
    }
    // A thin pointer the native step can carry: the outer box holds the wide
    // pointer to the trait object.
    let host_state = Box::into_raw(Box::new(host_step));
    let request = NativeHostStepRequest {
        input_arrays: input_handles.as_ptr(),
        input_count: input_handles.len(),
        output_shape: output.shape.as_ptr(),
        output_dimension_count: output.shape.len(),
        output_dtype: output.dtype.to_raw(),
        run: run_host_step,
        release: release_host_step,
        host_state: host_state.cast(),
    };
    let mut output_array = MlxArray::empty();
    // SAFETY: every pointer in the request is valid for this call, which
    // copies what it keeps; the native step takes ownership of `host_state`
    // on success and failure alike and releases it exactly once through
    // `release_host_step`.
    let status = unsafe { astro_graph_host_step_append(&request, output_array.raw_mut()) };
    check_status(status, "add a host step to an MLX graph")?;
    Ok(output_array)
}

/// Runs the caller's step for the native side. A panic becomes a failed run,
/// because unwinding into MLX's C++ would abort the process.
unsafe extern "C" fn run_host_step(
    host_state: *mut c_void,
    native_inputs: *const NativeHostStepInput,
    input_count: usize,
    output_bytes: *mut c_void,
    output_byte_count: usize,
) -> c_int {
    // SAFETY: `host_state` is the box `append_host_step` leaked, alive until
    // `release_host_step`, and MLX's one CPU stream thread runs a step's
    // single run with nothing else reaching that box.
    let host_step = unsafe { &mut *host_state.cast::<Box<dyn HostStep>>() };
    let native_inputs: &[NativeHostStepInput] = if input_count == 0 {
        &[]
    } else {
        // SAFETY: the native step passes its own list of `input_count`
        // entries, alive for the whole run.
        unsafe { std::slice::from_raw_parts(native_inputs, input_count) }
    };
    let output: &mut [u8] = if output_byte_count == 0 {
        &mut []
    } else {
        // SAFETY: MLX allocated `output_byte_count` bytes for the step's
        // output before queuing the run, and nothing reads them until the
        // run returns.
        unsafe { std::slice::from_raw_parts_mut(output_bytes.cast::<u8>(), output_byte_count) }
    };
    let inputs = HostStepInputs { native_inputs };
    // The one closure here is what `catch_unwind` takes.
    match catch_unwind(AssertUnwindSafe(|| host_step.run(inputs, output))) {
        Ok(Ok(())) => 0,
        Ok(Err(HostStepFailed)) | Err(_) => 1,
    }
}

/// Drops the caller's step when neither the graph nor a queued run can reach
/// it any more.
unsafe extern "C" fn release_host_step(host_state: *mut c_void) {
    // SAFETY: the native step calls this exactly once with the pointer
    // `append_host_step` leaked.
    drop(unsafe { Box::from_raw(host_state.cast::<Box<dyn HostStep>>()) });
}

#[cfg(test)]
#[path = "host_step_tests.rs"]
mod host_step_tests;

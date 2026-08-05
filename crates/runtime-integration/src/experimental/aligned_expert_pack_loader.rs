use std::{
    ffi::{CString, c_int, c_void},
    path::Path,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, mlx_runtime::check_status, raw};

/// One MLX output tensor allocated by the Metal I/O expert-pack loader.
#[derive(Clone, Debug)]
pub struct MlxMetalExpertPackOutputTensor {
    shape: Vec<i32>,
    dtype: MlxDtype,
}

impl MlxMetalExpertPackOutputTensor {
    /// Describes one compact destination tensor filled by one or more file ranges.
    #[must_use]
    pub fn new(shape: Vec<i32>, dtype: MlxDtype) -> Self {
        Self { shape, dtype }
    }
}

/// One source-to-output byte range submitted to Metal I/O.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxMetalExpertPackLoadRange {
    output_tensor_index: usize,
    output_tensor_offset_bytes: usize,
    source_file_offset_bytes: u64,
    byte_count: usize,
}

impl MlxMetalExpertPackLoadRange {
    /// Creates one non-overlapping destination range for a packed expert tensor.
    #[must_use]
    pub const fn new(
        output_tensor_index: usize,
        output_tensor_offset_bytes: usize,
        source_file_offset_bytes: u64,
        byte_count: usize,
    ) -> Self {
        Self {
            output_tensor_index,
            output_tensor_offset_bytes,
            source_file_offset_bytes,
            byte_count,
        }
    }

    /// Returns the compact output tensor receiving this range.
    #[must_use]
    pub const fn output_tensor_index(&self) -> usize {
        self.output_tensor_index
    }

    /// Returns the destination offset within the compact output tensor.
    #[must_use]
    pub const fn output_tensor_offset_bytes(&self) -> usize {
        self.output_tensor_offset_bytes
    }

    /// Returns the source offset within the aligned expert pack.
    #[must_use]
    pub const fn source_file_offset_bytes(&self) -> u64 {
        self.source_file_offset_bytes
    }

    /// Returns the exact byte count loaded by this range.
    #[must_use]
    pub const fn byte_count(&self) -> usize {
        self.byte_count
    }
}

/// Completion metrics for one directly encoded Metal I/O expert-pack request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxMetalExpertPackLoadMetrics {
    pub requested_byte_count: u64,
    pub command_count: usize,
    pub host_encoding_elapsed_nanoseconds: u64,
    pub queue_elapsed_nanoseconds: u64,
}

/// Lock-free aggregate of attributed direct Metal I/O expert-pack loads.
#[derive(Debug, Default)]
pub struct MlxMetalExpertPackLoadMetricsAccumulator {
    requested_byte_count: AtomicU64,
    command_count: AtomicU64,
    host_encoding_elapsed_nanoseconds: AtomicU64,
    completed_load_count: AtomicU64,
    queue_elapsed_nanoseconds: AtomicU64,
    maximum_queue_elapsed_nanoseconds: AtomicU64,
    failed_load_count: AtomicU64,
}

/// Immutable direct Metal I/O counters captured without waiting for completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxMetalExpertPackLoadMetricsSnapshot {
    pub requested_byte_count: u64,
    pub command_count: u64,
    pub host_encoding_elapsed_nanoseconds: u64,
    pub completed_load_count: u64,
    pub queue_elapsed_nanoseconds: u64,
    pub maximum_queue_elapsed_nanoseconds: u64,
    pub failed_load_count: u64,
}

impl MlxMetalExpertPackLoadMetricsAccumulator {
    #[must_use]
    pub fn snapshot(&self) -> MlxMetalExpertPackLoadMetricsSnapshot {
        MlxMetalExpertPackLoadMetricsSnapshot {
            requested_byte_count: self.requested_byte_count.load(Ordering::Relaxed),
            command_count: self.command_count.load(Ordering::Relaxed),
            host_encoding_elapsed_nanoseconds: self
                .host_encoding_elapsed_nanoseconds
                .load(Ordering::Relaxed),
            completed_load_count: self.completed_load_count.load(Ordering::Acquire),
            queue_elapsed_nanoseconds: self.queue_elapsed_nanoseconds.load(Ordering::Relaxed),
            maximum_queue_elapsed_nanoseconds: self
                .maximum_queue_elapsed_nanoseconds
                .load(Ordering::Relaxed),
            failed_load_count: self.failed_load_count.load(Ordering::Relaxed),
        }
    }

    fn record_submission(&self, submission_metrics: MlxMetalExpertPackLoadMetrics) {
        self.requested_byte_count
            .fetch_add(submission_metrics.requested_byte_count, Ordering::Relaxed);
        self.command_count.fetch_add(
            u64::try_from(submission_metrics.command_count).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.host_encoding_elapsed_nanoseconds.fetch_add(
            submission_metrics.host_encoding_elapsed_nanoseconds,
            Ordering::Relaxed,
        );
    }

    fn record_completion(&self, queue_elapsed_nanoseconds: u64, load_succeeded: c_int) {
        self.queue_elapsed_nanoseconds
            .fetch_add(queue_elapsed_nanoseconds, Ordering::Relaxed);
        self.maximum_queue_elapsed_nanoseconds
            .fetch_max(queue_elapsed_nanoseconds, Ordering::Relaxed);
        if load_succeeded == 0 {
            self.failed_load_count.fetch_add(1, Ordering::Relaxed);
        }
        self.completed_load_count.fetch_add(1, Ordering::Release);
    }
}

/// Arrays and native lifetime required for an in-flight Metal I/O expert-pack load.
#[derive(Debug)]
pub struct MlxMetalExpertPackLoad {
    native_load_handle: NativeMetalExpertPackLoadHandle,
    output_arrays: Vec<MlxArray>,
}

impl MlxMetalExpertPackLoad {
    /// Returns one MLX-owned output array while keeping its Metal I/O handle alive.
    pub fn output_array(&self, output_tensor_index: usize) -> Result<&MlxArray, MlxRuntimeError> {
        self.output_arrays.get(output_tensor_index).ok_or_else(|| {
            runtime_operation_error(
                "read a Metal I/O expert-pack output array",
                format!(
                    "output tensor index {output_tensor_index} exceeds {} output arrays",
                    self.output_arrays.len()
                ),
            )
        })
    }

    /// Waits only for measurement or teardown and verifies successful Metal I/O completion.
    pub fn wait_for_completion(&self) -> Result<MlxMetalExpertPackLoadMetrics, MlxRuntimeError> {
        let mut native_metrics = raw::astronomical_metal_expert_loader_metrics {
            requested_byte_count: 0,
            command_count: 0,
            host_encoding_elapsed_nanoseconds: 0,
            queue_elapsed_nanoseconds: 0,
            final_status: 0,
        };
        // SAFETY: The native handle remains owned by `self`, and the output
        // pointer is valid writable storage for the C metrics struct.
        let completion_status = unsafe {
            raw::astronomical_metal_expert_loader_wait(
                self.native_load_handle.native_handle.as_ptr(),
                &mut native_metrics,
            )
        };
        check_status(completion_status, "wait for a Metal I/O expert-pack load")?;
        if native_metrics.final_status != 3 {
            return Err(runtime_operation_error(
                "wait for a Metal I/O expert-pack load",
                format!(
                    "Metal I/O completed with unexpected status {}",
                    native_metrics.final_status
                ),
            ));
        }
        Ok(MlxMetalExpertPackLoadMetrics {
            requested_byte_count: native_metrics.requested_byte_count,
            command_count: native_metrics.command_count,
            host_encoding_elapsed_nanoseconds: native_metrics.host_encoding_elapsed_nanoseconds,
            queue_elapsed_nanoseconds: native_metrics.queue_elapsed_nanoseconds,
        })
    }
}

#[derive(Debug)]
struct NativeMetalExpertPackLoadHandle {
    native_handle: NonNull<raw::astronomical_metal_expert_loader_handle>,
}

impl Drop for NativeMetalExpertPackLoadHandle {
    fn drop(&mut self) {
        // SAFETY: This owner releases exactly one native handle after the
        // containing output-array vector drops. The native release waits for
        // Metal I/O completion before the final destination-buffer owner is released.
        unsafe {
            raw::astronomical_metal_expert_loader_free(self.native_handle.as_ptr());
        }
    }
}

impl MlxRuntime {
    /// Loads packed expert ranges and optionally records asynchronous completion metrics.
    pub fn load_metal_expert_pack_ranges(
        &self,
        source_file_path: &Path,
        output_tensors: &[MlxMetalExpertPackOutputTensor],
        load_ranges: &[MlxMetalExpertPackLoadRange],
        metal_expert_pack_load_metrics: Option<Arc<MlxMetalExpertPackLoadMetricsAccumulator>>,
    ) -> Result<MlxMetalExpertPackLoad, MlxRuntimeError> {
        const OPERATION: &str = "submit a Metal I/O expert-pack load";
        let source_file_path_text = source_file_path.to_str().ok_or_else(|| {
            runtime_operation_error(OPERATION, "source file path is not valid UTF-8")
        })?;
        let source_file_path_c_string = CString::new(source_file_path_text).map_err(|_| {
            runtime_operation_error(OPERATION, "source file path contains an interior NUL byte")
        })?;
        let native_output_tensors = output_tensors
            .iter()
            .map(|output_tensor| {
                Ok(raw::astronomical_metal_expert_loader_output_tensor {
                    shape: output_tensor.shape.as_ptr(),
                    dimension_count: i32::try_from(output_tensor.shape.len()).map_err(|_| {
                        runtime_operation_error(
                            OPERATION,
                            "Metal I/O output tensor rank exceeds the MLX C integer range",
                        )
                    })?,
                    dtype: output_tensor.dtype.to_raw(),
                })
            })
            .collect::<Result<Vec<_>, MlxRuntimeError>>()?;
        let native_load_ranges = load_ranges
            .iter()
            .map(
                |load_range| raw::astronomical_metal_expert_loader_load_range {
                    output_tensor_index: load_range.output_tensor_index,
                    output_tensor_offset_bytes: load_range.output_tensor_offset_bytes,
                    source_file_offset_bytes: load_range.source_file_offset_bytes,
                    byte_count: load_range.byte_count,
                },
            )
            .collect::<Vec<_>>();
        let mut raw_output_arrays = (0..output_tensors.len())
            .map(|_| MlxArray::empty_raw())
            .collect::<Vec<_>>();
        let mut native_load_handle_pointer = std::ptr::null_mut();
        let mut native_submission_metrics = raw::astronomical_metal_expert_loader_metrics {
            requested_byte_count: 0,
            command_count: 0,
            host_encoding_elapsed_nanoseconds: 0,
            queue_elapsed_nanoseconds: 0,
            final_status: 0,
        };
        // SAFETY: The C strings and descriptor vectors remain live throughout
        // the call, output handles are uniquely writable, and the runtime owns
        // the supplied live MLX GPU stream.
        let submission_status = match metal_expert_pack_load_metrics.as_ref() {
            Some(metal_expert_pack_load_metrics) => {
                let completion_callback_context =
                    Arc::into_raw(Arc::clone(metal_expert_pack_load_metrics))
                        .cast_mut()
                        .cast::<c_void>();
                // SAFETY: Native code owns the transferred Arc reference and releases it exactly
                // once after invoking the completion callback or rejecting submission.
                unsafe {
                    raw::astronomical_metal_expert_loader_start(
                        source_file_path_c_string.as_ptr(),
                        native_output_tensors.as_ptr(),
                        native_output_tensors.len(),
                        native_load_ranges.as_ptr(),
                        native_load_ranges.len(),
                        self.gpu_stream().raw(),
                        raw_output_arrays.as_mut_ptr(),
                        &mut native_load_handle_pointer,
                        &mut native_submission_metrics,
                        completion_callback_context,
                        Some(record_metal_expert_pack_load_completion),
                        Some(release_metal_expert_pack_load_metrics),
                    )
                }
            }
            None => unsafe {
                raw::astronomical_metal_expert_loader_start(
                    source_file_path_c_string.as_ptr(),
                    native_output_tensors.as_ptr(),
                    native_output_tensors.len(),
                    native_load_ranges.as_ptr(),
                    native_load_ranges.len(),
                    self.gpu_stream().raw(),
                    raw_output_arrays.as_mut_ptr(),
                    &mut native_load_handle_pointer,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    None,
                    None,
                )
            },
        };
        if let Err(submission_error) = check_status(submission_status, OPERATION) {
            free_raw_output_arrays(&mut raw_output_arrays);
            return Err(submission_error);
        }
        if let Some(metal_expert_pack_load_metrics) = metal_expert_pack_load_metrics.as_ref() {
            metal_expert_pack_load_metrics.record_submission(MlxMetalExpertPackLoadMetrics {
                requested_byte_count: native_submission_metrics.requested_byte_count,
                command_count: native_submission_metrics.command_count,
                host_encoding_elapsed_nanoseconds: native_submission_metrics
                    .host_encoding_elapsed_nanoseconds,
                queue_elapsed_nanoseconds: 0,
            });
        }
        let native_load_handle = match NonNull::new(native_load_handle_pointer) {
            Some(native_load_handle) => native_load_handle,
            None => {
                free_raw_output_arrays(&mut raw_output_arrays);
                return Err(runtime_operation_error(
                    OPERATION,
                    "native Metal I/O loader returned no completion handle",
                ));
            }
        };
        if raw_output_arrays
            .iter()
            .any(|raw_output_array| raw_output_array.ctx.is_null())
        {
            free_raw_output_arrays(&mut raw_output_arrays);
            // SAFETY: The native start call returned this handle to the caller,
            // so Rust must release it when rejecting malformed output handles.
            unsafe {
                raw::astronomical_metal_expert_loader_free(native_load_handle.as_ptr());
            }
            return Err(runtime_operation_error(
                OPERATION,
                "native Metal I/O loader returned an empty output array handle",
            ));
        }
        let native_load_handle = NativeMetalExpertPackLoadHandle {
            native_handle: native_load_handle,
        };
        let output_arrays = raw_output_arrays
            .into_iter()
            .map(MlxArray::from_populated_owned_raw)
            .collect::<Vec<_>>();
        Ok(MlxMetalExpertPackLoad {
            native_load_handle,
            output_arrays,
        })
    }
}

unsafe extern "C" fn record_metal_expert_pack_load_completion(
    completion_callback_context: *mut c_void,
    queue_elapsed_nanoseconds: u64,
    load_succeeded: c_int,
) {
    if completion_callback_context.is_null() {
        return;
    }
    // SAFETY: Native code retains one Arc reference until this callback and its release callback
    // have completed, so the accumulator remains live for this non-owning access.
    let metal_expert_pack_load_metrics =
        unsafe { &*completion_callback_context.cast::<MlxMetalExpertPackLoadMetricsAccumulator>() };
    metal_expert_pack_load_metrics.record_completion(queue_elapsed_nanoseconds, load_succeeded);
}

unsafe extern "C" fn release_metal_expert_pack_load_metrics(
    completion_callback_context: *mut c_void,
) {
    if completion_callback_context.is_null() {
        return;
    }
    // SAFETY: This pointer came from Arc::into_raw and native code releases it exactly once.
    unsafe {
        drop(Arc::from_raw(
            completion_callback_context.cast::<MlxMetalExpertPackLoadMetricsAccumulator>(),
        ));
    }
}

fn free_raw_output_arrays(raw_output_arrays: &mut [raw::mlx_array]) {
    for raw_output_array in raw_output_arrays {
        // SAFETY: This error-path helper releases only raw output arrays that
        // have not been moved into an owning `MlxArray`.
        unsafe {
            raw::mlx_array_free(*raw_output_array);
        }
        *raw_output_array = MlxArray::empty_raw();
    }
}

fn runtime_operation_error(
    operation: &'static str,
    description: impl Into<String>,
) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.into(),
    }
}

//! Running a loaded model, synchronously or on its own dispatch queue.

use std::ffi::c_char;
use std::time::Duration;

use super::error::{CoreMlProgramError, NATIVE_ERROR_CAPACITY, native_error_text};
use super::program::CoreMlProgram;
use crate::core_ml::ffi::{
    astronomical_core_ml_program_begin_predict,
    astronomical_core_ml_program_copy_asynchronous_error,
    astronomical_core_ml_program_finish_predict, astronomical_core_ml_program_predict,
};

/// A prediction running on the program's own dispatch queue while the
/// caller does other work, typically an MLX evaluation on the GPU.
///
/// The guard borrows the program mutably, so its IOSurfaces cannot be
/// rewritten or read until the prediction is finished; dropping the guard
/// without finishing waits for the prediction rather than abandoning it.
pub struct PendingPrediction<'program> {
    program: &'program mut CoreMlProgram,
    is_finished: bool,
}

impl PendingPrediction<'_> {
    /// Waits up to `timeout` and returns the prediction's own wall time.
    pub fn finish(mut self, timeout: Duration) -> Result<Duration, CoreMlProgramError> {
        let mut elapsed_nanoseconds = 0_u64;
        let timeout_nanoseconds = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: the handle is live (borrowed from the program) and the
        // out-pointer is valid and exclusive.
        let status = unsafe {
            astronomical_core_ml_program_finish_predict(
                self.program.native_program.as_ptr(),
                timeout_nanoseconds,
                &mut elapsed_nanoseconds,
            )
        };
        match status {
            0 => {
                self.is_finished = true;
                Ok(Duration::from_nanos(elapsed_nanoseconds))
            }
            // Waiting on would hang the request forever, which is what the timeout is for.
            // The program is still in flight, so it is abandoned rather than released.
            -2 => {
                self.is_finished = true;
                self.program.is_abandoned = true;
                Err(CoreMlProgramError::PredictionTimeout(timeout))
            }
            _ => {
                self.is_finished = true;
                let mut native_error = [0 as c_char; NATIVE_ERROR_CAPACITY];
                // SAFETY: the prediction has finished, so nothing else writes the error
                // it copies, and the buffer is writable for its full stated capacity.
                unsafe {
                    astronomical_core_ml_program_copy_asynchronous_error(
                        self.program.native_program.as_ptr().cast_const(),
                        native_error.as_mut_ptr(),
                        NATIVE_ERROR_CAPACITY as u32,
                    );
                }
                Err(CoreMlProgramError::Prediction(native_error_text(
                    &native_error,
                )))
            }
        }
    }
}

impl Drop for PendingPrediction<'_> {
    fn drop(&mut self) {
        if self.is_finished {
            return;
        }
        let mut elapsed_nanoseconds = 0_u64;
        // SAFETY: as in `finish`; an unbounded wait is the only sound choice
        // here, because the surfaces must not be touched while in flight.
        unsafe {
            astronomical_core_ml_program_finish_predict(
                self.program.native_program.as_ptr(),
                u64::MAX,
                &mut elapsed_nanoseconds,
            );
        }
    }
}

impl CoreMlProgram {
    /// Runs one prediction and returns Core ML's wall time for it alone.
    pub fn predict(&mut self) -> Result<Duration, CoreMlProgramError> {
        let mut elapsed_nanoseconds = 0_u64;
        let mut native_error = [0 as c_char; NATIVE_ERROR_CAPACITY];
        // SAFETY: the handle is live, both out-pointers are valid and
        // exclusive, and the error buffer is writable for its capacity.
        let status = unsafe {
            astronomical_core_ml_program_predict(
                self.native_program.as_ptr(),
                &mut elapsed_nanoseconds,
                native_error.as_mut_ptr(),
                NATIVE_ERROR_CAPACITY as u32,
            )
        };
        if status == 0 {
            Ok(Duration::from_nanos(elapsed_nanoseconds))
        } else {
            Err(CoreMlProgramError::Prediction(native_error_text(
                &native_error,
            )))
        }
    }

    /// One prediction that gives up after `timeout`, for a thread something else will
    /// eventually wait on. A plain [`Self::predict`] on a wedged Neural Engine never
    /// returns, which would hang whatever joins that thread.
    pub fn predict_within(&mut self, timeout: Duration) -> Result<Duration, CoreMlProgramError> {
        self.begin_prediction()?.finish(timeout)
    }

    /// Runs the program once on zeros, which wires its weights like any other input.
    pub fn run_once_on_zeros(&mut self, timeout: Duration) -> Result<(), CoreMlProgramError> {
        let zero_activations = vec![0_u16; self.shape().activation_element_count()];
        self.write_activations(&zero_activations)?;
        self.predict_within(timeout)?;
        Ok(())
    }

    /// Starts a prediction on the program's dispatch queue and returns at
    /// once, so the caller's thread is free to drive the GPU meanwhile.
    pub fn begin_prediction(&mut self) -> Result<PendingPrediction<'_>, CoreMlProgramError> {
        // SAFETY: the handle is live; the bridge refuses a second start while
        // one is in flight, and the returned guard prevents that from Rust.
        let status =
            unsafe { astronomical_core_ml_program_begin_predict(self.native_program.as_ptr()) };
        if status != 0 {
            return Err(CoreMlProgramError::PredictionStart);
        }
        Ok(PendingPrediction {
            program: self,
            is_finished: false,
        })
    }
}

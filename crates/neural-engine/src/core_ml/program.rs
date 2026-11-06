//! A prefill program, or a whole MLP chain, loaded into Core ML with a chosen
//! set of compute units, and Apple's own report of where it was placed.

use std::ffi::{CString, c_char};
use std::path::Path;
use std::ptr::NonNull;

use super::compute_units::{CoreMlComputeUnits, LayerPlacement, layer_placement_from_native};
use super::error::{CoreMlProgramError, NATIVE_ERROR_CAPACITY, native_error_text};
use crate::core_ml::ffi::{
    NativeCoreMlProgram, astronomical_core_ml_program_load,
    astronomical_core_ml_program_model_load_seconds, astronomical_core_ml_program_placement,
    astronomical_core_ml_program_placement_seconds, astronomical_core_ml_program_release,
};
use crate::program_shape::ProgramShape;

pub struct CoreMlProgram {
    pub(super) native_program: NonNull<NativeCoreMlProgram>,
    pub(super) shape: ProgramShape,
}

// SAFETY: the handle has one owner and moves between threads with it; Core ML
// models and the bridge's dispatch queue have no thread affinity, and every
// call into the bridge takes this owner by reference, so calls cannot overlap
// across threads without `Sync`, which this type does not implement.
unsafe impl Send for CoreMlProgram {}

impl std::fmt::Debug for CoreMlProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CoreMlProgram")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl CoreMlProgram {
    /// Loads a written `.mlpackage` or a compiled `.mlmodelc` for
    /// `compute_units`. Compilation happens inside Core ML, so no developer tools
    /// are needed at run time. Files at a compiled model's path stay where they are
    /// when this program is released.
    pub fn load_from_path(
        model_path: &Path,
        shape: ProgramShape,
        compute_units: CoreMlComputeUnits,
    ) -> Result<Self, CoreMlProgramError> {
        Self::load(model_path, shape, compute_units, true)
    }

    /// Loads a program whose placement the caller proved on an earlier load, without
    /// asking Core ML for a compute plan again. [`Self::placement`] then reports an
    /// unknown placement, so this is only for a program with a record of that proof.
    pub fn load_from_path_without_placement_check(
        model_path: &Path,
        shape: ProgramShape,
        compute_units: CoreMlComputeUnits,
    ) -> Result<Self, CoreMlProgramError> {
        Self::load(model_path, shape, compute_units, false)
    }

    fn load(
        model_path: &Path,
        shape: ProgramShape,
        compute_units: CoreMlComputeUnits,
        checks_placement: bool,
    ) -> Result<Self, CoreMlProgramError> {
        let model_path_text = CString::new(model_path.to_string_lossy().as_bytes())
            .map_err(CoreMlProgramError::from_interior_nul)?;
        let mut native_error = [0 as c_char; NATIVE_ERROR_CAPACITY];
        // SAFETY: the path is a valid NUL-terminated string, the error buffer
        // is writable for its full stated capacity, and ProgramShape
        // guarantees the dimensions fit in i32.
        let native_program = unsafe {
            astronomical_core_ml_program_load(
                model_path_text.as_ptr(),
                compute_units.native_code(),
                i32::from(checks_placement),
                shape.input_channels() as i32,
                shape.output_channels() as i32,
                shape.token_count() as i32,
                native_error.as_mut_ptr(),
                NATIVE_ERROR_CAPACITY as u32,
            )
        };
        match NonNull::new(native_program) {
            Some(native_program) => Ok(Self {
                native_program,
                shape,
            }),
            None => Err(CoreMlProgramError::Load(native_error_text(&native_error))),
        }
    }

    #[must_use]
    pub fn token_count(&self) -> usize {
        self.shape.token_count()
    }

    #[must_use]
    pub fn output_channel_count(&self) -> usize {
        self.shape.output_channels()
    }

    #[must_use]
    pub fn shape(&self) -> ProgramShape {
        self.shape
    }

    /// Seconds Core ML spent creating the model, and then planning its placement.
    #[must_use]
    pub fn load_phase_seconds(&self) -> (f64, f64) {
        let native_program = self.native_program.as_ptr().cast_const();
        // SAFETY: the handle is valid for this borrow and these only read two doubles.
        unsafe {
            (
                astronomical_core_ml_program_model_load_seconds(native_program),
                astronomical_core_ml_program_placement_seconds(native_program),
            )
        }
    }

    #[must_use]
    pub fn placement(&self) -> LayerPlacement {
        // SAFETY: the handle is live for the lifetime of `self`.
        let native_placement =
            unsafe { astronomical_core_ml_program_placement(self.native_program.as_ptr()) };
        layer_placement_from_native(native_placement)
    }
}

impl Drop for CoreMlProgram {
    fn drop(&mut self) {
        // SAFETY: the handle came from the bridge's load function and is
        // released exactly once, here.
        unsafe { astronomical_core_ml_program_release(self.native_program.as_ptr()) }
    }
}

#[cfg(test)]
#[path = "program_tests.rs"]
mod tests;

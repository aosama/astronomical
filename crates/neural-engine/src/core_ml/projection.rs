//! A prefill projection, or a whole MLP chain, loaded into Core ML with a chosen
//! set of compute units, and Apple's own report of where it was placed.

use std::ffi::{CString, c_char};
use std::path::Path;
use std::ptr::NonNull;

use super::compute_units::{CoreMlComputeUnits, LayerPlacement, layer_placement_from_native};
use super::error::{CoreMlProjectionError, NATIVE_ERROR_CAPACITY, native_error_text};
use crate::core_ml::ffi::{
    NativeCoreMlProjection, astronomical_core_ml_projection_load,
    astronomical_core_ml_projection_model_load_seconds, astronomical_core_ml_projection_placement,
    astronomical_core_ml_projection_placement_seconds, astronomical_core_ml_projection_release,
};
use crate::projection_shape::ProjectionShape;

pub struct CoreMlProjection {
    pub(super) native_projection: NonNull<NativeCoreMlProjection>,
    pub(super) shape: ProjectionShape,
}

// SAFETY: the handle has one owner and moves between threads with it; Core ML
// models and the bridge's dispatch queue have no thread affinity, and every
// call into the bridge takes this owner by reference, so calls cannot overlap
// across threads without `Sync`, which this type does not implement.
unsafe impl Send for CoreMlProjection {}

impl std::fmt::Debug for CoreMlProjection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CoreMlProjection")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl CoreMlProjection {
    /// Loads a written model document, `.mlpackage`, or compiled `.mlmodelc` for
    /// `compute_units`. Compilation happens inside Core ML, so no developer tools
    /// are needed at run time. Files at a compiled model's path stay where they are
    /// when this program is released.
    pub fn load_from_path(
        model_path: &Path,
        shape: ProjectionShape,
        compute_units: CoreMlComputeUnits,
    ) -> Result<Self, CoreMlProjectionError> {
        let model_path_text = CString::new(model_path.to_string_lossy().as_bytes())
            .map_err(CoreMlProjectionError::from_interior_nul)?;
        let mut native_error = [0 as c_char; NATIVE_ERROR_CAPACITY];
        // SAFETY: the path is a valid NUL-terminated string, the error buffer
        // is writable for its full stated capacity, and ProjectionShape
        // guarantees the dimensions fit in i32.
        let native_projection = unsafe {
            astronomical_core_ml_projection_load(
                model_path_text.as_ptr(),
                compute_units.native_code(),
                shape.input_channels() as i32,
                shape.output_channels() as i32,
                shape.token_count() as i32,
                native_error.as_mut_ptr(),
                NATIVE_ERROR_CAPACITY as u32,
            )
        };
        match NonNull::new(native_projection) {
            Some(native_projection) => Ok(Self {
                native_projection,
                shape,
            }),
            None => Err(CoreMlProjectionError::Load(native_error_text(
                &native_error,
            ))),
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
    pub fn shape(&self) -> ProjectionShape {
        self.shape
    }

    /// Seconds Core ML spent creating the model, and then planning its placement.
    #[must_use]
    pub fn load_phase_seconds(&self) -> (f64, f64) {
        let native_projection = self.native_projection.as_ptr().cast_const();
        // SAFETY: the handle is valid for this borrow and these only read two doubles.
        unsafe {
            (
                astronomical_core_ml_projection_model_load_seconds(native_projection),
                astronomical_core_ml_projection_placement_seconds(native_projection),
            )
        }
    }

    #[must_use]
    pub fn placement(&self) -> LayerPlacement {
        // SAFETY: the handle is live for the lifetime of `self`.
        let native_placement =
            unsafe { astronomical_core_ml_projection_placement(self.native_projection.as_ptr()) };
        layer_placement_from_native(native_placement)
    }
}

impl Drop for CoreMlProjection {
    fn drop(&mut self) {
        // SAFETY: the handle came from the bridge's load function and is
        // released exactly once, here.
        unsafe { astronomical_core_ml_projection_release(self.native_projection.as_ptr()) }
    }
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;

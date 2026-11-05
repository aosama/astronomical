//! A prefill projection, or a whole MLP chain, loaded into Core ML with a chosen
//! set of compute units, and Apple's own report of where it was placed.

use std::ffi::{CString, c_char};
use std::fs;
use std::path::Path;
use std::ptr::NonNull;

use super::compute_units::{CoreMlComputeUnits, LayerPlacement, layer_placement_from_native};
use super::error::{CoreMlProjectionError, NATIVE_ERROR_CAPACITY, native_error_text};
use crate::core_ml_projection_document::encode_projection_document;
use crate::ml_package_layout::write_ml_package;
use crate::ml_program_mlp_document::{MlpFp16Weights, encode_ml_program_mlp};
use crate::ml_program_projection_document::{MlProgramWeights, encode_ml_program_projection};
use crate::ml_program_rotated_mlp_document::{
    RotatedMlpLayout, RotatedMlpWeights, encode_ml_program_rotated_mlp,
};
use crate::native_bridge::{
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
    /// Writes the model document into `model_directory`, compiles it, and
    /// loads it for `compute_units`. Compilation happens inside Core ML, so
    /// no developer tools are needed at run time. `contraction_part_count`
    /// splits the input channels into equal summed parts (one keeps a single
    /// convolution); see [`encode_projection_document`].
    pub fn load(
        shape: ProjectionShape,
        fp16_weights: &[u16],
        compute_units: CoreMlComputeUnits,
        contraction_part_count: usize,
        model_directory: &Path,
    ) -> Result<Self, CoreMlProjectionError> {
        let model_document =
            encode_projection_document(shape, fp16_weights, contraction_part_count)?;
        let model_path = model_directory.join("projection.mlmodel");
        fs::write(&model_path, model_document)?;
        Self::load_model_at(&model_path, shape, compute_units)
    }

    /// As [`Self::load`], but writes an ML Program package, which can keep
    /// the weights 4-bit; see [`encode_ml_program_projection`].
    pub fn load_ml_program(
        shape: ProjectionShape,
        weights: MlProgramWeights<'_>,
        compute_units: CoreMlComputeUnits,
        contraction_part_count: usize,
        model_directory: &Path,
    ) -> Result<Self, CoreMlProjectionError> {
        let package_contents =
            encode_ml_program_projection(shape, weights, contraction_part_count)?;
        let package_path = write_ml_package(model_directory, &package_contents)?;
        Self::load_model_at(&package_path, shape, compute_units)
    }

    /// As [`Self::load_ml_program`] for a whole MLP chain, whose activations
    /// and outputs are `shape`'s `[hidden][tokens]`; see
    /// [`encode_ml_program_mlp`].
    pub fn load_ml_program_mlp(
        shape: ProjectionShape,
        intermediate_channels: usize,
        weights: MlpFp16Weights<'_>,
        contraction_part_width: usize,
        compute_units: CoreMlComputeUnits,
        model_directory: &Path,
    ) -> Result<Self, CoreMlProjectionError> {
        let package_contents = encode_ml_program_mlp(
            shape,
            intermediate_channels,
            weights,
            contraction_part_width,
        )?;
        let package_path = write_ml_package(model_directory, &package_contents)?;
        Self::load_model_at(&package_path, shape, compute_units)
    }

    /// Loads one whole rotated ternary MLP. See
    /// [`crate::ml_program_rotated_mlp_document`].
    pub fn load_ml_program_rotated_mlp(
        layout: RotatedMlpLayout,
        weights: RotatedMlpWeights<'_>,
        compute_units: CoreMlComputeUnits,
        model_directory: &Path,
    ) -> Result<Self, CoreMlProjectionError> {
        let package_contents = encode_ml_program_rotated_mlp(layout, weights)?;
        let package_path = write_ml_package(model_directory, &package_contents)?;
        Self::load_model_at(&package_path, layout.shape, compute_units)
    }

    pub(super) fn load_model_at(
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

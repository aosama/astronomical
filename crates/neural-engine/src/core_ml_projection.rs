//! A prefill projection, or a whole MLP chain, loaded into Core ML with a
//! chosen set of compute units, plus Apple's own report of where Core ML
//! placed it and a zero-copy view of its output surface.

use std::ffi::{CStr, CString, c_char, c_void};
use std::fs;
use std::path::Path;
use std::ptr::NonNull;
use std::time::Duration;

use thiserror::Error;

use crate::core_ml_projection_document::encode_projection_document;
use crate::ml_package_layout::write_ml_package;
use crate::ml_program_mlp_document::{MlpFp16Weights, encode_ml_program_mlp};
use crate::ml_program_projection_document::{MlProgramWeights, encode_ml_program_projection};
use crate::ml_program_rotated_mlp_document::{
    RotatedMlpLayout, RotatedMlpWeights, encode_ml_program_rotated_mlp,
};
use crate::native_bridge::{
    NativeCoreMlProjection, NativeLayerPlacement, astronomical_core_ml_projection_begin_predict,
    astronomical_core_ml_projection_discard_compiled_model,
    astronomical_core_ml_projection_finish_predict, astronomical_core_ml_projection_load,
    astronomical_core_ml_projection_lock_output, astronomical_core_ml_projection_placement,
    astronomical_core_ml_projection_predict, astronomical_core_ml_projection_read_output,
    astronomical_core_ml_projection_release, astronomical_core_ml_projection_unlock_output,
    astronomical_core_ml_projection_write_input,
};
use crate::projection_model_interface::ProjectionDocumentError;
use crate::projection_shape::ProjectionShape;

const NATIVE_ERROR_CAPACITY: usize = 512;

/// The devices Core ML may schedule on (`MLComputeUnits`). There is no
/// "Neural Engine only" setting; excluding the GPU is how a GPU fallback is
/// kept from passing as a Neural Engine result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreMlComputeUnits {
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
}

impl CoreMlComputeUnits {
    fn native_code(self) -> i32 {
        match self {
            Self::CpuOnly => 0,
            Self::CpuAndGpu => 1,
            Self::CpuAndNeuralEngine => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputePlacement {
    Unknown,
    Cpu,
    Gpu,
    NeuralEngine,
}

/// `MLComputePlan`'s verdict for the projection layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerPlacement {
    pub preferred: ComputePlacement,
    pub neural_engine_supported: bool,
    pub gpu_supported: bool,
    pub cpu_supported: bool,
    /// Across the whole ML Program, not only the verdict's convolution.
    pub operations_off_neural_engine: i32,
}

#[derive(Debug, Error)]
pub enum CoreMlProjectionError {
    #[error(transparent)]
    Document(#[from] ProjectionDocumentError),
    #[error("could not write the Core ML model document: {0}")]
    ModelFile(#[from] std::io::Error),
    #[error("the model path contains an interior NUL byte")]
    ModelPath,
    #[error("Core ML could not load the projection: {0}")]
    Load(String),
    #[error("expected {expected} fp16 values, got {actual}")]
    ElementCountMismatch { expected: usize, actual: usize },
    #[error("the IOSurface copy failed")]
    SurfaceCopy,
    #[error("the output IOSurface could not be locked for reading")]
    SurfaceLock,
    #[error("Core ML prediction failed: {0}")]
    Prediction(String),
    #[error("the asynchronous prediction could not start")]
    PredictionStart,
    #[error("the asynchronous prediction did not finish within {0:?}")]
    PredictionTimeout(Duration),
}

/// A prediction running on the projection's own dispatch queue while the
/// caller does other work, typically an MLX evaluation on the GPU.
///
/// The guard borrows the projection mutably, so its IOSurfaces cannot be
/// rewritten or read until the prediction is finished; dropping the guard
/// without finishing waits for the prediction rather than abandoning it.
pub struct PendingPrediction<'projection> {
    projection: &'projection mut CoreMlProjection,
    is_finished: bool,
}

impl PendingPrediction<'_> {
    /// Waits up to `timeout` and returns the prediction's own wall time.
    pub fn finish(mut self, timeout: Duration) -> Result<Duration, CoreMlProjectionError> {
        let mut elapsed_nanoseconds = 0_u64;
        let timeout_nanoseconds = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: the handle is live (borrowed from the projection) and the
        // out-pointer is valid and exclusive.
        let status = unsafe {
            astronomical_core_ml_projection_finish_predict(
                self.projection.native_projection.as_ptr(),
                timeout_nanoseconds,
                &mut elapsed_nanoseconds,
            )
        };
        match status {
            0 => {
                self.is_finished = true;
                Ok(Duration::from_nanos(elapsed_nanoseconds))
            }
            -2 => Err(CoreMlProjectionError::PredictionTimeout(timeout)),
            _ => {
                self.is_finished = true;
                Err(CoreMlProjectionError::Prediction(
                    "the asynchronous prediction failed".to_owned(),
                ))
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
            astronomical_core_ml_projection_finish_predict(
                self.projection.native_projection.as_ptr(),
                u64::MAX,
                &mut elapsed_nanoseconds,
            );
        }
    }
}

pub struct CoreMlProjection {
    native_projection: NonNull<NativeCoreMlProjection>,
    shape: ProjectionShape,
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

/// The output IOSurface, locked for reading until this guard drops.
///
/// Rung 4 of issue #161 hands this memory to MLX without a host copy, because
/// a fused projection's output runs to tens of megabytes per chunk. The
/// borrow keeps a new prediction from rewriting the surface while it is read.
pub struct LockedOutputSurface<'projection> {
    projection: &'projection CoreMlProjection,
    base_address: *const c_void,
    bytes_per_row: usize,
}

impl LockedOutputSurface<'_> {
    /// The first fp16 value of `[output_channels][tokens]`, or `None` when
    /// the surface pads its rows and so cannot be read as one contiguous
    /// array. Dereferencing it is valid only while this guard is alive.
    #[must_use]
    pub fn contiguous_fp16_values(&self) -> Option<*const u16> {
        let row_bytes = self.projection.shape.token_count() * size_of::<u16>();
        (self.bytes_per_row == row_bytes).then_some(self.base_address.cast())
    }
}

impl Drop for LockedOutputSurface<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard holds the lock taken in `lock_output_surface` on
        // a handle that its borrow keeps alive, and releases it exactly once.
        unsafe {
            astronomical_core_ml_projection_unlock_output(
                self.projection.native_projection.as_ptr(),
            );
        }
    }
}

impl CoreMlProjection {
    /// Locks the output IOSurface for reading without a host copy.
    pub fn lock_output_surface(&self) -> Result<LockedOutputSurface<'_>, CoreMlProjectionError> {
        let mut base_address = std::ptr::null();
        let mut bytes_per_row = 0_u64;
        // SAFETY: the handle is live and both out-pointers are valid and
        // exclusive; the bridge refuses while a prediction is in flight.
        let status = unsafe {
            astronomical_core_ml_projection_lock_output(
                self.native_projection.as_ptr(),
                &mut base_address,
                &mut bytes_per_row,
            )
        };
        if status != 0 {
            return Err(CoreMlProjectionError::SurfaceLock);
        }
        Ok(LockedOutputSurface {
            projection: self,
            base_address,
            bytes_per_row: bytes_per_row as usize,
        })
    }

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

    fn load_model_at(
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

    /// Removes the compiled model from the temporary directory while the loaded
    /// program stays usable. Release also removes it, but a process that exits
    /// without releasing would otherwise leave one copy of the weights behind per
    /// program, which is the leak this exists to close.
    pub fn discard_compiled_model_files(&mut self) {
        // SAFETY: the handle is valid for this borrow, and the bridge clears its
        // stored path so release cannot remove the directory a second time.
        unsafe {
            astronomical_core_ml_projection_discard_compiled_model(self.native_projection.as_ptr());
        }
    }

    #[must_use]
    pub fn placement(&self) -> LayerPlacement {
        // SAFETY: the handle is live for the lifetime of `self`.
        let native_placement =
            unsafe { astronomical_core_ml_projection_placement(self.native_projection.as_ptr()) };
        layer_placement_from_native(native_placement)
    }

    /// Copies fp16 activations (`[input_channels][tokens]`) into the input
    /// IOSurface. Kept outside `predict` so the timing excludes host copies.
    pub fn write_activations(
        &mut self,
        fp16_activations: &[u16],
    ) -> Result<(), CoreMlProjectionError> {
        check_element_count(
            self.shape.activation_element_count(),
            fp16_activations.len(),
        )?;
        // SAFETY: the slice holds exactly the element count the surface was
        // created for, and the bridge only reads from it.
        let status = unsafe {
            astronomical_core_ml_projection_write_input(
                self.native_projection.as_ptr(),
                fp16_activations.as_ptr(),
                fp16_activations.len() as u64,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(CoreMlProjectionError::SurfaceCopy)
        }
    }

    /// Runs one prediction and returns Core ML's wall time for it alone.
    pub fn predict(&mut self) -> Result<Duration, CoreMlProjectionError> {
        let mut elapsed_nanoseconds = 0_u64;
        let mut native_error = [0 as c_char; NATIVE_ERROR_CAPACITY];
        // SAFETY: the handle is live, both out-pointers are valid and
        // exclusive, and the error buffer is writable for its capacity.
        let status = unsafe {
            astronomical_core_ml_projection_predict(
                self.native_projection.as_ptr(),
                &mut elapsed_nanoseconds,
                native_error.as_mut_ptr(),
                NATIVE_ERROR_CAPACITY as u32,
            )
        };
        if status == 0 {
            Ok(Duration::from_nanos(elapsed_nanoseconds))
        } else {
            Err(CoreMlProjectionError::Prediction(native_error_text(
                &native_error,
            )))
        }
    }

    /// Starts a prediction on the projection's dispatch queue and returns at
    /// once, so the caller's thread is free to drive the GPU meanwhile.
    pub fn begin_prediction(&mut self) -> Result<PendingPrediction<'_>, CoreMlProjectionError> {
        // SAFETY: the handle is live; the bridge refuses a second start while
        // one is in flight, and the returned guard prevents that from Rust.
        let status = unsafe {
            astronomical_core_ml_projection_begin_predict(self.native_projection.as_ptr())
        };
        if status != 0 {
            return Err(CoreMlProjectionError::PredictionStart);
        }
        Ok(PendingPrediction {
            projection: self,
            is_finished: false,
        })
    }

    /// Copies the fp16 result (`[output_channels][tokens]`) out of the output
    /// IOSurface.
    pub fn read_projected(&self) -> Result<Vec<u16>, CoreMlProjectionError> {
        let mut fp16_projected = vec![0_u16; self.shape.projected_element_count()];
        // SAFETY: the buffer holds exactly the surface's element count and
        // the bridge writes only within it.
        let status = unsafe {
            astronomical_core_ml_projection_read_output(
                self.native_projection.as_ptr(),
                fp16_projected.as_mut_ptr(),
                fp16_projected.len() as u64,
            )
        };
        if status == 0 {
            Ok(fp16_projected)
        } else {
            Err(CoreMlProjectionError::SurfaceCopy)
        }
    }
}

impl Drop for CoreMlProjection {
    fn drop(&mut self) {
        // SAFETY: the handle came from the bridge's load function and is
        // released exactly once, here.
        unsafe { astronomical_core_ml_projection_release(self.native_projection.as_ptr()) }
    }
}

impl CoreMlProjectionError {
    fn from_interior_nul(_interior_nul: std::ffi::NulError) -> Self {
        Self::ModelPath
    }
}

fn check_element_count(expected: usize, actual: usize) -> Result<(), CoreMlProjectionError> {
    if expected == actual {
        Ok(())
    } else {
        Err(CoreMlProjectionError::ElementCountMismatch { expected, actual })
    }
}

fn native_error_text(native_error: &[c_char]) -> String {
    // SAFETY: the buffer is zero-initialized and the bridge always leaves its
    // last byte NUL, so a terminator exists inside the slice.
    let message = unsafe { CStr::from_ptr(native_error.as_ptr()) };
    message.to_string_lossy().into_owned()
}

fn placement_from_native_code(native_code: i32) -> ComputePlacement {
    match native_code {
        1 => ComputePlacement::Cpu,
        2 => ComputePlacement::Gpu,
        3 => ComputePlacement::NeuralEngine,
        _ => ComputePlacement::Unknown,
    }
}

fn layer_placement_from_native(native_placement: NativeLayerPlacement) -> LayerPlacement {
    LayerPlacement {
        preferred: placement_from_native_code(native_placement.preferred_placement),
        neural_engine_supported: native_placement.neural_engine_supported != 0,
        gpu_supported: native_placement.gpu_supported != 0,
        cpu_supported: native_placement.cpu_supported != 0,
        operations_off_neural_engine: native_placement.operations_off_neural_engine,
    }
}

#[cfg(test)]
#[path = "core_ml_projection_tests.rs"]
mod tests;

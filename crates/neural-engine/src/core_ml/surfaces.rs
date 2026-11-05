//! The IOSurfaces a loaded model reads from and writes to: host copies in and
//! out, and a locked view of the output that hands it on without a copy.

use std::ffi::c_void;

use super::error::CoreMlProjectionError;
use super::projection::CoreMlProjection;
use crate::core_ml::ffi::{
    astronomical_core_ml_projection_lock_output, astronomical_core_ml_projection_read_output,
    astronomical_core_ml_projection_unlock_output, astronomical_core_ml_projection_write_input,
};

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

fn check_element_count(expected: usize, actual: usize) -> Result<(), CoreMlProjectionError> {
    if expected == actual {
        Ok(())
    } else {
        Err(CoreMlProjectionError::ElementCountMismatch { expected, actual })
    }
}

#[cfg(test)]
#[path = "surfaces_tests.rs"]
mod tests;

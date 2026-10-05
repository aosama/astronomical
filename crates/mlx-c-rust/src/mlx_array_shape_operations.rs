//! Generated MLX-C wrappers for the array shape family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::mlx_bindings_support::{c_string_argument, optional_i32_slice};
use crate::raw;
use crate::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxDtype};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_atleast_1d` on the runtime GPU stream.
    pub fn atleast_1d(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX atleast_1d", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_atleast_1d(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_atleast_2d` on the runtime GPU stream.
    pub fn atleast_2d(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX atleast_2d", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_atleast_2d(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_atleast_3d` on the runtime GPU stream.
    pub fn atleast_3d(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX atleast_3d", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_atleast_3d(output, a.raw(), stream) }
        })
    }

    /// Builds the MLX-C `mlx_broadcast_arrays` outputs on the runtime GPU stream.
    pub fn broadcast_arrays(&self, inputs: &MlxArrayVector) -> Result<MlxArrayVector, MlxCError> {
        self.output_vector_array("apply MLX broadcast_arrays", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_broadcast_arrays(output, inputs.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_copy` on the runtime GPU stream.
    pub fn copy(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX copy", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_copy(output, a.raw(), stream) }
        })
    }

    /// Synchronizes the evaluation of `dependencies` after `inputs` through
    /// MLX-C `mlx_depends`.
    pub fn depends(
        &self,
        inputs: &MlxArrayVector,
        dependencies: &MlxArrayVector,
    ) -> Result<MlxArrayVector, MlxCError> {
        let mut output = MlxArrayVector::empty("apply MLX depends")?;
        // SAFETY: Inputs are live and output is uniquely writable.
        let status =
            unsafe { raw::mlx_depends(output.raw_mut(), inputs.raw(), dependencies.raw()) };
        crate::error::check_status(status, "apply MLX depends")?;
        Ok(output)
    }

    /// Applies MLX-C `mlx_diag` on the runtime GPU stream.
    pub fn diag(&self, a: &MlxArray, k: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX diag", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_diag(output, a.raw(), k, stream) }
        })
    }

    /// Applies MLX-C `mlx_diagonal` on the runtime GPU stream.
    pub fn diagonal(
        &self,
        a: &MlxArray,
        offset: i32,
        axis1: i32,
        axis2: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX diagonal", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_diagonal(output, a.raw(), offset, axis1, axis2, stream) }
        })
    }

    /// Applies MLX-C `mlx_expand_dims_axes` on the runtime GPU stream.
    pub fn expand_dims_axes(&self, a: &MlxArray, axes: &[i32]) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX expand_dims_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_expand_dims_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_flatten` on the runtime GPU stream.
    pub fn flatten(
        &self,
        a: &MlxArray,
        start_axis: i32,
        end_axis: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX flatten", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_flatten(output, a.raw(), start_axis, end_axis, stream) }
        })
    }

    /// Applies MLX-C `mlx_flip` on the runtime GPU stream.
    pub fn flip(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX flip", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_flip(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_flip_axes` on the runtime GPU stream.
    pub fn flip_axes(&self, a: &MlxArray, axes: &[i32]) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX flip_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_flip_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_flip_axis` on the runtime GPU stream.
    pub fn flip_axis(&self, a: &MlxArray, axis: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX flip_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_flip_axis(output, a.raw(), axis, stream) }
        })
    }

    /// Applies MLX-C `mlx_identity` on the runtime GPU stream.
    pub fn identity(&self, n: i32, dtype: MlxDtype) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX identity", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_identity(output, n, dtype.to_raw(), stream) }
        })
    }

    /// Builds the MLX-C `mlx_meshgrid` outputs on the runtime GPU stream.
    pub fn meshgrid(
        &self,
        arrays: &MlxArrayVector,
        sparse: bool,
        indexing: &str,
    ) -> Result<MlxArrayVector, MlxCError> {
        let indexing_argument = c_string_argument(indexing, "meshgrid")?;
        self.output_vector_array("apply MLX meshgrid", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_meshgrid(
                    output,
                    arrays.raw(),
                    sparse,
                    indexing_argument.as_ptr(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_moveaxis` on the runtime GPU stream.
    pub fn moveaxis(
        &self,
        a: &MlxArray,
        source: i32,
        destination: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX moveaxis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_moveaxis(output, a.raw(), source, destination, stream) }
        })
    }

    /// Applies MLX-C `mlx_repeat` on the runtime GPU stream.
    pub fn repeat(&self, arr: &MlxArray, repeats: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX repeat", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_repeat(output, arr.raw(), repeats, stream) }
        })
    }

    /// Applies MLX-C `mlx_roll` on the runtime GPU stream.
    pub fn roll(&self, a: &MlxArray, shift: &[i32]) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX roll", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_roll(
                    output,
                    a.raw(),
                    optional_i32_slice(shift),
                    shift.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_roll_axes` on the runtime GPU stream.
    pub fn roll_axes(
        &self,
        a: &MlxArray,
        shift: &[i32],
        axes: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX roll_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_roll_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(shift),
                    shift.len(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_roll_axis` on the runtime GPU stream.
    pub fn roll_axis(&self, a: &MlxArray, shift: &[i32], axis: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX roll_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_roll_axis(
                    output,
                    a.raw(),
                    optional_i32_slice(shift),
                    shift.len(),
                    axis,
                    stream,
                )
            }
        })
    }

    /// Builds the MLX-C `mlx_split` outputs on the runtime GPU stream.
    pub fn split(
        &self,
        a: &MlxArray,
        num_splits: i32,
        axis: i32,
    ) -> Result<MlxArrayVector, MlxCError> {
        self.output_vector_array("apply MLX split", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_split(output, a.raw(), num_splits, axis, stream) }
        })
    }

    /// Builds the MLX-C `mlx_split_sections` outputs on the runtime GPU stream.
    pub fn split_sections(
        &self,
        a: &MlxArray,
        indices: &[i32],
        axis: i32,
    ) -> Result<MlxArrayVector, MlxCError> {
        self.output_vector_array("apply MLX split_sections", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_split_sections(
                    output,
                    a.raw(),
                    optional_i32_slice(indices),
                    indices.len(),
                    axis,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_squeeze` on the runtime GPU stream.
    pub fn squeeze(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX squeeze", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_squeeze(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_squeeze_axes` on the runtime GPU stream.
    pub fn squeeze_axes(&self, a: &MlxArray, axes: &[i32]) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX squeeze_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_squeeze_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_swapaxes` on the runtime GPU stream.
    pub fn swapaxes(&self, a: &MlxArray, axis1: i32, axis2: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX swapaxes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_swapaxes(output, a.raw(), axis1, axis2, stream) }
        })
    }

    /// Applies MLX-C `mlx_tile` on the runtime GPU stream.
    pub fn tile(&self, arr: &MlxArray, reps: &[i32]) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX tile", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_tile(
                    output,
                    arr.raw(),
                    optional_i32_slice(reps),
                    reps.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_trace` on the runtime GPU stream.
    pub fn trace(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX trace", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_trace(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_trace_axes` on the runtime GPU stream.
    pub fn trace_axes(
        &self,
        a: &MlxArray,
        offset: i32,
        axis1: i32,
        axis2: i32,
        dtype: MlxDtype,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX trace_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_trace_axes(
                    output,
                    a.raw(),
                    offset,
                    axis1,
                    axis2,
                    dtype.to_raw(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_transpose` on the runtime GPU stream.
    pub fn transpose(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX transpose", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_transpose(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_unflatten` on the runtime GPU stream.
    pub fn unflatten(&self, a: &MlxArray, axis: i32, shape: &[i32]) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX unflatten", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_unflatten(
                    output,
                    a.raw(),
                    axis,
                    optional_i32_slice(shape),
                    shape.len(),
                    stream,
                )
            }
        })
    }

    /// Builds the MLX-C `mlx_unstack` outputs on the runtime GPU stream.
    pub fn unstack(&self, a: &MlxArray) -> Result<MlxArrayVector, MlxCError> {
        self.output_vector_array("apply MLX unstack", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_unstack(output, a.raw(), stream) }
        })
    }

    /// Builds the MLX-C `mlx_unstack_axis` outputs on the runtime GPU stream.
    pub fn unstack_axis(&self, a: &MlxArray, axis: i32) -> Result<MlxArrayVector, MlxCError> {
        self.output_vector_array("apply MLX unstack_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_unstack_axis(output, a.raw(), axis, stream) }
        })
    }

    /// Applies MLX-C `mlx_view` on the runtime GPU stream.
    pub fn view(&self, a: &MlxArray, dtype: MlxDtype) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX view", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_view(output, a.raw(), dtype.to_raw(), stream) }
        })
    }
}

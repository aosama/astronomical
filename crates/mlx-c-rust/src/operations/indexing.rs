//! Generated MLX-C wrappers for the indexing family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::bindings::support::{c_string_argument, optional_i32_slice};
use crate::raw;
use crate::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_einsum` on the runtime GPU stream.
    pub fn einsum(
        &self,
        subscripts: &str,
        operands: &MlxArrayVector,
    ) -> Result<MlxArray, MlxCError> {
        let subscripts_argument = c_string_argument(subscripts, "einsum")?;
        self.output_array("apply MLX einsum", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_einsum(output, subscripts_argument.as_ptr(), operands.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_gather` on the runtime GPU stream.
    pub fn gather(
        &self,
        a: &MlxArray,
        indices: &MlxArrayVector,
        axes: &[i32],
        slice_sizes: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX gather", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_gather(
                    output,
                    a.raw(),
                    indices.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    optional_i32_slice(slice_sizes),
                    slice_sizes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_gather_single` on the runtime GPU stream.
    pub fn gather_single(
        &self,
        a: &MlxArray,
        indices: &MlxArray,
        axis: i32,
        slice_sizes: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX gather_single", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_gather_single(
                    output,
                    a.raw(),
                    indices.raw(),
                    axis,
                    optional_i32_slice(slice_sizes),
                    slice_sizes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_masked_scatter` on the runtime GPU stream.
    pub fn masked_scatter(
        &self,
        a: &MlxArray,
        mask: &MlxArray,
        src: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX masked_scatter", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_masked_scatter(output, a.raw(), mask.raw(), src.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_scatter` on the runtime GPU stream.
    pub fn scatter(
        &self,
        a: &MlxArray,
        indices: &MlxArrayVector,
        updates: &MlxArray,
        axes: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter(
                    output,
                    a.raw(),
                    indices.raw(),
                    updates.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_scatter_add` on the runtime GPU stream.
    pub fn scatter_add_along_axes(
        &self,
        a: &MlxArray,
        indices: &MlxArrayVector,
        updates: &MlxArray,
        axes: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter_add_along_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_add(
                    output,
                    a.raw(),
                    indices.raw(),
                    updates.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_scatter_add_axis` on the runtime GPU stream.
    pub fn scatter_add_axis(
        &self,
        a: &MlxArray,
        indices: &MlxArray,
        values: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter_add_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_add_axis(
                    output,
                    a.raw(),
                    indices.raw(),
                    values.raw(),
                    axis,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_scatter_max` on the runtime GPU stream.
    pub fn scatter_max(
        &self,
        a: &MlxArray,
        indices: &MlxArrayVector,
        updates: &MlxArray,
        axes: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter_max", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_max(
                    output,
                    a.raw(),
                    indices.raw(),
                    updates.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_scatter_max_single` on the runtime GPU stream.
    pub fn scatter_max_single(
        &self,
        a: &MlxArray,
        indices: &MlxArray,
        updates: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter_max_single", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_max_single(
                    output,
                    a.raw(),
                    indices.raw(),
                    updates.raw(),
                    axis,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_scatter_min` on the runtime GPU stream.
    pub fn scatter_min(
        &self,
        a: &MlxArray,
        indices: &MlxArrayVector,
        updates: &MlxArray,
        axes: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter_min", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_min(
                    output,
                    a.raw(),
                    indices.raw(),
                    updates.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_scatter_min_single` on the runtime GPU stream.
    pub fn scatter_min_single(
        &self,
        a: &MlxArray,
        indices: &MlxArray,
        updates: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter_min_single", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_min_single(
                    output,
                    a.raw(),
                    indices.raw(),
                    updates.raw(),
                    axis,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_scatter_prod` on the runtime GPU stream.
    pub fn scatter_prod(
        &self,
        a: &MlxArray,
        indices: &MlxArrayVector,
        updates: &MlxArray,
        axes: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter_prod", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_prod(
                    output,
                    a.raw(),
                    indices.raw(),
                    updates.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_scatter_prod_single` on the runtime GPU stream.
    pub fn scatter_prod_single(
        &self,
        a: &MlxArray,
        indices: &MlxArray,
        updates: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter_prod_single", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_prod_single(
                    output,
                    a.raw(),
                    indices.raw(),
                    updates.raw(),
                    axis,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_scatter_single` on the runtime GPU stream.
    pub fn scatter_single(
        &self,
        a: &MlxArray,
        indices: &MlxArray,
        updates: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX scatter_single", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_single(output, a.raw(), indices.raw(), updates.raw(), axis, stream)
            }
        })
    }

    /// Applies MLX-C `mlx_searchsorted` on the runtime GPU stream.
    pub fn searchsorted(
        &self,
        sorted_sequence: &MlxArray,
        values: &MlxArray,
        side: &str,
    ) -> Result<MlxArray, MlxCError> {
        let side_argument = c_string_argument(side, "searchsorted")?;
        self.output_array("apply MLX searchsorted", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_searchsorted(
                    output,
                    sorted_sequence.raw(),
                    values.raw(),
                    side_argument.as_ptr(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_slice_dynamic` on the runtime GPU stream.
    pub fn slice_dynamic(
        &self,
        a: &MlxArray,
        start: &MlxArray,
        axes: &[i32],
        slice_size: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX slice_dynamic", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_slice_dynamic(
                    output,
                    a.raw(),
                    start.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    optional_i32_slice(slice_size),
                    slice_size.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_slice_update_add` on the runtime GPU stream.
    pub fn slice_update_add(
        &self,
        src: &MlxArray,
        update: &MlxArray,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX slice_update_add", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_slice_update_add(
                    output,
                    src.raw(),
                    update.raw(),
                    optional_i32_slice(start),
                    start.len(),
                    optional_i32_slice(stop),
                    stop.len(),
                    optional_i32_slice(strides),
                    strides.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_slice_update_dynamic` on the runtime GPU stream.
    pub fn slice_update_dynamic(
        &self,
        src: &MlxArray,
        update: &MlxArray,
        start: &MlxArray,
        axes: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX slice_update_dynamic", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_slice_update_dynamic(
                    output,
                    src.raw(),
                    update.raw(),
                    start.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_slice_update_max` on the runtime GPU stream.
    pub fn slice_update_max(
        &self,
        src: &MlxArray,
        update: &MlxArray,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX slice_update_max", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_slice_update_max(
                    output,
                    src.raw(),
                    update.raw(),
                    optional_i32_slice(start),
                    start.len(),
                    optional_i32_slice(stop),
                    stop.len(),
                    optional_i32_slice(strides),
                    strides.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_slice_update_min` on the runtime GPU stream.
    pub fn slice_update_min(
        &self,
        src: &MlxArray,
        update: &MlxArray,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX slice_update_min", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_slice_update_min(
                    output,
                    src.raw(),
                    update.raw(),
                    optional_i32_slice(start),
                    start.len(),
                    optional_i32_slice(stop),
                    stop.len(),
                    optional_i32_slice(strides),
                    strides.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_slice_update_prod` on the runtime GPU stream.
    pub fn slice_update_prod(
        &self,
        src: &MlxArray,
        update: &MlxArray,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX slice_update_prod", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_slice_update_prod(
                    output,
                    src.raw(),
                    update.raw(),
                    optional_i32_slice(start),
                    start.len(),
                    optional_i32_slice(stop),
                    stop.len(),
                    optional_i32_slice(strides),
                    strides.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_take` on the runtime GPU stream.
    pub fn take(&self, a: &MlxArray, indices: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX take", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_take(output, a.raw(), indices.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_tensordot` on the runtime GPU stream.
    pub fn tensordot(
        &self,
        a: &MlxArray,
        b: &MlxArray,
        axes_a: &[i32],
        axes_b: &[i32],
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX tensordot", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_tensordot(
                    output,
                    a.raw(),
                    b.raw(),
                    optional_i32_slice(axes_a),
                    axes_a.len(),
                    optional_i32_slice(axes_b),
                    axes_b.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_tensordot_axis` on the runtime GPU stream.
    pub fn tensordot_axis(
        &self,
        a: &MlxArray,
        b: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX tensordot_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_tensordot_axis(output, a.raw(), b.raw(), axis, stream) }
        })
    }

    /// Applies MLX-C `mlx_vecdot` on the runtime GPU stream.
    pub fn vecdot(&self, a: &MlxArray, b: &MlxArray, axis: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX vecdot", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_vecdot(output, a.raw(), b.raw(), axis, stream) }
        })
    }
}

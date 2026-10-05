//! Generated MLX-C wrappers for the cumulative scan family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::mlx_bindings_support::raw_optional_dtype;
use crate::raw;
use crate::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_cummax` on the runtime GPU stream.
    pub fn cummax(
        &self,
        a: &MlxArray,
        reverse: bool,
        inclusive: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX cummax", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_cummax(output, a.raw(), reverse, inclusive, stream) }
        })
    }

    /// Applies MLX-C `mlx_cummax_axis` on the runtime GPU stream.
    pub fn cummax_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        reverse: bool,
        inclusive: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX cummax_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_cummax_axis(output, a.raw(), axis, reverse, inclusive, stream) }
        })
    }

    /// Applies MLX-C `mlx_cummin` on the runtime GPU stream.
    pub fn cummin(
        &self,
        a: &MlxArray,
        reverse: bool,
        inclusive: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX cummin", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_cummin(output, a.raw(), reverse, inclusive, stream) }
        })
    }

    /// Applies MLX-C `mlx_cummin_axis` on the runtime GPU stream.
    pub fn cummin_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        reverse: bool,
        inclusive: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX cummin_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_cummin_axis(output, a.raw(), axis, reverse, inclusive, stream) }
        })
    }

    /// Applies MLX-C `mlx_cumprod` on the runtime GPU stream.
    pub fn cumprod(
        &self,
        a: &MlxArray,
        reverse: bool,
        inclusive: bool,
        dtype: Option<MlxDtype>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX cumprod", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_cumprod(
                    output,
                    a.raw(),
                    reverse,
                    inclusive,
                    raw_optional_dtype(dtype),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_cumprod_axis` on the runtime GPU stream.
    pub fn cumprod_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        reverse: bool,
        inclusive: bool,
        dtype: Option<MlxDtype>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX cumprod_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_cumprod_axis(
                    output,
                    a.raw(),
                    axis,
                    reverse,
                    inclusive,
                    raw_optional_dtype(dtype),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_cumsum` on the runtime GPU stream.
    pub fn cumsum_across_all_axes(
        &self,
        a: &MlxArray,
        reverse: bool,
        inclusive: bool,
        dtype: Option<MlxDtype>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX cumsum_across_all_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_cumsum(
                    output,
                    a.raw(),
                    reverse,
                    inclusive,
                    raw_optional_dtype(dtype),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_logcumsumexp` on the runtime GPU stream.
    pub fn logcumsumexp(
        &self,
        a: &MlxArray,
        reverse: bool,
        inclusive: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX logcumsumexp", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_logcumsumexp(output, a.raw(), reverse, inclusive, stream) }
        })
    }

    /// Applies MLX-C `mlx_logcumsumexp_axis` on the runtime GPU stream.
    pub fn logcumsumexp_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        reverse: bool,
        inclusive: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX logcumsumexp_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_logcumsumexp_axis(output, a.raw(), axis, reverse, inclusive, stream) }
        })
    }
}

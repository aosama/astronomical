//! Generated MLX-C wrappers for the reduction family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::mlx_bindings_support::optional_i32_slice;
use crate::raw;
use crate::{MlxArray, MlxBindingsContext, MlxCError};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_all` on the runtime GPU stream.
    pub fn all(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX all", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_all(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_all_axes` on the runtime GPU stream.
    pub fn all_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX all_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_all_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_all_axis` on the runtime GPU stream.
    pub fn all_axis(&self, a: &MlxArray, axis: i32, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX all_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_all_axis(output, a.raw(), axis, keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_any` on the runtime GPU stream.
    pub fn any(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX any", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_any(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_any_axes` on the runtime GPU stream.
    pub fn any_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX any_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_any_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_any_axis` on the runtime GPU stream.
    pub fn any_axis(&self, a: &MlxArray, axis: i32, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX any_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_any_axis(output, a.raw(), axis, keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_argmax` on the runtime GPU stream.
    pub fn argmax(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX argmax", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_argmax(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_argmin` on the runtime GPU stream.
    pub fn argmin(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX argmin", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_argmin(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_argmin_axis` on the runtime GPU stream.
    pub fn argmin_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX argmin_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_argmin_axis(output, a.raw(), axis, keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_count_nonzero` on the runtime GPU stream.
    pub fn count_nonzero(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX count_nonzero", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_count_nonzero(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_count_nonzero_axes` on the runtime GPU stream.
    pub fn count_nonzero_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX count_nonzero_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_count_nonzero_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_count_nonzero_axis` on the runtime GPU stream.
    pub fn count_nonzero_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX count_nonzero_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_count_nonzero_axis(output, a.raw(), axis, keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_logsumexp` on the runtime GPU stream.
    pub fn logsumexp(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX logsumexp", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_logsumexp(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_logsumexp_axes` on the runtime GPU stream.
    pub fn logsumexp_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX logsumexp_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_logsumexp_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_logsumexp_axis` on the runtime GPU stream.
    pub fn logsumexp_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX logsumexp_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_logsumexp_axis(output, a.raw(), axis, keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_max` on the runtime GPU stream.
    pub fn max(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX max", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_max(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_max_axes` on the runtime GPU stream.
    pub fn max_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX max_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_max_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_mean` on the runtime GPU stream.
    pub fn mean(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX mean", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_mean(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_mean_axes` on the runtime GPU stream.
    pub fn mean_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX mean_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_mean_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_mean_axis` on the runtime GPU stream.
    pub fn mean_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX mean_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_mean_axis(output, a.raw(), axis, keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_median` on the runtime GPU stream.
    pub fn median(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX median", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_median(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_median_axes` on the runtime GPU stream.
    pub fn median_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX median_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_median_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_median_axis` on the runtime GPU stream.
    pub fn median_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX median_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_median_axis(output, a.raw(), axis, keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_min` on the runtime GPU stream.
    pub fn min(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX min", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_min(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_min_axes` on the runtime GPU stream.
    pub fn min_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX min_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_min_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_min_axis` on the runtime GPU stream.
    pub fn min_axis(&self, a: &MlxArray, axis: i32, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX min_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_min_axis(output, a.raw(), axis, keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_prod` on the runtime GPU stream.
    pub fn prod(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX prod", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_prod(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_prod_axes` on the runtime GPU stream.
    pub fn prod_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX prod_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_prod_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_prod_axis` on the runtime GPU stream.
    pub fn prod_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX prod_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_prod_axis(output, a.raw(), axis, keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_softmax` on the runtime GPU stream.
    pub fn softmax(&self, a: &MlxArray, precise: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX softmax", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_softmax(output, a.raw(), precise, stream) }
        })
    }

    /// Applies MLX-C `mlx_softmax_axes` on the runtime GPU stream.
    pub fn softmax_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        precise: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX softmax_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_softmax_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    precise,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_std` on the runtime GPU stream.
    pub fn std(&self, a: &MlxArray, keepdims: bool, ddof: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX std", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_std(output, a.raw(), keepdims, ddof, stream) }
        })
    }

    /// Applies MLX-C `mlx_std_axes` on the runtime GPU stream.
    pub fn std_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
        ddof: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX std_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_std_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    ddof,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_std_axis` on the runtime GPU stream.
    pub fn std_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        keepdims: bool,
        ddof: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX std_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_std_axis(output, a.raw(), axis, keepdims, ddof, stream) }
        })
    }

    /// Applies MLX-C `mlx_sum` on the runtime GPU stream.
    pub fn sum(&self, a: &MlxArray, keepdims: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX sum", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_sum(output, a.raw(), keepdims, stream) }
        })
    }

    /// Applies MLX-C `mlx_sum_axes` on the runtime GPU stream.
    pub fn sum_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX sum_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_sum_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_var` on the runtime GPU stream.
    pub fn var(&self, a: &MlxArray, keepdims: bool, ddof: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX var", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_var(output, a.raw(), keepdims, ddof, stream) }
        })
    }

    /// Applies MLX-C `mlx_var_axes` on the runtime GPU stream.
    pub fn var_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
        ddof: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX var_axes", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_var_axes(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    keepdims,
                    ddof,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_var_axis` on the runtime GPU stream.
    pub fn var_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        keepdims: bool,
        ddof: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX var_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_var_axis(output, a.raw(), axis, keepdims, ddof, stream) }
        })
    }
}

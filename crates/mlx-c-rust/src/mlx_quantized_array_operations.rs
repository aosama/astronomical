//! Generated MLX-C wrappers for the quantized arrays family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::mlx_bindings_support::{c_string_argument, raw_optional_int};
use crate::raw;
use crate::{MlxArray, MlxBindingsContext, MlxCError};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_block_masked_mm` on the runtime GPU stream.
    pub fn block_masked_mm(
        &self,
        a: &MlxArray,
        b: &MlxArray,
        block_size: i32,
        mask_out: Option<&MlxArray>,
        mask_lhs: Option<&MlxArray>,
        mask_rhs: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX block_masked_mm", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_block_masked_mm(
                    output,
                    a.raw(),
                    b.raw(),
                    block_size,
                    mask_out.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    mask_lhs.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    mask_rhs.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_gather_qqmm` on the runtime GPU stream.
    pub fn gather_qqmm(
        &self,
        x: &MlxArray,
        w: &MlxArray,
        scales_w: Option<&MlxArray>,
        lhs_indices: Option<&MlxArray>,
        rhs_indices: Option<&MlxArray>,
        group_size: Option<i32>,
        bits: Option<i32>,
        mode: &str,
        global_scale_x: Option<&MlxArray>,
        global_scale_w: Option<&MlxArray>,
        sorted_indices: bool,
    ) -> Result<MlxArray, MlxCError> {
        let mode_argument = c_string_argument(mode, "gather_qqmm")?;
        self.output_array("apply MLX gather_qqmm", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_gather_qqmm(
                    output,
                    x.raw(),
                    w.raw(),
                    scales_w.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    lhs_indices.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    rhs_indices.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    raw_optional_int(group_size),
                    raw_optional_int(bits),
                    mode_argument.as_ptr(),
                    global_scale_x.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    global_scale_w.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    sorted_indices,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_qqmm` on the runtime GPU stream.
    pub fn qqmm(
        &self,
        x: &MlxArray,
        w: &MlxArray,
        w_scales: Option<&MlxArray>,
        group_size: Option<i32>,
        bits: Option<i32>,
        mode: &str,
        global_scale_x: Option<&MlxArray>,
        global_scale_w: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        let mode_argument = c_string_argument(mode, "qqmm")?;
        self.output_array("apply MLX qqmm", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_qqmm(
                    output,
                    x.raw(),
                    w.raw(),
                    w_scales.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    raw_optional_int(group_size),
                    raw_optional_int(bits),
                    mode_argument.as_ptr(),
                    global_scale_x.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    global_scale_w.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }
}

//! Generated MLX-C wrappers for the general convolution family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::mlx_bindings_support::optional_i32_slice;
use crate::raw;
use crate::{MlxArray, MlxBindingsContext, MlxCError};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_conv_general` on the runtime GPU stream.
    pub fn conv_general(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        stride: &[i32],
        padding_lo: &[i32],
        padding_hi: &[i32],
        kernel_dilation: &[i32],
        input_dilation: &[i32],
        groups: i32,
        flip: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX conv_general", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_conv_general(
                    output,
                    input.raw(),
                    weight.raw(),
                    optional_i32_slice(stride),
                    stride.len(),
                    optional_i32_slice(padding_lo),
                    padding_lo.len(),
                    optional_i32_slice(padding_hi),
                    padding_hi.len(),
                    optional_i32_slice(kernel_dilation),
                    kernel_dilation.len(),
                    optional_i32_slice(input_dilation),
                    input_dilation.len(),
                    groups,
                    flip,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_conv_transpose1d` on the runtime GPU stream.
    pub fn conv_transpose1d(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        stride: i32,
        padding: i32,
        dilation: i32,
        output_padding: i32,
        groups: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX conv_transpose1d", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_conv_transpose1d(
                    output,
                    input.raw(),
                    weight.raw(),
                    stride,
                    padding,
                    dilation,
                    output_padding,
                    groups,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_conv_transpose2d` on the runtime GPU stream.
    pub fn conv_transpose2d(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        stride_0: i32,
        stride_1: i32,
        padding_0: i32,
        padding_1: i32,
        dilation_0: i32,
        dilation_1: i32,
        output_padding_0: i32,
        output_padding_1: i32,
        groups: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX conv_transpose2d", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_conv_transpose2d(
                    output,
                    input.raw(),
                    weight.raw(),
                    stride_0,
                    stride_1,
                    padding_0,
                    padding_1,
                    dilation_0,
                    dilation_1,
                    output_padding_0,
                    output_padding_1,
                    groups,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_conv_transpose3d` on the runtime GPU stream.
    pub fn conv_transpose3d(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        stride_0: i32,
        stride_1: i32,
        stride_2: i32,
        padding_0: i32,
        padding_1: i32,
        padding_2: i32,
        dilation_0: i32,
        dilation_1: i32,
        dilation_2: i32,
        output_padding_0: i32,
        output_padding_1: i32,
        output_padding_2: i32,
        groups: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX conv_transpose3d", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_conv_transpose3d(
                    output,
                    input.raw(),
                    weight.raw(),
                    stride_0,
                    stride_1,
                    stride_2,
                    padding_0,
                    padding_1,
                    padding_2,
                    dilation_0,
                    dilation_1,
                    dilation_2,
                    output_padding_0,
                    output_padding_1,
                    output_padding_2,
                    groups,
                    stream,
                )
            }
        })
    }
}

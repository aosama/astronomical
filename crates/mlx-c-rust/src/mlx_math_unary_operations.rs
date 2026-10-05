//! Generated MLX-C wrappers for the unary math family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::mlx_bindings_support::{
    c_string_argument, optional_i32_slice, optional_i64_slice, raw_optional_float,
};
use crate::raw;
use crate::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxDtype};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_abs` on the runtime GPU stream.
    pub fn abs(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX abs", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_abs(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_arccos` on the runtime GPU stream.
    pub fn arccos(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX arccos", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_arccos(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_arccosh` on the runtime GPU stream.
    pub fn arccosh(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX arccosh", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_arccosh(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_arcsin` on the runtime GPU stream.
    pub fn arcsin(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX arcsin", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_arcsin(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_arcsinh` on the runtime GPU stream.
    pub fn arcsinh(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX arcsinh", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_arcsinh(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_arctan` on the runtime GPU stream.
    pub fn arctan(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX arctan", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_arctan(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_arctanh` on the runtime GPU stream.
    pub fn arctanh(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX arctanh", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_arctanh(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_argpartition` on the runtime GPU stream.
    pub fn argpartition(&self, a: &MlxArray, kth: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX argpartition", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_argpartition(output, a.raw(), kth, stream) }
        })
    }

    /// Applies MLX-C `mlx_argsort` on the runtime GPU stream.
    pub fn argsort(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX argsort", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_argsort(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_array_equal` on the runtime GPU stream.
    pub fn array_equal(
        &self,
        a: &MlxArray,
        b: &MlxArray,
        equal_nan: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX array_equal", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_array_equal(output, a.raw(), b.raw(), equal_nan, stream) }
        })
    }

    /// Applies MLX-C `mlx_as_strided` on the runtime GPU stream.
    pub fn as_strided(
        &self,
        a: &MlxArray,
        shape: &[i32],
        strides: &[i64],
        offset: usize,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX as_strided", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_as_strided(
                    output,
                    a.raw(),
                    optional_i32_slice(shape),
                    shape.len(),
                    optional_i64_slice(strides),
                    strides.len(),
                    offset,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_ceil` on the runtime GPU stream.
    pub fn ceil(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX ceil", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_ceil(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_concatenate` on the runtime GPU stream.
    pub fn concatenate(&self, arrays: &MlxArrayVector) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX concatenate", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_concatenate(output, arrays.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_conjugate` on the runtime GPU stream.
    pub fn conjugate(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX conjugate", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_conjugate(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_cosh` on the runtime GPU stream.
    pub fn cosh(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX cosh", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_cosh(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_degrees` on the runtime GPU stream.
    pub fn degrees(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX degrees", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_degrees(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_diff` on the runtime GPU stream.
    pub fn diff(&self, a: &MlxArray, n: i32, axis: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX diff", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_diff(output, a.raw(), n, axis, stream) }
        })
    }

    /// Applies MLX-C `mlx_erfinv` on the runtime GPU stream.
    pub fn erfinv(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX erfinv", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_erfinv(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_expm1` on the runtime GPU stream.
    pub fn expm1(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX expm1", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_expm1(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_floor` on the runtime GPU stream.
    pub fn floor(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX floor", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_floor(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_from_fp8` on the runtime GPU stream.
    pub fn from_fp8(&self, x: &MlxArray, dtype: MlxDtype) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX from_fp8", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_from_fp8(output, x.raw(), dtype.to_raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_imag` on the runtime GPU stream.
    pub fn imag(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX imag", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_imag(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_isfinite` on the runtime GPU stream.
    pub fn isfinite(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX isfinite", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_isfinite(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_isinf` on the runtime GPU stream.
    pub fn isinf(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX isinf", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_isinf(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_isnan` on the runtime GPU stream.
    pub fn isnan(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX isnan", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_isnan(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_isneginf` on the runtime GPU stream.
    pub fn isneginf(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX isneginf", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_isneginf(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_isposinf` on the runtime GPU stream.
    pub fn isposinf(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX isposinf", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_isposinf(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_left_shift` on the runtime GPU stream.
    pub fn left_shift(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX left_shift", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_left_shift(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_log` on the runtime GPU stream.
    pub fn log(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX log", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_log(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_log10` on the runtime GPU stream.
    pub fn log10(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX log10", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_log10(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_log2` on the runtime GPU stream.
    pub fn log2(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX log2", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_log2(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_nan_to_num` on the runtime GPU stream.
    pub fn nan_to_num(
        &self,
        a: &MlxArray,
        nan: f32,
        posinf: Option<f32>,
        neginf: Option<f32>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX nan_to_num", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_nan_to_num(
                    output,
                    a.raw(),
                    nan,
                    raw_optional_float(posinf),
                    raw_optional_float(neginf),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_not_equal` on the runtime GPU stream.
    pub fn not_equal(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX not_equal", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_not_equal(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_number_of_elements` on the runtime GPU stream.
    pub fn number_of_elements(
        &self,
        a: &MlxArray,
        axes: &[i32],
        inverted: bool,
        dtype: MlxDtype,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX number_of_elements", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_number_of_elements(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    inverted,
                    dtype.to_raw(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_pad_symmetric` on the runtime GPU stream.
    pub fn pad_symmetric(
        &self,
        a: &MlxArray,
        pad_width: i32,
        pad_value: &MlxArray,
        mode: &str,
    ) -> Result<MlxArray, MlxCError> {
        let mode_argument = c_string_argument(mode, "pad_symmetric")?;
        self.output_array("apply MLX pad_symmetric", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_pad_symmetric(
                    output,
                    a.raw(),
                    pad_width,
                    pad_value.raw(),
                    mode_argument.as_ptr(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_partition` on the runtime GPU stream.
    pub fn partition(&self, a: &MlxArray, kth: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX partition", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_partition(output, a.raw(), kth, stream) }
        })
    }

    /// Applies MLX-C `mlx_partition_axis` on the runtime GPU stream.
    pub fn partition_axis(&self, a: &MlxArray, kth: i32, axis: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX partition_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_partition_axis(output, a.raw(), kth, axis, stream) }
        })
    }

    /// Applies MLX-C `mlx_positive` on the runtime GPU stream.
    pub fn positive(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX positive", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_positive(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_radians` on the runtime GPU stream.
    pub fn radians(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX radians", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_radians(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_real` on the runtime GPU stream.
    pub fn real(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX real", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_real(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_reciprocal` on the runtime GPU stream.
    pub fn reciprocal(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX reciprocal", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_reciprocal(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_right_shift` on the runtime GPU stream.
    pub fn right_shift(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX right_shift", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_right_shift(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_round` on the runtime GPU stream.
    pub fn round(&self, a: &MlxArray, decimals: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX round", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_round(output, a.raw(), decimals, stream) }
        })
    }

    /// Applies MLX-C `mlx_rsqrt` on the runtime GPU stream.
    pub fn rsqrt(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX rsqrt", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_rsqrt(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_segmented_mm` on the runtime GPU stream.
    pub fn segmented_mm(
        &self,
        a: &MlxArray,
        b: &MlxArray,
        segments: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX segmented_mm", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_segmented_mm(output, a.raw(), b.raw(), segments.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_sign` on the runtime GPU stream.
    pub fn sign(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX sign", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_sign(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_sinh` on the runtime GPU stream.
    pub fn sinh(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX sinh", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_sinh(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_sort` on the runtime GPU stream.
    pub fn sort(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX sort", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_sort(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_sort_axis` on the runtime GPU stream.
    pub fn sort_axis(&self, a: &MlxArray, axis: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX sort_axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_sort_axis(output, a.raw(), axis, stream) }
        })
    }

    /// Applies MLX-C `mlx_square` on the runtime GPU stream.
    pub fn square(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX square", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_square(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_stack` on the runtime GPU stream.
    pub fn stack(&self, arrays: &MlxArrayVector) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX stack", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_stack(output, arrays.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_stop_gradient` on the runtime GPU stream.
    pub fn stop_gradient(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX stop_gradient", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_stop_gradient(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_tan` on the runtime GPU stream.
    pub fn tan(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX tan", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_tan(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_to_fp8` on the runtime GPU stream.
    pub fn to_fp8(&self, x: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX to_fp8", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_to_fp8(output, x.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_topk` on the runtime GPU stream.
    pub fn topk(&self, a: &MlxArray, k: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX topk", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_topk(output, a.raw(), k, stream) }
        })
    }

    /// Applies MLX-C `mlx_trunc` on the runtime GPU stream.
    pub fn trunc(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX trunc", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_trunc(output, a.raw(), stream) }
        })
    }
}

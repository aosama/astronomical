//! Generated MLX-C wrappers for the array creation family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::bindings::support::{optional_i32_slice, raw_optional_float};
use crate::raw;
use crate::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_bartlett` on the runtime GPU stream.
    pub fn bartlett(&self, size: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX bartlett", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_bartlett(output, size, stream) }
        })
    }

    /// Applies MLX-C `mlx_blackman` on the runtime GPU stream.
    pub fn blackman(&self, size: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX blackman", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_blackman(output, size, stream) }
        })
    }

    /// Applies MLX-C `mlx_eye` on the runtime GPU stream.
    pub fn eye(&self, n: i32, m: i32, k: i32, dtype: MlxDtype) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX eye", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_eye(output, n, m, k, dtype.to_raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_full_like` on the runtime GPU stream.
    pub fn full_like(
        &self,
        a: &MlxArray,
        vals: &MlxArray,
        dtype: MlxDtype,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX full_like", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_full_like(output, a.raw(), vals.raw(), dtype.to_raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_hadamard_transform` on the runtime GPU stream.
    pub fn hadamard_transform(
        &self,
        a: &MlxArray,
        scale: Option<f32>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX hadamard_transform", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_hadamard_transform(output, a.raw(), raw_optional_float(scale), stream)
            }
        })
    }

    /// Applies MLX-C `mlx_hamming` on the runtime GPU stream.
    pub fn hamming(&self, size: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX hamming", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_hamming(output, size, stream) }
        })
    }

    /// Applies MLX-C `mlx_hanning` on the runtime GPU stream.
    pub fn hanning(&self, size: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX hanning", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_hanning(output, size, stream) }
        })
    }

    /// Applies MLX-C `mlx_linspace` on the runtime GPU stream.
    pub fn linspace(
        &self,
        start: f64,
        stop: f64,
        num: i32,
        dtype: MlxDtype,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linspace", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linspace(output, start, stop, num, dtype.to_raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_linspace_endpoint` on the runtime GPU stream.
    pub fn linspace_endpoint(
        &self,
        start: f64,
        stop: f64,
        num: i32,
        endpoint: bool,
        dtype: MlxDtype,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linspace_endpoint", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_linspace_endpoint(
                    output,
                    start,
                    stop,
                    num,
                    endpoint,
                    dtype.to_raw(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_ones` on the runtime GPU stream.
    pub fn ones(&self, shape: &[i32], dtype: MlxDtype) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX ones", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_ones(
                    output,
                    optional_i32_slice(shape),
                    shape.len(),
                    dtype.to_raw(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_ones_like` on the runtime GPU stream.
    pub fn ones_like(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX ones_like", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_ones_like(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_ones_like_dtype` on the runtime GPU stream.
    pub fn ones_like_dtype(&self, a: &MlxArray, dtype: MlxDtype) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX ones_like_dtype", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_ones_like_dtype(output, a.raw(), dtype.to_raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_tri` on the runtime GPU stream.
    pub fn tri(&self, n: i32, m: i32, k: i32, dtype: MlxDtype) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX tri", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_tri(output, n, m, k, dtype.to_raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_tril` on the runtime GPU stream.
    pub fn tril(&self, x: &MlxArray, k: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX tril", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_tril(output, x.raw(), k, stream) }
        })
    }

    /// Applies MLX-C `mlx_triu` on the runtime GPU stream.
    pub fn triu(&self, x: &MlxArray, k: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX triu", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_triu(output, x.raw(), k, stream) }
        })
    }

    /// Applies MLX-C `mlx_zeros_like` on the runtime GPU stream.
    pub fn zeros_like(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX zeros_like", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_zeros_like(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_zeros_like_dtype` on the runtime GPU stream.
    pub fn zeros_like_dtype(&self, a: &MlxArray, dtype: MlxDtype) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX zeros_like_dtype", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_zeros_like_dtype(output, a.raw(), dtype.to_raw(), stream) }
        })
    }
}

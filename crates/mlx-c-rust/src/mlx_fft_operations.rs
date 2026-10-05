//! Generated MLX-C wrappers for the FFT family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::mlx_bindings_support::optional_i32_slice;
use crate::raw;
use crate::{MlxArray, MlxBindingsContext, MlxCError};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_fft_fft` on the runtime GPU stream.
    pub fn fft_fft(
        &self,
        a: &MlxArray,
        n: i32,
        axis: i32,
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_fft", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_fft_fft(output, a.raw(), n, axis, norm, stream) }
        })
    }

    /// Applies MLX-C `mlx_fft_fft2` on the runtime GPU stream.
    pub fn fft_fft2(
        &self,
        a: &MlxArray,
        n: &[i32],
        axes: &[i32],
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_fft2", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_fft2(
                    output,
                    a.raw(),
                    optional_i32_slice(n),
                    n.len(),
                    optional_i32_slice(axes),
                    axes.len(),
                    norm,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_fft_fftfreq` on the runtime GPU stream.
    pub fn fft_fftfreq(&self, n: i32, d: f64) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_fftfreq", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_fft_fftfreq(output, n, d, stream) }
        })
    }

    /// Applies MLX-C `mlx_fft_fftn` on the runtime GPU stream.
    pub fn fft_fftn(
        &self,
        a: &MlxArray,
        n: &[i32],
        axes: &[i32],
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_fftn", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_fftn(
                    output,
                    a.raw(),
                    optional_i32_slice(n),
                    n.len(),
                    optional_i32_slice(axes),
                    axes.len(),
                    norm,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_fft_fftshift` on the runtime GPU stream.
    pub fn fft_fftshift(&self, a: &MlxArray, axes: &[i32]) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_fftshift", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_fftshift(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_fft_ifft` on the runtime GPU stream.
    pub fn fft_ifft(
        &self,
        a: &MlxArray,
        n: i32,
        axis: i32,
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_ifft", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_fft_ifft(output, a.raw(), n, axis, norm, stream) }
        })
    }

    /// Applies MLX-C `mlx_fft_ifft2` on the runtime GPU stream.
    pub fn fft_ifft2(
        &self,
        a: &MlxArray,
        n: &[i32],
        axes: &[i32],
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_ifft2", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_ifft2(
                    output,
                    a.raw(),
                    optional_i32_slice(n),
                    n.len(),
                    optional_i32_slice(axes),
                    axes.len(),
                    norm,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_fft_ifftn` on the runtime GPU stream.
    pub fn fft_ifftn(
        &self,
        a: &MlxArray,
        n: &[i32],
        axes: &[i32],
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_ifftn", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_ifftn(
                    output,
                    a.raw(),
                    optional_i32_slice(n),
                    n.len(),
                    optional_i32_slice(axes),
                    axes.len(),
                    norm,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_fft_ifftshift` on the runtime GPU stream.
    pub fn fft_ifftshift(&self, a: &MlxArray, axes: &[i32]) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_ifftshift", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_ifftshift(
                    output,
                    a.raw(),
                    optional_i32_slice(axes),
                    axes.len(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_fft_irfft` on the runtime GPU stream.
    pub fn fft_irfft(
        &self,
        a: &MlxArray,
        n: i32,
        axis: i32,
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_irfft", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_fft_irfft(output, a.raw(), n, axis, norm, stream) }
        })
    }

    /// Applies MLX-C `mlx_fft_irfft2` on the runtime GPU stream.
    pub fn fft_irfft2(
        &self,
        a: &MlxArray,
        n: &[i32],
        axes: &[i32],
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_irfft2", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_irfft2(
                    output,
                    a.raw(),
                    optional_i32_slice(n),
                    n.len(),
                    optional_i32_slice(axes),
                    axes.len(),
                    norm,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_fft_irfftn` on the runtime GPU stream.
    pub fn fft_irfftn(
        &self,
        a: &MlxArray,
        n: &[i32],
        axes: &[i32],
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_irfftn", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_irfftn(
                    output,
                    a.raw(),
                    optional_i32_slice(n),
                    n.len(),
                    optional_i32_slice(axes),
                    axes.len(),
                    norm,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_fft_rfft` on the runtime GPU stream.
    pub fn fft_rfft(
        &self,
        a: &MlxArray,
        n: i32,
        axis: i32,
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_rfft", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_fft_rfft(output, a.raw(), n, axis, norm, stream) }
        })
    }

    /// Applies MLX-C `mlx_fft_rfft2` on the runtime GPU stream.
    pub fn fft_rfft2(
        &self,
        a: &MlxArray,
        n: &[i32],
        axes: &[i32],
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_rfft2", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_rfft2(
                    output,
                    a.raw(),
                    optional_i32_slice(n),
                    n.len(),
                    optional_i32_slice(axes),
                    axes.len(),
                    norm,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_fft_rfftfreq` on the runtime GPU stream.
    pub fn fft_rfftfreq(&self, n: i32, d: f64) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_rfftfreq", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_fft_rfftfreq(output, n, d, stream) }
        })
    }

    /// Applies MLX-C `mlx_fft_rfftn` on the runtime GPU stream.
    pub fn fft_rfftn(
        &self,
        a: &MlxArray,
        n: &[i32],
        axes: &[i32],
        norm: raw::mlx_fft_norm,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX fft_rfftn", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fft_rfftn(
                    output,
                    a.raw(),
                    optional_i32_slice(n),
                    n.len(),
                    optional_i32_slice(axes),
                    axes.len(),
                    norm,
                    stream,
                )
            }
        })
    }
}

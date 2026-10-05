//! Generated MLX-C wrappers for the binary math family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::raw;
use crate::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_allclose` on the runtime GPU stream.
    pub fn allclose(
        &self,
        a: &MlxArray,
        b: &MlxArray,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX allclose", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_allclose(output, a.raw(), b.raw(), rtol, atol, equal_nan, stream) }
        })
    }

    /// Applies MLX-C `mlx_arctan2` on the runtime GPU stream.
    pub fn arctan2(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX arctan2", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_arctan2(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_bitwise_and` on the runtime GPU stream.
    pub fn bitwise_and(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX bitwise_and", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_bitwise_and(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_bitwise_invert` on the runtime GPU stream.
    pub fn bitwise_invert(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX bitwise_invert", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_bitwise_invert(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_bitwise_or` on the runtime GPU stream.
    pub fn bitwise_or(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX bitwise_or", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_bitwise_or(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_bitwise_xor` on the runtime GPU stream.
    pub fn bitwise_xor(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX bitwise_xor", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_bitwise_xor(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Builds the MLX-C `mlx_divmod` outputs on the runtime GPU stream.
    pub fn divmod(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArrayVector, MlxCError> {
        self.output_vector_array("apply MLX divmod", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_divmod(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_equal` on the runtime GPU stream.
    pub fn equal(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX equal", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_equal(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_inner` on the runtime GPU stream.
    pub fn inner(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX inner", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_inner(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_isclose` on the runtime GPU stream.
    pub fn isclose(
        &self,
        a: &MlxArray,
        b: &MlxArray,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX isclose", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_isclose(output, a.raw(), b.raw(), rtol, atol, equal_nan, stream) }
        })
    }

    /// Applies MLX-C `mlx_kron` on the runtime GPU stream.
    pub fn kron(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX kron", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_kron(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_less_equal` on the runtime GPU stream.
    pub fn less_equal(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX less_equal", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_less_equal(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_logical_and` on the runtime GPU stream.
    pub fn logical_and(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX logical_and", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_logical_and(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_logical_not` on the runtime GPU stream.
    pub fn logical_not(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX logical_not", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_logical_not(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_logical_or` on the runtime GPU stream.
    pub fn logical_or(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX logical_or", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_logical_or(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_logical_xor` on the runtime GPU stream.
    pub fn logical_xor(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX logical_xor", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_logical_xor(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_maximum` on the runtime GPU stream.
    pub fn maximum(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX maximum", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_maximum(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_minimum` on the runtime GPU stream.
    pub fn minimum(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX minimum", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_minimum(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_outer` on the runtime GPU stream.
    pub fn outer(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX outer", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_outer(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_remainder` on the runtime GPU stream.
    pub fn remainder(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX remainder", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_remainder(output, a.raw(), b.raw(), stream) }
        })
    }
}

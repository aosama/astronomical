//! Generated MLX-C wrappers for the linear algebra family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::mlx_bindings_support::{c_string_argument, optional_i32_slice};
use crate::raw;
use crate::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_linalg_cholesky` on the runtime GPU stream.
    pub fn linalg_cholesky(&self, a: &MlxArray, upper: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_cholesky", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_cholesky(output, a.raw(), upper, stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_cholesky_inv` on the runtime GPU stream.
    pub fn linalg_cholesky_inv(&self, a: &MlxArray, upper: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_cholesky_inv", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_cholesky_inv(output, a.raw(), upper, stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_cross` on the runtime GPU stream.
    pub fn linalg_cross(
        &self,
        a: &MlxArray,
        b: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_cross", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_cross(output, a.raw(), b.raw(), axis, stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_det` on the runtime GPU stream.
    pub fn linalg_det(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_det", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_det(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_eig`, producing both outputs on the runtime GPU stream.
    pub fn linalg_eig(&self, a: &MlxArray) -> Result<(MlxArray, MlxArray), MlxCError> {
        self.output_array_pair("apply MLX linalg_eig", |output, second_output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_eig(output, second_output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_eigh`, producing both outputs on the runtime GPU stream.
    pub fn linalg_eigh(
        &self,
        a: &MlxArray,
        upper_lower: &str,
    ) -> Result<(MlxArray, MlxArray), MlxCError> {
        let upper_lower_argument = c_string_argument(upper_lower, "linalg_eigh")?;
        self.output_array_pair("apply MLX linalg_eigh", |output, second_output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_linalg_eigh(
                    output,
                    second_output,
                    a.raw(),
                    upper_lower_argument.as_ptr(),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_linalg_eigvals` on the runtime GPU stream.
    pub fn linalg_eigvals(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_eigvals", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_eigvals(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_eigvalsh` on the runtime GPU stream.
    pub fn linalg_eigvalsh(&self, a: &MlxArray, upper_lower: &str) -> Result<MlxArray, MlxCError> {
        let upper_lower_argument = c_string_argument(upper_lower, "linalg_eigvalsh")?;
        self.output_array("apply MLX linalg_eigvalsh", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_linalg_eigvalsh(output, a.raw(), upper_lower_argument.as_ptr(), stream)
            }
        })
    }

    /// Applies MLX-C `mlx_linalg_inv` on the runtime GPU stream.
    pub fn linalg_inv(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_inv", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_inv(output, a.raw(), stream) }
        })
    }

    /// Builds the MLX-C `mlx_linalg_lu` outputs on the runtime GPU stream.
    pub fn linalg_lu(&self, a: &MlxArray) -> Result<MlxArrayVector, MlxCError> {
        self.output_vector_array("apply MLX linalg_lu", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_lu(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_lu_factor`, producing both outputs on the runtime GPU stream.
    pub fn linalg_lu_factor(&self, a: &MlxArray) -> Result<(MlxArray, MlxArray), MlxCError> {
        self.output_array_pair(
            "apply MLX linalg_lu_factor",
            |output, second_output, stream| {
                // SAFETY: Inputs and stream are live and output is uniquely writable.
                unsafe { raw::mlx_linalg_lu_factor(output, second_output, a.raw(), stream) }
            },
        )
    }

    /// Applies MLX-C `mlx_linalg_norm` on the runtime GPU stream.
    pub fn linalg_norm(
        &self,
        a: &MlxArray,
        ord: f64,
        axis: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_norm", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_linalg_norm(
                    output,
                    a.raw(),
                    ord,
                    optional_i32_slice(axis),
                    axis.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_linalg_norm_l2` on the runtime GPU stream.
    pub fn linalg_norm_l2(
        &self,
        a: &MlxArray,
        axis: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_norm_l2", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_linalg_norm_l2(
                    output,
                    a.raw(),
                    optional_i32_slice(axis),
                    axis.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_linalg_norm_matrix` on the runtime GPU stream.
    pub fn linalg_norm_matrix(
        &self,
        a: &MlxArray,
        ord: &str,
        axis: &[i32],
        keepdims: bool,
    ) -> Result<MlxArray, MlxCError> {
        let ord_argument = c_string_argument(ord, "linalg_norm_matrix")?;
        self.output_array("apply MLX linalg_norm_matrix", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_linalg_norm_matrix(
                    output,
                    a.raw(),
                    ord_argument.as_ptr(),
                    optional_i32_slice(axis),
                    axis.len(),
                    keepdims,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_linalg_pinv` on the runtime GPU stream.
    pub fn linalg_pinv(&self, a: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_pinv", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_pinv(output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_qr`, producing both outputs on the runtime GPU stream.
    pub fn linalg_qr(&self, a: &MlxArray) -> Result<(MlxArray, MlxArray), MlxCError> {
        self.output_array_pair("apply MLX linalg_qr", |output, second_output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_qr(output, second_output, a.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_slogdet`, producing both outputs on the runtime GPU stream.
    pub fn linalg_slogdet(&self, a: &MlxArray) -> Result<(MlxArray, MlxArray), MlxCError> {
        self.output_array_pair(
            "apply MLX linalg_slogdet",
            |output, second_output, stream| {
                // SAFETY: Inputs and stream are live and output is uniquely writable.
                unsafe { raw::mlx_linalg_slogdet(output, second_output, a.raw(), stream) }
            },
        )
    }

    /// Applies MLX-C `mlx_linalg_solve` on the runtime GPU stream.
    pub fn linalg_solve(&self, a: &MlxArray, b: &MlxArray) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_solve", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_solve(output, a.raw(), b.raw(), stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_solve_triangular` on the runtime GPU stream.
    pub fn linalg_solve_triangular(
        &self,
        a: &MlxArray,
        b: &MlxArray,
        upper: bool,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_solve_triangular", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_solve_triangular(output, a.raw(), b.raw(), upper, stream) }
        })
    }

    /// Builds the MLX-C `mlx_linalg_svd` outputs on the runtime GPU stream.
    pub fn linalg_svd(&self, a: &MlxArray, compute_uv: bool) -> Result<MlxArrayVector, MlxCError> {
        self.output_vector_array("apply MLX linalg_svd", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_svd(output, a.raw(), compute_uv, stream) }
        })
    }

    /// Applies MLX-C `mlx_linalg_tri_inv` on the runtime GPU stream.
    pub fn linalg_tri_inv(&self, a: &MlxArray, upper: bool) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX linalg_tri_inv", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_linalg_tri_inv(output, a.raw(), upper, stream) }
        })
    }
}

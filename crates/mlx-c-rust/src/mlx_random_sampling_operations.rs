//! Generated MLX-C wrappers for the random sampling family.
//!
//! Each wrapper maps one MLX-C operation onto the runtime GPU stream
//! with the crate's owned-handle and captured-error semantics. The
//! coverage contract keeps this family in lockstep with the pinned
//! upstream headers.

use crate::mlx_bindings_support::optional_i32_slice;
use crate::raw;
use crate::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype};
impl MlxBindingsContext {
    /// Applies MLX-C `mlx_random_bernoulli` on the runtime GPU stream.
    pub fn random_bernoulli(
        &self,
        p: &MlxArray,
        shape: &[i32],
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_bernoulli", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_bernoulli(
                    output,
                    p.raw(),
                    optional_i32_slice(shape),
                    shape.len(),
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_bits` on the runtime GPU stream.
    pub fn random_bits(
        &self,
        shape: &[i32],
        width: i32,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_bits", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_bits(
                    output,
                    optional_i32_slice(shape),
                    shape.len(),
                    width,
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_categorical_num_samples` on the runtime GPU stream.
    pub fn random_categorical_num_samples(
        &self,
        logits_: &MlxArray,
        axis: i32,
        num_samples: i32,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array(
            "apply MLX random_categorical_num_samples",
            |output, stream| {
                // SAFETY: Inputs and stream are live and output is uniquely writable.
                unsafe {
                    raw::mlx_random_categorical_num_samples(
                        output,
                        logits_.raw(),
                        axis,
                        num_samples,
                        key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                        stream,
                    )
                }
            },
        )
    }

    /// Applies MLX-C `mlx_random_categorical_shape` on the runtime GPU stream.
    pub fn random_categorical_shape(
        &self,
        logits: &MlxArray,
        axis: i32,
        shape: &[i32],
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_categorical_shape", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_categorical_shape(
                    output,
                    logits.raw(),
                    axis,
                    optional_i32_slice(shape),
                    shape.len(),
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_gumbel` on the runtime GPU stream.
    pub fn random_gumbel(
        &self,
        shape: &[i32],
        dtype: MlxDtype,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_gumbel", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_gumbel(
                    output,
                    optional_i32_slice(shape),
                    shape.len(),
                    dtype.to_raw(),
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_laplace` on the runtime GPU stream.
    pub fn random_laplace(
        &self,
        shape: &[i32],
        dtype: MlxDtype,
        loc: f32,
        scale: f32,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_laplace", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_laplace(
                    output,
                    optional_i32_slice(shape),
                    shape.len(),
                    dtype.to_raw(),
                    loc,
                    scale,
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_multivariate_normal` on the runtime GPU stream.
    pub fn random_multivariate_normal(
        &self,
        mean: &MlxArray,
        cov: &MlxArray,
        shape: &[i32],
        dtype: MlxDtype,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_multivariate_normal", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_multivariate_normal(
                    output,
                    mean.raw(),
                    cov.raw(),
                    optional_i32_slice(shape),
                    shape.len(),
                    dtype.to_raw(),
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_normal_broadcast` on the runtime GPU stream.
    pub fn random_normal_broadcast(
        &self,
        shape: &[i32],
        dtype: MlxDtype,
        loc: Option<&MlxArray>,
        scale: Option<&MlxArray>,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_normal_broadcast", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_normal_broadcast(
                    output,
                    optional_i32_slice(shape),
                    shape.len(),
                    dtype.to_raw(),
                    loc.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    scale.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_permutation` on the runtime GPU stream.
    pub fn random_permutation(
        &self,
        x: &MlxArray,
        axis: i32,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_permutation", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_permutation(
                    output,
                    x.raw(),
                    axis,
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_permutation_arange` on the runtime GPU stream.
    pub fn random_permutation_arange(
        &self,
        x: i32,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_permutation_arange", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_permutation_arange(
                    output,
                    x,
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_randint` on the runtime GPU stream.
    pub fn random_randint(
        &self,
        low: &MlxArray,
        high: &MlxArray,
        shape: &[i32],
        dtype: MlxDtype,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_randint", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_randint(
                    output,
                    low.raw(),
                    high.raw(),
                    optional_i32_slice(shape),
                    shape.len(),
                    dtype.to_raw(),
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_split_num` on the runtime GPU stream.
    pub fn random_split_num(&self, key: &MlxArray, num: i32) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_split_num", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_random_split_num(output, key.raw(), num, stream) }
        })
    }

    /// Applies MLX-C `mlx_random_truncated_normal` on the runtime GPU stream.
    pub fn random_truncated_normal(
        &self,
        lower: &MlxArray,
        upper: &MlxArray,
        shape: &[i32],
        dtype: MlxDtype,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_truncated_normal", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_truncated_normal(
                    output,
                    lower.raw(),
                    upper.raw(),
                    optional_i32_slice(shape),
                    shape.len(),
                    dtype.to_raw(),
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_random_uniform` on the runtime GPU stream.
    pub fn random_uniform(
        &self,
        low: &MlxArray,
        high: &MlxArray,
        shape: &[i32],
        dtype: MlxDtype,
        key: Option<&MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array("apply MLX random_uniform", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_random_uniform(
                    output,
                    low.raw(),
                    high.raw(),
                    optional_i32_slice(shape),
                    shape.len(),
                    dtype.to_raw(),
                    key.map_or(MlxArray::empty_raw(), MlxArray::raw),
                    stream,
                )
            }
        })
    }

    /// Seeds MLX's default random key sequence through MLX-C `mlx_random_seed`.
    pub fn seed_random_generation(&self, seed: u64) -> Result<(), MlxCError> {
        // SAFETY: Seeding takes a plain integer and reports status through the
        // captured-error machinery.
        let status = unsafe { raw::mlx_random_seed(seed) };
        crate::error::check_status(status, "seed MLX random generation")
    }
}

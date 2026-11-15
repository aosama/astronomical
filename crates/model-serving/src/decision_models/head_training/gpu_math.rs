//! The few MLX building blocks head training is written in, with one error
//! type, so each family's forward and backward pass reads as its formula.
//!
//! Every array here is 32-bit. The served heads compute their trainable
//! parts in 32-bit (Clef widens its bfloat16 head at load; Laya's scorer
//! reads its 16-bit weights as 32-bit), and the gotcha this guards against
//! is training in a lower precision than the result is used at.
//!
//! Matrix products need care. On GPUs with TF32 support MLX runs a
//! float32 matrix product in TF32 (a 10-bit mantissa, about 1e-3 relative),
//! on by default through its `MLX_ENABLE_TF32` and read once per process,
//! so it cannot be switched off for training alone without changing what
//! serving computes. Serving does not always take that path: Kev scores one
//! question at a time, a matrix-vector product that stays in float32, and a
//! TF32 copy of its head scored 0.009 away from the model, enough to fail
//! the parity check that proves training starts from the served model. So
//! every product here goes through [`GpuMath::precise_matmul`], which
//! recovers nearly full float32 accuracy from three TF32 products.
//! Parameters, optimizer state and every update are element-wise float32
//! operations, so small updates are never rounded away either.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxCError, MlxDtype};
use thiserror::Error;

/// `1 / √2` and `1 / √(2π)`, for the exact GELU's derivative.
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
const FRAC_1_SQRT_2PI: f32 = 0.398_942_3;

/// A graphics-processor operation failed during training.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("head training failed on the graphics processor: {0}")]
pub(crate) struct GpuTrainingError(pub(crate) String);

impl From<MlxCError> for GpuTrainingError {
    fn from(mlx_error: MlxCError) -> Self {
        Self(mlx_error.to_string())
    }
}

pub(crate) type GpuResult<T> = Result<T, GpuTrainingError>;

/// The runtime, with the compound operations training needs.
#[derive(Clone, Copy)]
pub(crate) struct GpuMath<'runtime> {
    pub(crate) runtime: &'runtime MlxRuntime,
}

impl GpuMath<'_> {
    /// `input · weightᵀ + bias` for `input` `[rows, in]` and `weight` `[out, in]`.
    pub(crate) fn linear(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        bias: &MlxArray,
    ) -> GpuResult<MlxArray> {
        let projected =
            self.precise_matmul(input, &self.runtime.transpose_axes(weight, &[1, 0])?)?;
        Ok(self.runtime.add(&projected, bias)?)
    }

    /// `left · right` to nearly float32 accuracy on TF32 hardware.
    ///
    /// Each operand is split into a bfloat16-exact high part and the float32
    /// remainder. High × high is exact in TF32 (8-bit mantissas), and the two
    /// high × low products lose only TF32's rounding of a remainder already
    /// 2⁻⁹ smaller, so the sum is good to about 4e-6 relative; the dropped
    /// low × low term is smaller still. Three products instead of one cost
    /// nothing that matters at the size of a decision head.
    pub(crate) fn precise_matmul(&self, left: &MlxArray, right: &MlxArray) -> GpuResult<MlxArray> {
        let runtime = self.runtime;
        let (left_high, left_low) = self.high_and_low_parts(left)?;
        let (right_high, right_low) = self.high_and_low_parts(right)?;
        let corrections = runtime.add(
            &runtime.matmul(&left_high, &right_low)?,
            &runtime.matmul(&left_low, &right_high)?,
        )?;
        Ok(runtime.add(&runtime.matmul(&left_high, &right_high)?, &corrections)?)
    }

    /// `input` rounded to bfloat16 (as float32), and what that rounding left out.
    fn high_and_low_parts(&self, input: &MlxArray) -> GpuResult<(MlxArray, MlxArray)> {
        let runtime = self.runtime;
        let high = runtime.astype(
            &runtime.astype(input, MlxDtype::BFloat16)?,
            MlxDtype::Float32,
        )?;
        let low = runtime.subtract(input, &high)?;
        Ok((high, low))
    }

    /// `d gelu / dx = Φ(x) + x · φ(x)` for the exact (error-function) GELU.
    pub(crate) fn gelu_derivative(&self, input: &MlxArray) -> GpuResult<MlxArray> {
        let runtime = self.runtime;
        let cumulative = runtime.multiply_scalar(
            &runtime.add(
                &self.scalar(1.0)?,
                &runtime.erf(&runtime.multiply_scalar(input, FRAC_1_SQRT_2)?)?,
            )?,
            0.5,
        )?;
        let density = runtime.multiply_scalar(
            &runtime.exp(&runtime.multiply_scalar(&runtime.square(input)?, -0.5)?)?,
            FRAC_1_SQRT_2PI,
        )?;
        Ok(runtime.add(&cumulative, &runtime.multiply(input, &density)?)?)
    }

    /// A 32-bit scalar.
    pub(crate) fn scalar(&self, value: f32) -> GpuResult<MlxArray> {
        Ok(self.runtime.array_from_f32(&[value], &[])?)
    }

    /// The sum over the first axis: a `[rows, width]` gradient summed into
    /// a bias's `[width]`.
    pub(crate) fn column_sum(&self, input: &MlxArray) -> GpuResult<MlxArray> {
        Ok(self.runtime.sum_axis(input, 0, false)?)
    }

    /// `left ᵀ · right` for two `[rows, ·]` arrays: a weight's gradient from
    /// its output gradient and its input.
    pub(crate) fn transposed_product(
        &self,
        left: &MlxArray,
        right: &MlxArray,
    ) -> GpuResult<MlxArray> {
        self.precise_matmul(&self.runtime.transpose_axes(left, &[1, 0])?, right)
    }

    /// A host vector as a 32-bit array of `shape`.
    pub(crate) fn upload(&self, values: &[f32], shape: &[usize]) -> GpuResult<MlxArray> {
        let mut array_shape = Vec::with_capacity(shape.len());
        for dimension in shape {
            array_shape.push(*dimension as i32);
        }
        Ok(self.runtime.array_from_f32(values, &array_shape)?)
    }

    /// Host indices as a 32-bit integer array.
    pub(crate) fn upload_indices(&self, indices: &[usize]) -> GpuResult<MlxArray> {
        let mut index_values = Vec::with_capacity(indices.len());
        for index in indices {
            index_values.push(*index as i32);
        }
        Ok(self
            .runtime
            .array_from_i32(&index_values, &[index_values.len() as i32])?)
    }

    /// An array read back to the host as 32-bit floats.
    pub(crate) fn download(&self, input: &MlxArray) -> GpuResult<Vec<f32>> {
        Ok(self
            .runtime
            .astype(input, MlxDtype::Float32)?
            .to_vec_f32()?)
    }
}

#[cfg(test)]
#[path = "gpu_math_tests.rs"]
mod tests;

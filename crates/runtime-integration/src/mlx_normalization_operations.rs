//! Validated normalization wrappers over MLX's fused kernels.

use crate::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Applies fused LayerNorm without affine weight or bias arrays.
    pub fn layer_norm_without_weight_and_bias(
        &self,
        input: &MlxArray,
        epsilon: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        const OPERATION: &str = "apply non-affine MLX LayerNorm";
        if input.shape().is_empty()
            || !matches!(
                input.dtype(),
                MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::BFloat16
            )
            || !epsilon.is_finite()
            || epsilon < 0.0
        {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: OPERATION,
                description:
                    "input must be a non-scalar floating array and epsilon must be finite and nonnegative"
                        .to_owned(),
            });
        }
        self.output_array(OPERATION, |output, stream| {
            // SAFETY: Input and stream are live, and MLX-C defines empty handles
            // as the absence of affine LayerNorm weight and bias arrays.
            unsafe {
                raw::mlx_fast_layer_norm(
                    output,
                    input.raw(),
                    MlxArray::empty_raw(),
                    MlxArray::empty_raw(),
                    epsilon,
                    stream,
                )
            }
        })
    }
}

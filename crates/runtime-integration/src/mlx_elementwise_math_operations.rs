//! Validated elementwise math wrappers over the pinned MLX-C operations.

use crate::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Applies the elementwise square root while preserving lazy evaluation.
    pub fn sqrt(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        validate_floating_array(input, "apply MLX square root")?;
        self.output_array("apply MLX square root", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_sqrt(output, input.raw(), stream) }
        })
    }

    /// Clamps a floating array to inclusive scalar bounds in its current dtype.
    pub fn clip(
        &self,
        input: &MlxArray,
        minimum: f32,
        maximum: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        const OPERATION: &str = "clip an MLX array";
        validate_floating_array(input, OPERATION)?;
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err(runtime_operation_error(
                OPERATION,
                "clip bounds must be finite and ordered",
            ));
        }
        let float_minimum = self.array_from_f32(&[minimum], &[])?;
        let float_maximum = self.array_from_f32(&[maximum], &[])?;
        let minimum_bound = self.astype(&float_minimum, input.dtype())?;
        let maximum_bound = self.astype(&float_maximum, input.dtype())?;
        self.output_array(OPERATION, |output, stream| {
            // SAFETY: Input, scalar bounds, and stream are live and output is unique.
            unsafe {
                raw::mlx_clip(
                    output,
                    input.raw(),
                    minimum_bound.raw(),
                    maximum_bound.raw(),
                    stream,
                )
            }
        })
    }
}

fn validate_floating_array(
    input: &MlxArray,
    operation: &'static str,
) -> Result<(), MlxRuntimeError> {
    if !matches!(
        input.dtype(),
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::BFloat16
    ) {
        return Err(runtime_operation_error(
            operation,
            "input must have a supported floating dtype",
        ));
    }
    Ok(())
}

fn runtime_operation_error(operation: &'static str, description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.to_owned(),
    }
}

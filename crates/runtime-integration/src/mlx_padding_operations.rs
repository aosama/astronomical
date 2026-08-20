//! Validated axis-specific padding for lazy MLX tensors.

use crate::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Applies axis-specific constant padding, including asymmetric image padding.
    pub fn pad(
        &self,
        input: &MlxArray,
        axes: &[i32],
        low_padding: &[i32],
        high_padding: &[i32],
        pad_value: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        const OPERATION: &str = "pad an MLX array";
        validate_padding_arguments(input, axes, low_padding, high_padding, pad_value)?;
        let float_pad_value = self.array_from_f32(&[pad_value], &[])?;
        let typed_pad_value = self.astype(&float_pad_value, input.dtype())?;
        self.output_array(OPERATION, |output_array, stream| {
            // SAFETY: Input, scalar, and stream are live; axis and padding
            // vectors were validated and remain borrowed for this graph call.
            unsafe {
                raw::mlx_pad(
                    output_array,
                    input.raw(),
                    axes.as_ptr(),
                    axes.len(),
                    low_padding.as_ptr(),
                    low_padding.len(),
                    high_padding.as_ptr(),
                    high_padding.len(),
                    typed_pad_value.raw(),
                    c"constant".as_ptr(),
                    stream,
                )
            }
        })
    }
}

fn validate_padding_arguments(
    input: &MlxArray,
    axes: &[i32],
    low_padding: &[i32],
    high_padding: &[i32],
    pad_value: f32,
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "pad an MLX array";
    if axes.is_empty() || axes.len() != low_padding.len() || axes.len() != high_padding.len() {
        return Err(runtime_operation_error(
            OPERATION,
            "padding axes and low/high sizes must have one matching nonempty length",
        ));
    }
    if !matches!(
        input.dtype(),
        MlxDtype::Float16 | MlxDtype::Float32 | MlxDtype::BFloat16
    ) || !pad_value.is_finite()
    {
        return Err(runtime_operation_error(
            OPERATION,
            "padding requires a supported floating array and finite pad value",
        ));
    }
    if low_padding.iter().chain(high_padding).any(|size| *size < 0) {
        return Err(runtime_operation_error(
            OPERATION,
            "padding sizes must be nonnegative",
        ));
    }
    let input_shape = input.shape();
    let rank = i32::try_from(input_shape.len())
        .map_err(|_| runtime_operation_error(OPERATION, "input rank exceeds i32 range"))?;
    let mut normalized_axes = Vec::with_capacity(axes.len());
    for axis in axes {
        if *axis < -rank || *axis >= rank {
            return Err(runtime_operation_error(
                OPERATION,
                "padding axes must refer to existing input dimensions",
            ));
        }
        let normalized_axis = if *axis < 0 { *axis + rank } else { *axis };
        if normalized_axes.contains(&normalized_axis) {
            return Err(runtime_operation_error(
                OPERATION,
                "padding axes must be unique",
            ));
        }
        normalized_axes.push(normalized_axis);
    }
    for ((normalized_axis, low_size), high_size) in
        normalized_axes.iter().zip(low_padding).zip(high_padding)
    {
        let axis_index = usize::try_from(*normalized_axis)
            .map_err(|_| runtime_operation_error(OPERATION, "padding axis conversion failed"))?;
        input_shape[axis_index]
            .checked_add(*low_size)
            .and_then(|size| size.checked_add(*high_size))
            .ok_or_else(|| {
                runtime_operation_error(OPERATION, "padded dimension exceeds i32 range")
            })?;
    }
    Ok(())
}

fn runtime_operation_error(operation: &'static str, description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: description.to_owned(),
    }
}

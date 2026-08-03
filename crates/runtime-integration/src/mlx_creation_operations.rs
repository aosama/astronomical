use crate::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, raw};

impl MlxRuntime {
    /// Copies float32 values into a new owned MLX array after validating shape arithmetic.
    pub fn array_from_f32(
        &self,
        values: &[f32],
        shape: &[i32],
    ) -> Result<MlxArray, MlxRuntimeError> {
        MlxArray::from_f32(values, shape)
    }

    /// Copies int32 values into a new owned MLX array after validating shape arithmetic.
    pub fn array_from_i32(
        &self,
        values: &[i32],
        shape: &[i32],
    ) -> Result<MlxArray, MlxRuntimeError> {
        MlxArray::from_i32(values, shape)
    }

    /// Copies uint32 values into a new owned MLX array after validating shape arithmetic.
    pub fn array_from_u32(
        &self,
        values: &[u32],
        shape: &[i32],
    ) -> Result<MlxArray, MlxRuntimeError> {
        MlxArray::from_u32(values, shape)
    }

    /// Creates an int32 half-open range with unit stride.
    pub fn arange_i32(&self, start: i32, stop: i32) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("create an MLX integer range", |output_array, stream| {
            // SAFETY: Scalar arguments are copied and output is uniquely writable.
            unsafe {
                raw::mlx_arange(
                    output_array,
                    f64::from(start),
                    f64::from(stop),
                    1.0,
                    MlxDtype::Int32.to_raw(),
                    stream,
                )
            }
        })
    }

    /// Creates a zero-filled MLX array with a validated static shape.
    pub fn zeros(&self, shape: &[i32], dtype: MlxDtype) -> Result<MlxArray, MlxRuntimeError> {
        validate_creation_shape(shape)?;
        self.output_array("create a zero-filled MLX array", |output_array, stream| {
            // SAFETY: Shape remains borrowed for this graph-building call, dtype
            // is a known MLX dtype, stream is live, and output is uniquely writable.
            unsafe {
                raw::mlx_zeros(
                    output_array,
                    shape.as_ptr(),
                    shape.len(),
                    dtype.to_raw(),
                    stream,
                )
            }
        })
    }
}

fn validate_creation_shape(shape: &[i32]) -> Result<(), MlxRuntimeError> {
    if shape.iter().any(|dimension_size| *dimension_size < 0) {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "create a zero-filled MLX array",
            description: "array dimensions must be nonnegative".to_owned(),
        });
    }

    let element_count = shape
        .iter()
        .try_fold(1_usize, |element_count, dimension_size| {
            let dimension_size = usize::try_from(*dimension_size).ok()?;
            element_count.checked_mul(dimension_size)
        });
    if element_count.is_none() {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "create a zero-filled MLX array",
            description: "array element count overflows usize".to_owned(),
        });
    }

    Ok(())
}

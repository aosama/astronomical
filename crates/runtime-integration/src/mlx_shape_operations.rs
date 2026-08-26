use crate::{
    MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, mlx_array_vector::MlxArrayVector, raw,
};

impl MlxRuntime {
    /// Reorders array dimensions using an explicit complete permutation.
    pub fn transpose_axes(
        &self,
        input: &MlxArray,
        axes: &[i32],
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("transpose MLX array axes", |output, stream| {
            // SAFETY: Axes remain borrowed for this copying graph operation.
            unsafe {
                raw::mlx_transpose_axes(output, input.raw(), axes.as_ptr(), axes.len(), stream)
            }
        })
    }

    /// Reshapes an array without changing its logical element count.
    pub fn reshape(&self, input: &MlxArray, shape: &[i32]) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("reshape an MLX array", |output, stream| {
            // SAFETY: Shape remains borrowed for this copying graph operation.
            unsafe { raw::mlx_reshape(output, input.raw(), shape.as_ptr(), shape.len(), stream) }
        })
    }

    /// Selects indices along one array axis through MLX-C `mlx_take_axis`.
    ///
    /// Qwen3.5-MoE uses this for learned positions, quantized text embeddings, and the
    /// dense global-expert-ID to compact-page-slot lookup used by paged expert execution.
    /// This call records a lazy MLX operation on the runtime stream; it does not copy
    /// indices to Rust or synchronously evaluate the resulting array.
    pub fn take_axis(
        &self,
        input: &MlxArray,
        indices: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("take values from an MLX array axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_take_axis(output, input.raw(), indices.raw(), axis, stream) }
        })
    }

    /// Selects per-position indices along one array axis.
    pub fn take_along_axis(
        &self,
        input: &MlxArray,
        indices: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("take MLX values along an axis", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_take_along_axis(output, input.raw(), indices.raw(), axis, stream) }
        })
    }

    /// Writes selected values into a copy of one array along an existing axis.
    pub fn put_along_axis(
        &self,
        input: &MlxArray,
        indices: &MlxArray,
        values: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_put_along_axis_arguments(input, indices, values, axis)?;
        self.output_array("put MLX values along an axis", |output, stream| {
            // SAFETY: Inputs and stream are live, axis and shapes were
            // validated, and output is uniquely writable.
            unsafe {
                raw::mlx_put_along_axis(
                    output,
                    input.raw(),
                    indices.raw(),
                    values.raw(),
                    axis,
                    stream,
                )
            }
        })
    }

    /// Adds `updates` into a copy of `input` at `indices` along one axis.
    ///
    /// Compact mixture-of-experts split reduction lands several assignment rows
    /// on the same token. `put_along_axis` overwrites; this keeps every routed
    /// expert's contribution. MLX requires `updates.ndim() == input.ndim() +
    /// indices.ndim()`. For destination `[token_rows, hidden]`, pass indices
    /// `[assignment_count]` and updates `[assignment_count, 1, hidden]`.
    pub fn scatter_add(
        &self,
        input: &MlxArray,
        indices: &MlxArray,
        updates: &MlxArray,
        axis: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("scatter-add values into an MLX array", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_scatter_add_single(
                    output,
                    input.raw(),
                    indices.raw(),
                    updates.raw(),
                    axis,
                    stream,
                )
            }
        })
    }

    /// Slices an array with one static start, stop, and stride per axis.
    pub fn slice(
        &self,
        input: &MlxArray,
        starts: &[i32],
        stops: &[i32],
        strides: &[i32],
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_static_slice(input, starts, stops, strides)?;
        self.output_array("slice an MLX array", |output, stream| {
            // SAFETY: Slice bounds remain borrowed for this graph-building call,
            // input and stream are live, and output is uniquely writable.
            unsafe {
                raw::mlx_slice(
                    output,
                    input.raw(),
                    starts.as_ptr(),
                    starts.len(),
                    stops.as_ptr(),
                    stops.len(),
                    strides.as_ptr(),
                    strides.len(),
                    stream,
                )
            }
        })
    }

    /// Replaces one static slice through MLX-C `mlx_slice_update`.
    ///
    /// MLX arrays are immutable graph values: this returns a new lazy array. The
    /// visual splice uses it to replace contiguous `<|image_pad|>` embedding runs
    /// without copying the complete embedding tensor through Rust.
    pub fn slice_update(
        &self,
        source: &MlxArray,
        update: &MlxArray,
        starts: &[i32],
        stops: &[i32],
        strides: &[i32],
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_static_slice(source, starts, stops, strides)?;
        self.output_array("update an MLX array slice", |output, stream| {
            // SAFETY: Source, update, and stream are live; slice vectors match
            // the source rank; output is uniquely writable.
            unsafe {
                raw::mlx_slice_update(
                    output,
                    source.raw(),
                    update.raw(),
                    starts.as_ptr(),
                    starts.len(),
                    stops.as_ptr(),
                    stops.len(),
                    strides.as_ptr(),
                    strides.len(),
                    stream,
                )
            }
        })
    }

    /// Inserts one singleton dimension.
    pub fn expand_dims(&self, input: &MlxArray, axis: i32) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("expand an MLX array dimension", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_expand_dims(output, input.raw(), axis, stream) }
        })
    }

    /// Concatenates arrays through MLX-C `mlx_concatenate_axis`.
    pub fn concatenate_axis(
        &self,
        arrays: &[&MlxArray],
        axis: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let array_vector = MlxArrayVector::new(arrays)?;
        self.output_array("concatenate MLX arrays", |output, stream| {
            // SAFETY: The vector and stream are live and output is uniquely writable.
            unsafe { raw::mlx_concatenate_axis(output, array_vector.raw(), axis, stream) }
        })
    }

    /// Broadcasts an array to a validated static shape.
    pub fn broadcast_to(
        &self,
        input: &MlxArray,
        shape: &[i32],
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_static_shape("broadcast an MLX array", shape)?;
        let broadcast_values =
            self.output_array("broadcast an MLX array", |output_array, stream| {
                // SAFETY: Input and stream are live, shape remains borrowed for this
                // graph-building call, and output is uniquely writable.
                unsafe {
                    raw::mlx_broadcast_to(
                        output_array,
                        input.raw(),
                        shape.as_ptr(),
                        shape.len(),
                        stream,
                    )
                }
            })?;
        self.contiguous_row_major(
            &broadcast_values,
            "materialize broadcast MLX array contiguously",
        )
    }

    /// Repeats each value `repeat_count` times along one existing axis.
    pub fn repeat_axis(
        &self,
        input: &MlxArray,
        repeat_count: i32,
        axis: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        validate_repeat_axis_arguments(input, repeat_count, axis)?;
        self.output_array("repeat MLX values along an axis", |output_array, stream| {
            // SAFETY: Input and stream are live, scalar arguments were validated,
            // and output is uniquely writable for MLX to populate.
            unsafe { raw::mlx_repeat_axis(output_array, input.raw(), repeat_count, axis, stream) }
        })
    }

    /// Stacks arrays with identical shapes along one new axis.
    pub fn stack_axis(&self, arrays: &[&MlxArray], axis: i32) -> Result<MlxArray, MlxRuntimeError> {
        validate_stack_axis_arguments(arrays, axis)?;
        let array_vector = MlxArrayVector::new(arrays)?;
        self.output_array(
            "stack MLX arrays along a new axis",
            |output_array, stream| {
                // SAFETY: The vector keeps all source arrays live for this graph-building
                // call, axis was validated, stream is live, and output is uniquely writable.
                unsafe { raw::mlx_stack_axis(output_array, array_vector.raw(), axis, stream) }
            },
        )
    }

    /// Removes one existing singleton axis.
    pub fn squeeze_axis(&self, input: &MlxArray, axis: i32) -> Result<MlxArray, MlxRuntimeError> {
        validate_squeeze_axis_arguments(input, axis)?;
        self.output_array("squeeze one MLX singleton axis", |output_array, stream| {
            // SAFETY: Input and stream are live, axis was validated to name a
            // singleton dimension, and output is uniquely writable.
            unsafe { raw::mlx_squeeze_axis(output_array, input.raw(), axis, stream) }
        })
    }
}

fn validate_static_slice(
    input: &MlxArray,
    starts: &[i32],
    stops: &[i32],
    strides: &[i32],
) -> Result<(), MlxRuntimeError> {
    let rank = input.shape().len();
    if starts.len() != rank || stops.len() != rank || strides.len() != rank {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "slice an MLX array",
            description: "slice starts, stops, and strides must match the input rank".to_owned(),
        });
    }
    if strides.contains(&0) {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "slice an MLX array",
            description: "slice strides must be nonzero".to_owned(),
        });
    }
    Ok(())
}

fn validate_put_along_axis_arguments(
    input: &MlxArray,
    indices: &MlxArray,
    values: &MlxArray,
    axis: i32,
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "put MLX values along an axis";
    let input_shape = input.shape();
    let index_shape = indices.shape();
    if input.dtype() != values.dtype() || index_shape != values.shape() {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "input/value dtypes and index/value shapes must match".to_owned(),
        });
    }
    if !matches!(
        indices.dtype(),
        MlxDtype::UInt8
            | MlxDtype::UInt16
            | MlxDtype::UInt32
            | MlxDtype::UInt64
            | MlxDtype::Int8
            | MlxDtype::Int16
            | MlxDtype::Int32
            | MlxDtype::Int64
    ) {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "put-along-axis indices must have an integral dtype".to_owned(),
        });
    }
    let rank = i32::try_from(input_shape.len()).map_err(|_| MlxRuntimeError::RuntimeOperation {
        operation: OPERATION,
        description: "input rank exceeds the MLX integer range".to_owned(),
    })?;
    if rank == 0 || index_shape.len() != input_shape.len() || axis < -rank || axis >= rank {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "axis and index rank must match the destination array".to_owned(),
        });
    }
    let normalized_axis = if axis < 0 { axis + rank } else { axis } as usize;
    if index_shape
        .iter()
        .enumerate()
        .any(|(dimension_index, dimension)| {
            dimension_index != normalized_axis && *dimension != input_shape[dimension_index]
        })
    {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: OPERATION,
            description: "non-selected dimensions must match the destination shape".to_owned(),
        });
    }
    Ok(())
}

fn validate_static_shape(operation: &'static str, shape: &[i32]) -> Result<(), MlxRuntimeError> {
    if shape.iter().any(|dimension_size| *dimension_size < 0) {
        return Err(runtime_operation_error(
            operation,
            "shape dimensions must be nonnegative",
        ));
    }

    let element_count = shape
        .iter()
        .try_fold(1_usize, |element_count, dimension_size| {
            let dimension_size = usize::try_from(*dimension_size).ok()?;
            element_count.checked_mul(dimension_size)
        });
    if element_count.is_none() {
        return Err(runtime_operation_error(
            operation,
            "shape element count overflows usize",
        ));
    }
    Ok(())
}

fn validate_repeat_axis_arguments(
    input: &MlxArray,
    repeat_count: i32,
    axis: i32,
) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "repeat MLX values along an axis";
    if repeat_count <= 0 {
        return Err(runtime_operation_error(
            OPERATION,
            "repeat count must be positive",
        ));
    }

    let rank = i32::try_from(input.shape().len())
        .map_err(|_| runtime_operation_error(OPERATION, "input rank exceeds i32 range"))?;
    if rank == 0 || axis < -rank || axis >= rank {
        return Err(runtime_operation_error(
            OPERATION,
            "axis must refer to an existing input dimension",
        ));
    }
    Ok(())
}

fn validate_stack_axis_arguments(arrays: &[&MlxArray], axis: i32) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "stack MLX arrays along a new axis";
    let first_array = arrays
        .first()
        .ok_or_else(|| runtime_operation_error(OPERATION, "at least one array is required"))?;
    let first_shape = first_array.shape();
    if arrays
        .iter()
        .any(|candidate_array| candidate_array.shape() != first_shape)
    {
        return Err(runtime_operation_error(
            OPERATION,
            "all stacked arrays must have the same shape",
        ));
    }

    let output_rank = i32::try_from(first_shape.len() + 1)
        .map_err(|_| runtime_operation_error(OPERATION, "output rank exceeds i32 range"))?;
    if axis < -output_rank || axis >= output_rank {
        return Err(runtime_operation_error(
            OPERATION,
            "axis must refer to a valid inserted output dimension",
        ));
    }
    Ok(())
}

fn validate_squeeze_axis_arguments(input: &MlxArray, axis: i32) -> Result<(), MlxRuntimeError> {
    const OPERATION: &str = "squeeze one MLX singleton axis";
    let input_shape = input.shape();
    let rank = i32::try_from(input_shape.len())
        .map_err(|_| runtime_operation_error(OPERATION, "input rank exceeds i32 range"))?;
    if rank == 0 || axis < -rank || axis >= rank {
        return Err(runtime_operation_error(
            OPERATION,
            "axis must refer to an existing input dimension",
        ));
    }

    let normalized_axis = if axis < 0 { axis + rank } else { axis };
    let axis_index = usize::try_from(normalized_axis)
        .map_err(|_| runtime_operation_error(OPERATION, "axis conversion failed"))?;
    if input_shape[axis_index] != 1 {
        return Err(runtime_operation_error(
            OPERATION,
            "squeezed axis must have size one",
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

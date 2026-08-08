use crate::{
    MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError, mlx_array_vector::MlxArrayVector,
    mlx_runtime::check_status, raw,
};

impl MlxRuntime {
    /// Builds lazy matrix multiplication on the runtime's GPU stream.
    pub fn matmul(&self, left: &MlxArray, right: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("multiply MLX arrays", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_matmul(output, left.raw(), right.raw(), stream) }
        })
    }

    /// Builds matrix multiplication while selecting matrix batches on-device.
    ///
    /// This maps directly to MLX `gather_mm`; unlike `take` followed by
    /// `matmul`, it does not materialize one weight matrix per assignment.
    pub fn gather_dense_matmul(
        &self,
        left: &MlxArray,
        right: &MlxArray,
        left_indices: Option<&MlxArray>,
        right_indices: Option<&MlxArray>,
        sorted_indices: bool,
    ) -> Result<MlxArray, MlxRuntimeError> {
        if left_indices.is_none() && right_indices.is_none() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "build dense gather_mm",
                description: "at least one matrix-batch index array is required".to_owned(),
            });
        }
        let raw_left_indices = left_indices.map_or_else(MlxArray::empty_raw, MlxArray::raw);
        let raw_right_indices = right_indices.map_or_else(MlxArray::empty_raw, MlxArray::raw);
        self.output_array("build dense gather_mm", |output, stream| {
            // SAFETY: Arrays and stream are live; absent index arrays use MLX's
            // official empty-handle convention; output is uniquely writable.
            unsafe {
                raw::mlx_gather_mm(
                    output,
                    left.raw(),
                    right.raw(),
                    raw_left_indices,
                    raw_right_indices,
                    sorted_indices,
                    stream,
                )
            }
        })
    }

    /// Applies MLX-C `mlx_addmm`: `alpha * (left @ right) + beta * bias`.
    ///
    /// See `mlx-c/mlx/c/ops.h::mlx_addmm` and its forwarding implementation in
    /// `ops.cpp`. Keep affine model layers on this fused operation: separate
    /// `mlx_matmul` then `mlx_add` changes BF16 accumulation/rounding and violates
    /// the Qwen3.5 vision numerical contract.
    pub fn addmm(
        &self,
        bias: &MlxArray,
        left: &MlxArray,
        right: &MlxArray,
        alpha: f32,
        beta: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply fused MLX addmm", |output, stream| {
            // SAFETY: Arrays and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_addmm(
                    output,
                    bias.raw(),
                    left.raw(),
                    right.raw(),
                    alpha,
                    beta,
                    stream,
                )
            }
        })
    }

    /// Casts an array while preserving lazy evaluation.
    pub fn astype(&self, input: &MlxArray, dtype: MlxDtype) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("cast an MLX array", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_astype(output, input.raw(), dtype.to_raw(), stream) }
        })
    }

    /// Applies fused RMS normalization with a required certified weight.
    pub fn rms_norm(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        epsilon: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX RMS normalization", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_fast_rms_norm(output, input.raw(), weight.raw(), epsilon, stream) }
        })
    }

    /// Applies fused RMS normalization without an affine weight.
    pub fn rms_norm_without_weight(
        &self,
        input: &MlxArray,
        epsilon: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array(
            "apply unweighted MLX RMS normalization",
            |output, stream| {
                // SAFETY: Input and stream are live, the empty handle is MLX's
                // official convention for no affine weight, and output is unique.
                unsafe {
                    raw::mlx_fast_rms_norm(
                        output,
                        input.raw(),
                        MlxArray::empty_raw(),
                        epsilon,
                        stream,
                    )
                }
            },
        )
    }

    /// Applies fused LayerNorm through MLX-C `mlx_fast_layer_norm`.
    ///
    /// The declaration lives in `mlx-c/mlx/c/fast.h`; `fast.cpp` forwards to
    /// `mlx::core::fast::layer_norm`, allowing MLX to choose its Metal kernel.
    pub fn layer_norm(
        &self,
        input: &MlxArray,
        weight: &MlxArray,
        bias: &MlxArray,
        epsilon: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX LayerNorm", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_fast_layer_norm(
                    output,
                    input.raw(),
                    weight.raw(),
                    bias.raw(),
                    epsilon,
                    stream,
                )
            }
        })
    }

    /// Adds two broadcast-compatible arrays.
    pub fn add(&self, left: &MlxArray, right: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("add MLX arrays", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_add(output, left.raw(), right.raw(), stream) }
        })
    }

    /// Subtracts two broadcast-compatible arrays.
    pub fn subtract(&self, left: &MlxArray, right: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("subtract MLX arrays", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_subtract(output, left.raw(), right.raw(), stream) }
        })
    }

    /// Divides two broadcast-compatible arrays.
    pub fn divide(&self, left: &MlxArray, right: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("divide MLX arrays", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_divide(output, left.raw(), right.raw(), stream) }
        })
    }

    /// Applies elementwise floor division to broadcast-compatible arrays.
    pub fn floor_divide(
        &self,
        left: &MlxArray,
        right: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("floor-divide MLX arrays", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_floor_divide(output, left.raw(), right.raw(), stream) }
        })
    }

    /// Negates every element in an array.
    pub fn negative(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("negate an MLX array", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_negative(output, input.raw(), stream) }
        })
    }

    /// Applies the elementwise natural exponential.
    pub fn exp(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX exp", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_exp(output, input.raw(), stream) }
        })
    }

    /// Applies the elementwise natural logarithm of one plus the input.
    pub fn log1p(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX log1p", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_log1p(output, input.raw(), stream) }
        })
    }

    /// Applies the numerically stable elementwise `log(exp(left) + exp(right))`.
    pub fn logaddexp(
        &self,
        left: &MlxArray,
        right: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX logaddexp", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_logaddexp(output, left.raw(), right.raw(), stream) }
        })
    }

    /// Multiplies two broadcast-compatible arrays.
    pub fn multiply(&self, left: &MlxArray, right: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("multiply MLX arrays elementwise", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_multiply(output, left.raw(), right.raw(), stream) }
        })
    }

    /// Multiplies an array by a scalar represented in the array's dtype.
    pub fn multiply_scalar(
        &self,
        input: &MlxArray,
        scalar: f32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let float_scalar = self.array_from_f32(&[scalar], &[])?;
        let typed_scalar = self.astype(&float_scalar, input.dtype())?;
        self.multiply(input, &typed_scalar)
    }

    /// Compares two broadcast-compatible arrays elementwise.
    pub fn greater_equal(
        &self,
        left: &MlxArray,
        right: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("compare MLX arrays", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe { raw::mlx_greater_equal(output, left.raw(), right.raw(), stream) }
        })
    }

    /// Compares two broadcast-compatible arrays elementwise for strict greater-than.
    pub fn greater(&self, left: &MlxArray, right: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array(
            "compare MLX arrays for strict greater-than",
            |output, stream| {
                // SAFETY: Inputs and stream are live and output is uniquely writable.
                unsafe { raw::mlx_greater(output, left.raw(), right.raw(), stream) }
            },
        )
    }

    /// Computes the cumulative sum along one axis.
    pub fn cumsum(
        &self,
        input: &MlxArray,
        axis: i32,
        reverse: bool,
        inclusive: bool,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("compute MLX cumulative sum", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_cumsum(output, input.raw(), axis, reverse, inclusive, stream) }
        })
    }

    /// Selects values from two broadcast-compatible arrays using a boolean mask.
    pub fn where_select(
        &self,
        condition: &MlxArray,
        when_true: &MlxArray,
        when_false: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("select values with an MLX mask", |output, stream| {
            // SAFETY: Inputs and stream are live and output is uniquely writable.
            unsafe {
                raw::mlx_where(
                    output,
                    condition.raw(),
                    when_true.raw(),
                    when_false.raw(),
                    stream,
                )
            }
        })
    }

    /// Applies precise softmax along one axis.
    pub fn softmax_axis(&self, input: &MlxArray, axis: i32) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX softmax", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_softmax_axis(output, input.raw(), axis, true, stream) }
        })
    }

    /// Sums values along one axis.
    pub fn sum_axis(
        &self,
        input: &MlxArray,
        axis: i32,
        keep_dimensions: bool,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("sum MLX array values along an axis", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_sum_axis(output, input.raw(), axis, keep_dimensions, stream) }
        })
    }

    /// Returns maximum values along one axis.
    pub fn max_axis(
        &self,
        input: &MlxArray,
        axis: i32,
        keep_dimensions: bool,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("max MLX array values along an axis", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_max_axis(output, input.raw(), axis, keep_dimensions, stream) }
        })
    }

    /// Applies the elementwise sigmoid function.
    pub fn sigmoid(&self, input: &MlxArray) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("apply MLX sigmoid", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_sigmoid(output, input.raw(), stream) }
        })
    }

    /// Returns argmax indices along one axis.
    pub fn argmax_axis(&self, input: &MlxArray, axis: i32) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("compute MLX argmax", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_argmax_axis(output, input.raw(), axis, false, stream) }
        })
    }

    /// Returns indices that partition values along one axis around `kth`.
    pub fn argpartition_axis(
        &self,
        input: &MlxArray,
        kth: i32,
        axis: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("compute MLX argpartition", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_argpartition_axis(output, input.raw(), kth, axis, stream) }
        })
    }

    /// Returns indices that sort values along one axis in ascending order.
    pub fn argsort_axis(&self, input: &MlxArray, axis: i32) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array("compute MLX argsort", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_argsort_axis(output, input.raw(), axis, stream) }
        })
    }

    /// Returns top values along one axis and materializes them contiguously for bounded Rust copies.
    pub fn topk_axis(
        &self,
        input: &MlxArray,
        count: i32,
        axis: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let top_values = self.output_array("compute MLX top-k values", |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_topk_axis(output, input.raw(), count, axis, stream) }
        })?;
        self.contiguous_row_major(&top_values, "materialize MLX top-k values contiguously")
    }

    /// Evaluates a bounded group together so KV state does not retain prior graphs.
    pub fn evaluate_arrays(&self, arrays: &[&MlxArray]) -> Result<(), MlxRuntimeError> {
        let array_vector = MlxArrayVector::new(arrays)?;
        // SAFETY: The vector retains all live arrays for this synchronous evaluation.
        let status = unsafe { raw::mlx_eval(array_vector.raw()) };
        check_status(status, "evaluate an MLX array group")
    }

    /// Asynchronously submits a bounded group to the GPU without blocking the CPU.
    ///
    /// A later host read waits if execution is still in flight. Dependent graphs
    /// can be built and submitted meanwhile, keeping CPU graph construction and
    /// GPU execution overlapped.
    pub fn async_eval_arrays(&self, arrays: &[&MlxArray]) -> Result<(), MlxRuntimeError> {
        let array_vector = MlxArrayVector::new(arrays)?;
        // SAFETY: The vector retains all live arrays for this asynchronous evaluation.
        let status = unsafe { raw::mlx_async_eval(array_vector.raw()) };
        check_status(status, "asynchronously evaluate an MLX array group")
    }

    /// Materializes one uint32 array contiguously and copies its bounded values.
    pub fn copy_u32_values(&self, input: &MlxArray) -> Result<Vec<u32>, MlxRuntimeError> {
        let contiguous_values = self.build_contiguous_row_major_copy(input)?;
        contiguous_values.to_vec_u32()
    }

    /// Builds a lazy row-major contiguous copy without evaluating it.
    pub fn build_contiguous_row_major_copy(
        &self,
        input: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.contiguous_row_major(input, "build an MLX row-major contiguous copy")
    }

    pub(crate) fn contiguous_row_major(
        &self,
        input: &MlxArray,
        operation: &'static str,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.output_array(operation, |output, stream| {
            // SAFETY: Input and stream are live and output is uniquely writable.
            unsafe { raw::mlx_contiguous(output, input.raw(), false, stream) }
        })
    }

    pub(crate) fn output_array(
        &self,
        operation: &'static str,
        build_graph: impl FnOnce(*mut raw::mlx_array, raw::mlx_stream) -> i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let mut output = MlxArray::empty();
        let status = build_graph(output.raw_mut(), self.gpu_stream().raw());
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }
}

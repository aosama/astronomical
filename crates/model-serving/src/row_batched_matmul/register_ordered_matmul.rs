//! A 4-bit affine projection over 1 to 128 token rows from a weight converted
//! once into the register order TensorFold's `tf_lane_reg` reads (see
//! `register_order_source`).
//!
//! The converted weight replaces MLX's packed one rather than sitting beside
//! it: the codes take the same bytes, and scale and bias pairs the same bytes
//! as MLX's two arrays. A model that converts a weight serves every row count
//! of it through this kernel.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype, MlxMetalKernelOutput};

use super::dispatch_plan::{OUTPUT_COLUMNS_PER_SIMDGROUP, QUANTIZATION_GROUP_SIZE};
use super::register_ordered_plan::{
    REGISTER_BLOCK_WORDS, RegisterOrderedDispatchPlan, register_order_conversion_grid,
    register_ordered_slice_count,
};
use super::row_batched_quantized_matmul::{
    GroupSumLayout, RowBatchedMatmulKernels, RowBatchedProjectionError, check_quantization,
    group_sums, integer,
};

/// One projection's 4-bit codes in register order with its scale and bias
/// words, and the input slices every call of it takes.
#[derive(Debug)]
pub struct RegisterOrderedWeight {
    /// One 256-word block per 32-column tile and 64-value group, tile-major.
    pub codes: MlxArray,
    /// `[input / 64, output]` uint32: the bf16 scale in the low half, the bf16
    /// bias in the high half.
    pub scale_bias_words: MlxArray,
    pub output_width: i32,
    pub input_width: i32,
    pub input_slice_count: i32,
}

impl RegisterOrderedWeight {
    /// Converts MLX's packed 4-bit, group-64 weight [output, input / 8] with
    /// its bf16 scales and biases [output, input / 64]. The result is lazy;
    /// the caller evaluates it and may then drop the MLX arrays.
    pub fn convert(
        runtime: &MlxRuntime,
        kernels: &RowBatchedMatmulKernels,
        packed_weight: &MlxArray,
        scales: &MlxArray,
        biases: &MlxArray,
    ) -> Result<Self, RowBatchedProjectionError> {
        let (output_width, input_width) = check_quantization(packed_weight, scales, biases)?;
        let group_count = input_width / QUANTIZATION_GROUP_SIZE;
        let block_count = (output_width / OUTPUT_COLUMNS_PER_SIMDGROUP) * group_count;
        let (grid, threadgroup) = register_order_conversion_grid(output_width, input_width);
        let mut outputs = runtime.apply_metal_kernel(
            &kernels.register_order_conversion,
            &[packed_weight, scales, biases],
            &[
                MlxMetalKernelOutput::new(
                    vec![block_count * REGISTER_BLOCK_WORDS],
                    MlxDtype::UInt32,
                ),
                MlxMetalKernelOutput::new(vec![group_count, output_width], MlxDtype::UInt32),
            ],
            grid,
            threadgroup,
            &[integer("N", output_width), integer("K", input_width)],
        )?;
        let scale_bias_words = outputs
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)?;
        let codes = outputs
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)?;
        Ok(Self {
            codes,
            scale_bias_words,
            output_width,
            input_width,
            input_slice_count: register_ordered_slice_count(output_width, input_width),
        })
    }
}

/// `rows` [..., input] bf16 projected through `weight`: [..., output] bf16,
/// as MLX's `quantized_matmul` with `transpose = true` returns it. The leading
/// dimensions may hold 1 to 128 rows.
pub fn register_ordered_matmul(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    rows: &MlxArray,
    weight: &RegisterOrderedWeight,
) -> Result<MlxArray, RowBatchedProjectionError> {
    let rows_shape = rows.shape();
    if rows_shape.last() != Some(&weight.input_width) || rows.dtype() != MlxDtype::BFloat16 {
        return Err(RowBatchedProjectionError::UnsupportedQuantization {
            description: format!(
                "rows are {:?} {rows_shape:?}, expected bf16 rows {} wide",
                rows.dtype(),
                weight.input_width
            ),
        });
    }
    let row_count: i32 = rows_shape[..rows_shape.len() - 1].iter().product();
    let dispatch_plan = RegisterOrderedDispatchPlan::plan(
        row_count,
        weight.output_width,
        weight.input_width,
        weight.input_slice_count,
    )?;
    let flat_rows = runtime.reshape(rows, &[row_count, weight.input_width])?;
    let row_dimensions =
        runtime.array_from_i32(&[row_count, dispatch_plan.padded_row_count], &[2])?;
    let row_group_sums = group_sums(
        runtime,
        kernels,
        &flat_rows,
        &row_dimensions,
        GroupSumLayout {
            padded_row_count: dispatch_plan.padded_row_count,
            grid: dispatch_plan.group_sum_grid,
            threadgroup: dispatch_plan.group_sum_threadgroup,
        },
        weight.input_width,
    )?;
    let mut outputs = runtime.apply_metal_kernel(
        &kernels.register_ordered_matmul,
        &[
            &flat_rows,
            &row_group_sums,
            &weight.codes,
            &weight.scale_bias_words,
            &row_dimensions,
        ],
        &[MlxMetalKernelOutput::new(
            vec![row_count, weight.output_width],
            MlxDtype::BFloat16,
        )],
        dispatch_plan.grid,
        dispatch_plan.threadgroup,
        &[
            integer("N", weight.output_width),
            integer("K", weight.input_width),
            integer("SK", weight.input_slice_count),
        ],
    )?;
    let projected = outputs
        .pop()
        .ok_or(RowBatchedProjectionError::MissingOutput)?;
    let mut output_shape = rows_shape;
    *output_shape.last_mut().expect("rows have a width") = weight.output_width;
    Ok(runtime.reshape(&projected, &output_shape)?)
}

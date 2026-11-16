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
    /// Columns the kernel computes: a whole number of 32-column tiles.
    pub output_width: i32,
    /// Columns the projection really has; fewer than `output_width` when a
    /// narrow projection was padded with zero rows to fill its last tile.
    pub logical_output_width: i32,
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
        let logical_output_width = packed_weight.shape().first().copied().unwrap_or(0);
        let tiled = TiledRows::padded(runtime, [packed_weight, scales, biases])?;
        let [packed_weight, scales, biases] = [&tiled.packed_weight, &tiled.scales, &tiled.biases];
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
            logical_output_width,
            input_width,
            input_slice_count: register_ordered_slice_count(output_width, input_width),
        })
    }
}

/// MLX's three arrays with zero rows appended up to whole 32-column tiles.
/// A zero code with a zero scale and bias yields a zero output, so the padded
/// columns are cut off after the product and never read. The 27B's 48-wide
/// DeltaNet gates are the narrow case: through MLX's matmul at 16 rows the
/// two of them cost about 6 ms a verify over 48 layers; padded to 64 columns
/// they take this kernel like every other projection.
struct TiledRows {
    packed_weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

impl TiledRows {
    fn padded(
        runtime: &MlxRuntime,
        arrays: [&MlxArray; 3],
    ) -> Result<Self, RowBatchedProjectionError> {
        let row_count = arrays[0].shape().first().copied().unwrap_or(0);
        let missing_rows = (OUTPUT_COLUMNS_PER_SIMDGROUP
            - row_count % OUTPUT_COLUMNS_PER_SIMDGROUP)
            % OUTPUT_COLUMNS_PER_SIMDGROUP;
        let mut padded = Vec::with_capacity(3);
        for array in arrays {
            if missing_rows == 0 || array.shape().len() != 2 {
                padded.push(array.retain()?);
                continue;
            }
            let zero_rows = runtime.zeros(&[missing_rows, array.shape()[1]], array.dtype())?;
            padded.push(runtime.concatenate_axis(&[array, &zero_rows], 0)?);
        }
        let biases = padded
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)?;
        let scales = padded
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)?;
        let packed_weight = padded
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)?;
        Ok(Self {
            packed_weight,
            scales,
            biases,
        })
    }
}

/// One input's rows, flattened, with their per-group sums: what every
/// register-ordered projection of that input reads. Projections that share
/// an input (a layer's q, k and v; its gate and up) share one of these, so
/// the sums are computed once.
pub struct ProjectionInput {
    flat_rows: MlxArray,
    group_sums: MlxArray,
    row_dimensions: MlxArray,
    /// The rows' leading shape, `[..., input]` without the input width.
    leading_shape: Vec<i32>,
    row_count: i32,
    input_width: i32,
}

impl ProjectionInput {
    /// `rows` [..., input] bf16, 1 to 128 rows.
    pub fn prepare(
        runtime: &MlxRuntime,
        kernels: &RowBatchedMatmulKernels,
        rows: &MlxArray,
    ) -> Result<Self, RowBatchedProjectionError> {
        let rows_shape = rows.shape();
        let input_width = *rows_shape.last().unwrap_or(&0);
        if rows.dtype() != MlxDtype::BFloat16 || input_width <= 0 {
            return Err(RowBatchedProjectionError::UnsupportedQuantization {
                description: format!(
                    "rows are {:?} {rows_shape:?}, expected bf16 rows",
                    rows.dtype()
                ),
            });
        }
        let leading_shape = rows_shape[..rows_shape.len() - 1].to_vec();
        let row_count: i32 = leading_shape.iter().product();
        // The sums' layout depends on the rows and input width only, so any
        // output width that tiles plans them the same.
        let sums_plan = RegisterOrderedDispatchPlan::plan(row_count, 32, input_width, 1)?;
        // The kernels index the rows' memory directly, so the rows go in
        // with whatever leading shape they have: no reshape node a call.
        let flat_rows = rows.retain()?;
        let row_dimensions =
            kernels.row_dimensions(runtime, row_count, sums_plan.padded_row_count)?;
        let group_sums = group_sums(
            runtime,
            kernels,
            &flat_rows,
            &row_dimensions,
            GroupSumLayout {
                padded_row_count: sums_plan.padded_row_count,
                grid: sums_plan.group_sum_grid,
                threadgroup: sums_plan.group_sum_threadgroup,
            },
            input_width,
        )?;
        Ok(Self {
            flat_rows,
            group_sums,
            row_dimensions,
            leading_shape,
            row_count,
            input_width,
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
    let input = ProjectionInput::prepare(runtime, kernels, rows)?;
    register_ordered_projection(runtime, kernels, &input, weight)
}

/// A prepared input projected through `weight`: [..., output] bf16.
pub fn register_ordered_projection(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    input: &ProjectionInput,
    weight: &RegisterOrderedWeight,
) -> Result<MlxArray, RowBatchedProjectionError> {
    if input.input_width != weight.input_width {
        return Err(RowBatchedProjectionError::UnsupportedQuantization {
            description: format!(
                "rows are {} wide, the weight reads {}",
                input.input_width, weight.input_width
            ),
        });
    }
    let dispatch_plan = RegisterOrderedDispatchPlan::plan(
        input.row_count,
        weight.output_width,
        weight.input_width,
        weight.input_slice_count,
    )?;
    let mut computed_shape = input.leading_shape.clone();
    computed_shape.push(weight.output_width);
    let mut outputs = runtime.apply_metal_kernel(
        &kernels.register_ordered_matmul,
        &[
            &input.flat_rows,
            &input.group_sums,
            &weight.codes,
            &weight.scale_bias_words,
            &input.row_dimensions,
        ],
        &[MlxMetalKernelOutput::new(
            computed_shape.clone(),
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
    if weight.logical_output_width == weight.output_width {
        return Ok(projected);
    }
    // A padded narrow projection: cut the zero columns off the last axis.
    let mut logical_shape = computed_shape;
    *logical_shape.last_mut().expect("the output has a width") = weight.logical_output_width;
    let starts = vec![0; logical_shape.len()];
    let strides = vec![1; logical_shape.len()];
    Ok(runtime.slice(&projected, &starts, &logical_shape, &strides)?)
}

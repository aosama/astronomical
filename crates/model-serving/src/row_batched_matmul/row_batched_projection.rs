//! One 4-bit affine projection served by the row-batched kernel.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};

use super::dispatch_plan::{
    OUTPUT_COLUMNS_PER_SIMDGROUP, QUANTIZATION_GROUP_SIZE, RowBatchedDispatchPlan,
    RowBatchedDispatchRefusal, check_weight_shape,
};
use super::kernel_source::{
    GROUP_SUM_SOURCE, ROW_BATCHED_MATMUL_HEADER, ROW_BATCHED_MATMUL_SOURCE,
};

/// The two compiled kernels, built once and shared by every projection.
pub struct RowBatchedMatmulKernels {
    group_sum: MlxMetalKernel,
    matmul: MlxMetalKernel,
}

impl RowBatchedMatmulKernels {
    /// Declares both kernels. MLX compiles a custom kernel's Metal source only
    /// on its first evaluation, so this succeeds on chips without M5 matrix
    /// units too: it is not a capability check, and serving must prove the
    /// kernel through `kernel_capability` before routing rows to it.
    pub fn declare() -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            group_sum: MlxMetalKernel::new(
                "astro_row_batched_group_sum",
                &["X", "row_dimensions"],
                &["XS"],
                GROUP_SUM_SOURCE,
            )?,
            matmul: MlxMetalKernel::new_with_header(
                "astro_row_batched_matmul_4bit",
                &["X", "XS", "Wq", "SBt", "row_dimensions"],
                &["Y"],
                ROW_BATCHED_MATMUL_HEADER,
                ROW_BATCHED_MATMUL_SOURCE,
            )?,
        })
    }
}

/// A 4-bit, group-64 projection regrouped once at load: the codes into
/// contiguous per-tile groups (`tile_packed_weight`) and the scales and biases
/// into `[groups, output, 2]` bf16 pairs, so each simdgroup's reads for one
/// group are contiguous runs. Both hold the same bytes as MLX's layout.
pub struct RowBatchedProjection {
    tiled_weight: MlxArray,
    scale_bias_pairs: MlxArray,
    output_width: i32,
    input_width: i32,
}

impl RowBatchedProjection {
    /// Binds MLX's packed 4-bit, group-64 `weight` [output, input / 8] uint32
    /// with its bf16 `scales` and `biases` [output, input / 64]. The word count
    /// alone cannot tell 4-bit group 64 from other packings (8-bit has twice
    /// the words, group 32 twice the scales), so the scales' shape and every
    /// dtype are checked and anything else is refused rather than misread.
    /// The pairs cost 4 bytes a group per output column, the same as the
    /// scales and biases they are built from.
    pub fn prepare(
        runtime: &MlxRuntime,
        packed_weight: &MlxArray,
        scales: &MlxArray,
        biases: &MlxArray,
    ) -> Result<Self, RowBatchedProjectionError> {
        let (output_width, input_width) = check_quantization(packed_weight, scales, biases)?;
        check_weight_shape(output_width, input_width)?;
        let tiled_weight = tile_packed_weight(runtime, packed_weight, output_width, input_width)?;
        // Stacking along a new last axis writes a fresh row-contiguous bf16
        // array, which the kernel reads as uint4 words of four pairs.
        let scale_bias_pairs = runtime.stack_axis(
            &[&runtime.transpose(scales)?, &runtime.transpose(biases)?],
            -1,
        )?;
        runtime.evaluate_arrays(&[&tiled_weight, &scale_bias_pairs])?;
        Ok(Self {
            tiled_weight,
            scale_bias_pairs,
            output_width,
            input_width,
        })
    }

    pub fn output_width(&self) -> i32 {
        self.output_width
    }

    /// Plans a call over `row_count` rows, or says why this shape or row
    /// count must take MLX's own quantized matmul instead.
    pub fn plan(&self, row_count: i32) -> Result<(), RowBatchedDispatchRefusal> {
        RowBatchedDispatchPlan::plan(row_count, self.output_width, self.input_width).map(drop)
    }

    /// `rows` [M, input] bf16 times the weight transposed: [M, output] bf16.
    pub fn forward(
        &self,
        runtime: &MlxRuntime,
        kernels: &RowBatchedMatmulKernels,
        rows: &MlxArray,
    ) -> Result<MlxArray, RowBatchedProjectionError> {
        let row_count = rows.shape()[0];
        let dispatch_plan =
            RowBatchedDispatchPlan::plan(row_count, self.output_width, self.input_width)?;
        let row_dimensions =
            runtime.array_from_i32(&[row_count, dispatch_plan.padded_row_count], &[2])?;
        let group_sums =
            self.group_sums(runtime, kernels, rows, &row_dimensions, &dispatch_plan)?;
        let mut outputs = runtime.apply_metal_kernel(
            &kernels.matmul,
            &[
                rows,
                &group_sums,
                &self.tiled_weight,
                &self.scale_bias_pairs,
                &row_dimensions,
            ],
            &[MlxMetalKernelOutput::new(
                vec![row_count, self.output_width],
                MlxDtype::BFloat16,
            )],
            dispatch_plan.grid,
            dispatch_plan.threadgroup,
            &[
                integer("TMR", dispatch_plan.row_blocks_per_threadgroup),
                integer("N", self.output_width),
                integer("K", self.input_width),
                integer("NT", OUTPUT_COLUMNS_PER_SIMDGROUP),
                integer("SK", dispatch_plan.input_slice_count),
                integer("GS", QUANTIZATION_GROUP_SIZE),
                integer("EDGE", i32::from(dispatch_plan.last_block_passes_padding)),
            ],
        )?;
        outputs
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)
    }

    fn group_sums(
        &self,
        runtime: &MlxRuntime,
        kernels: &RowBatchedMatmulKernels,
        rows: &MlxArray,
        row_dimensions: &MlxArray,
        dispatch_plan: &RowBatchedDispatchPlan,
    ) -> Result<MlxArray, RowBatchedProjectionError> {
        let group_count = self.input_width / QUANTIZATION_GROUP_SIZE;
        let mut outputs = runtime.apply_metal_kernel(
            &kernels.group_sum,
            &[rows, row_dimensions],
            &[MlxMetalKernelOutput::new(
                vec![group_count, dispatch_plan.padded_row_count],
                MlxDtype::Float32,
            )],
            dispatch_plan.group_sum_grid,
            dispatch_plan.group_sum_threadgroup,
            &[
                integer("K", self.input_width),
                integer("GS", QUANTIZATION_GROUP_SIZE),
            ],
        )?;
        outputs
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)
    }
}

fn integer(name: &'static str, value: i32) -> MlxMetalKernelTemplateArgument {
    MlxMetalKernelTemplateArgument::Integer {
        name,
        integer_template_argument: value,
    }
}

/// Why a row-batched projection call failed.
#[derive(Debug, thiserror::Error)]
pub enum RowBatchedProjectionError {
    #[error(transparent)]
    Refused(#[from] RowBatchedDispatchRefusal),
    #[error(transparent)]
    Mlx(#[from] mlx_bridge::MlxCError),
    #[error(transparent)]
    Runtime(#[from] MlxRuntimeError),
    #[error("the row-batched kernel returned no output")]
    MissingOutput,
    #[error(
        "the row-batched kernel reads only 4-bit, group-64 weights with bf16 scales: {description}"
    )]
    UnsupportedQuantization { description: String },
}

/// Regroups MLX's packed `[output, input / 8]` codes into
/// `[output / NT][input / GS][NT columns x one group's words]`: the same bytes,
/// ordered so one column tile's codes for one group are contiguous.
fn tile_packed_weight(
    runtime: &MlxRuntime,
    packed_weight: &MlxArray,
    output_width: i32,
    input_width: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let words_per_group = QUANTIZATION_GROUP_SIZE / 8;
    let tile_shaped = runtime.reshape(
        packed_weight,
        &[
            output_width / OUTPUT_COLUMNS_PER_SIMDGROUP,
            OUTPUT_COLUMNS_PER_SIMDGROUP,
            input_width / QUANTIZATION_GROUP_SIZE,
            words_per_group,
        ],
    )?;
    let group_major = runtime.transpose_axes(&tile_shaped, &[0, 2, 1, 3])?;
    // Reshaping a transposed view materializes it in the new order.
    Ok(runtime.reshape(&group_major, &[output_width, input_width / 8])?)
}

/// The `(output, input)` widths of a 4-bit, group-64 projection with bf16
/// scales and biases, or why the tensors are some other packing.
fn check_quantization(
    packed_weight: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
) -> Result<(i32, i32), RowBatchedProjectionError> {
    let weight_shape = packed_weight.shape();
    if packed_weight.dtype() != MlxDtype::UInt32 || weight_shape.len() != 2 {
        return Err(unsupported(format!(
            "weight is {:?} {weight_shape:?}, not a 2-D uint32 matrix",
            packed_weight.dtype()
        )));
    }
    let output_width = weight_shape[0];
    let input_width = weight_shape[1] * 8;
    let expected_scale_shape = vec![output_width, input_width / QUANTIZATION_GROUP_SIZE];
    for (tensor_role, tensor) in [("scales", scales), ("biases", biases)] {
        if tensor.dtype() != MlxDtype::BFloat16 || tensor.shape() != expected_scale_shape {
            return Err(unsupported(format!(
                "{tensor_role} are {:?} {:?}, expected bf16 {expected_scale_shape:?}",
                tensor.dtype(),
                tensor.shape()
            )));
        }
    }
    Ok((output_width, input_width))
}

fn unsupported(description: String) -> RowBatchedProjectionError {
    RowBatchedProjectionError::UnsupportedQuantization { description }
}

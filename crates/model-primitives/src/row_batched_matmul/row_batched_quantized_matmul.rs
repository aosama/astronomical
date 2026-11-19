//! A 4-bit affine projection over a few token rows, read straight from MLX's
//! own packed weight, scales and biases.
//!
//! Nothing is repacked or copied: the same arrays MLX's quantized matmul
//! reads for prompts and one-row decode steps serve this kernel too, so a
//! model that drafts holds no second layout and no extra memory.

use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::dispatch_plan::{
    OUTPUT_COLUMNS_PER_SIMDGROUP, QUANTIZATION_GROUP_SIZE, RowBatchedDispatchPlan,
    RowBatchedDispatchRefusal, check_weight_shape,
};
use super::kernel_source::{
    GROUP_SUM_SOURCE, ROW_BATCHED_MATMUL_HEADER, ROW_BATCHED_MATMUL_SOURCE,
};
use super::register_order_source::{
    REGISTER_ORDER_CONVERSION_SOURCE, REGISTER_ORDERED_MATMUL_HEADER,
    REGISTER_ORDERED_MATMUL_SOURCE,
};

/// The two kernels, declared once and shared by every projection.
pub struct RowBatchedMatmulKernels {
    group_sum: MlxMetalKernel,
    matmul: MlxMetalKernel,
    pub register_order_conversion: MlxMetalKernel,
    pub register_ordered_matmul: MlxMetalKernel,
    /// `[rows, padded rows]` int32 arrays already uploaded, by row count: a
    /// pass makes hundreds of projection calls at one row count, and each
    /// upload is a host-to-GPU round the CPU pays while the GPU waits.
    row_dimension_arrays: std::cell::RefCell<std::collections::HashMap<(i32, i32), MlxArray>>,
}

/// The kernel handles are opaque MLX objects; a model that holds them only
/// says that it does.
impl std::fmt::Debug for RowBatchedMatmulKernels {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("RowBatchedMatmulKernels")
    }
}

impl RowBatchedMatmulKernels {
    /// Declares every kernel. MLX compiles a custom kernel's Metal source only
    /// on its first evaluation, so this succeeds on chips without M5 matrix
    /// units too: it is not a capability check, and serving must prove the
    /// kernel by evaluating it before routing rows to it.
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
                &["X", "XS", "Wq", "S", "B", "row_dimensions"],
                &["Y"],
                ROW_BATCHED_MATMUL_HEADER,
                ROW_BATCHED_MATMUL_SOURCE,
            )?,
            register_order_conversion: MlxMetalKernel::new(
                "astro_register_order_conversion_4bit",
                &["Wq", "S", "B"],
                &["R", "P"],
                REGISTER_ORDER_CONVERSION_SOURCE,
            )?,
            register_ordered_matmul: MlxMetalKernel::new_with_header(
                "astro_register_ordered_matmul_4bit",
                &["X", "XS", "W", "SB", "row_dimensions"],
                &["Y"],
                REGISTER_ORDERED_MATMUL_HEADER,
                REGISTER_ORDERED_MATMUL_SOURCE,
            )?,
            row_dimension_arrays: std::cell::RefCell::default(),
        })
    }

    /// The `[row_count, padded_row_count]` array the kernels read, uploaded
    /// once per distinct pair.
    pub fn row_dimensions(
        &self,
        runtime: &MlxRuntime,
        row_count: i32,
        padded_row_count: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let mut uploaded = self.row_dimension_arrays.borrow_mut();
        if let Some(row_dimensions) = uploaded.get(&(row_count, padded_row_count)) {
            return Ok(row_dimensions.retain()?);
        }
        let row_dimensions = runtime.array_from_i32(&[row_count, padded_row_count], &[2])?;
        uploaded.insert((row_count, padded_row_count), row_dimensions.retain()?);
        Ok(row_dimensions)
    }
}

/// `rows` [..., input] bf16 times MLX's packed 4-bit, group-64 `packed_weight`
/// [output, input / 8] transposed, with its bf16 `scales` and `biases`
/// [output, input / 64]: [..., output] bf16, as `quantized_matmul` with
/// `transpose = true` returns it. The leading dimensions may hold 1 to 32 rows.
pub fn row_batched_quantized_matmul(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    rows: &MlxArray,
    packed_weight: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
) -> Result<MlxArray, RowBatchedProjectionError> {
    let (output_width, input_width) = check_quantization(packed_weight, scales, biases)?;
    let rows_shape = rows.shape();
    if rows_shape.last() != Some(&input_width) || rows.dtype() != MlxDtype::BFloat16 {
        return Err(unsupported(format!(
            "rows are {:?} {rows_shape:?}, expected bf16 rows {input_width} wide",
            rows.dtype()
        )));
    }
    let row_count: i32 = rows_shape[..rows_shape.len() - 1].iter().product();
    let dispatch_plan = RowBatchedDispatchPlan::plan(row_count, output_width, input_width)?;
    let flat_rows = runtime.reshape(rows, &[row_count, input_width])?;
    let row_dimensions =
        runtime.array_from_i32(&[row_count, dispatch_plan.padded_row_count], &[2])?;
    let group_sums = group_sums(
        runtime,
        kernels,
        &flat_rows,
        &row_dimensions,
        GroupSumLayout {
            padded_row_count: dispatch_plan.padded_row_count,
            grid: dispatch_plan.group_sum_grid,
            threadgroup: dispatch_plan.group_sum_threadgroup,
        },
        input_width,
    )?;
    let mut outputs = runtime.apply_metal_kernel(
        &kernels.matmul,
        &[
            &flat_rows,
            &group_sums,
            packed_weight,
            scales,
            biases,
            &row_dimensions,
        ],
        &[MlxMetalKernelOutput::new(
            vec![row_count, output_width],
            MlxDtype::BFloat16,
        )],
        dispatch_plan.grid,
        dispatch_plan.threadgroup,
        &[
            integer("TMR", dispatch_plan.row_blocks_per_threadgroup),
            integer("N", output_width),
            integer("K", input_width),
            integer("NT", OUTPUT_COLUMNS_PER_SIMDGROUP),
            integer("SK", dispatch_plan.input_slice_count),
            integer("GS", QUANTIZATION_GROUP_SIZE),
            integer("EDGE", i32::from(dispatch_plan.last_block_passes_padding)),
        ],
    )?;
    let projected = outputs
        .pop()
        .ok_or(RowBatchedProjectionError::MissingOutput)?;
    let mut output_shape = rows_shape;
    *output_shape.last_mut().expect("rows have a width") = output_width;
    Ok(runtime.reshape(&projected, &output_shape)?)
}

/// Whether the kernel takes this call: the packing it reads and a row count
/// and shape it tiles. Callers take MLX's quantized matmul otherwise.
pub fn row_batched_quantized_matmul_accepts(
    rows: &MlxArray,
    packed_weight: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
) -> bool {
    let Ok((output_width, input_width)) = check_quantization(packed_weight, scales, biases) else {
        return false;
    };
    let rows_shape = rows.shape();
    let row_count: i32 = rows_shape[..rows_shape.len().saturating_sub(1)]
        .iter()
        .product();
    rows.dtype() == MlxDtype::BFloat16
        && rows_shape.last() == Some(&input_width)
        && RowBatchedDispatchPlan::plan(row_count, output_width, input_width).is_ok()
}

/// How tall the group sums are and the grid that writes them.
pub struct GroupSumLayout {
    pub padded_row_count: i32,
    pub grid: [i32; 3],
    pub threadgroup: [i32; 3],
}

/// Each row's sum over each 64-value group, `[input / 64, padded rows]`
/// float32, zero past the real rows.
pub fn group_sums(
    runtime: &MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    flat_rows: &MlxArray,
    row_dimensions: &MlxArray,
    layout: GroupSumLayout,
    input_width: i32,
) -> Result<MlxArray, RowBatchedProjectionError> {
    let group_count = input_width / QUANTIZATION_GROUP_SIZE;
    let mut outputs = runtime.apply_metal_kernel(
        &kernels.group_sum,
        &[flat_rows, row_dimensions],
        &[MlxMetalKernelOutput::new(
            vec![group_count, layout.padded_row_count],
            MlxDtype::Float32,
        )],
        layout.grid,
        layout.threadgroup,
        &[
            integer("K", input_width),
            integer("GS", QUANTIZATION_GROUP_SIZE),
        ],
    )?;
    outputs
        .pop()
        .ok_or(RowBatchedProjectionError::MissingOutput)
}

pub fn integer(name: &'static str, value: i32) -> MlxMetalKernelTemplateArgument {
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

/// The `(output, input)` widths of a 4-bit, group-64 projection with bf16
/// scales and biases, or why the tensors are some other packing. The word
/// count alone cannot tell 4-bit group 64 from other packings (8-bit has
/// twice the words, group 32 twice the scales), so the scales' shape and every
/// dtype are checked and anything else is refused rather than misread.
pub fn check_quantization(
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
    check_weight_shape(output_width, input_width)?;
    Ok((output_width, input_width))
}

fn unsupported(description: String) -> RowBatchedProjectionError {
    RowBatchedProjectionError::UnsupportedQuantization { description }
}

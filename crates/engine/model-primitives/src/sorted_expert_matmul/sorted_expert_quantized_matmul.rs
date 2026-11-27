//! Rows sorted by expert times each row's expert's 4-bit affine weights,
//! read straight from MLX's own packed stack: the multiply MLX's
//! `gather_qmm` makes for sorted right-hand indices, with the same bits.

use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::metal_kernel_declaration::declare_metal_kernel;
use crate::metal_kernel_template::integer;
use crate::metal_kernels::{
    ASTRO_PRIMITIVE_SORTED_EXPERT_FIRST_ROWS, ASTRO_PRIMITIVE_SORTED_EXPERT_MATMUL_4BIT,
    ASTRO_PRIMITIVE_SORTED_EXPERT_MATMUL_4BIT_TOKEN_ROWS,
};

use super::dispatch_plan::{
    QUANTIZATION_GROUP_SIZE, SortedExpertDispatchPlan, SortedExpertDispatchRefusal,
    SortedExpertLayout,
};

/// The kernels, declared once and shared by every expert layer: the expert
/// search, and the multiply reading sorted rows or token rows.
pub struct SortedExpertMatmulKernels {
    expert_first_rows: MlxMetalKernel,
    matmul: MlxMetalKernel,
    token_rows_matmul: MlxMetalKernel,
    /// `[rows]` int32 arrays already uploaded, by row count: a prompt chunk
    /// makes three multiplies a layer at one row count, and each upload is a
    /// host-to-GPU round the CPU pays while the GPU waits.
    row_count_arrays: std::cell::RefCell<std::collections::HashMap<i32, MlxArray>>,
}

/// The kernel handles are opaque MLX objects; a model that holds them only
/// says that it does.
impl std::fmt::Debug for SortedExpertMatmulKernels {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SortedExpertMatmulKernels")
    }
}

impl SortedExpertMatmulKernels {
    /// Declares both kernels. MLX compiles a custom kernel's source on its
    /// first evaluation, so this succeeds on chips without M5 matrix units
    /// too: serving must prove the kernel by evaluating it first.
    pub fn declare() -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            expert_first_rows: declare_metal_kernel(
                &ASTRO_PRIMITIVE_SORTED_EXPERT_FIRST_ROWS,
                &[],
            )?,
            matmul: declare_metal_kernel(&ASTRO_PRIMITIVE_SORTED_EXPERT_MATMUL_4BIT, &[])?,
            token_rows_matmul: declare_metal_kernel(
                &ASTRO_PRIMITIVE_SORTED_EXPERT_MATMUL_4BIT_TOKEN_ROWS,
                &[],
            )?,
            row_count_arrays: std::cell::RefCell::default(),
        })
    }

    /// The `[row_count]` array the kernels read, uploaded once per count.
    fn row_count_array(
        &self,
        runtime: &MlxRuntime,
        row_count: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let mut uploaded = self.row_count_arrays.borrow_mut();
        if let Some(row_count_array) = uploaded.get(&row_count) {
            return Ok(row_count_array.retain()?);
        }
        let row_count_array = runtime.array_from_i32(&[row_count], &[1])?;
        uploaded.insert(row_count, row_count_array.retain()?);
        Ok(row_count_array)
    }
}

/// MLX's packed expert stack: `packed_weight` [experts, output, input / 8]
/// uint32 with its bf16 `scales` and `biases` [experts, output, input / 64].
pub struct ExpertStack<'stack> {
    pub packed_weight: &'stack MlxArray,
    pub scales: &'stack MlxArray,
    pub biases: &'stack MlxArray,
}

/// The layout the measurements chose (`cargo measure-sorted-expert-matmul-verdict`):
/// 128 columns, staged rows and 96 rows a pass took the least GPU time on a
/// 35B-A3B gate projection over a prompt chunk.
pub const SORTED_EXPERT_LAYOUT: SortedExpertLayout = SortedExpertLayout {
    simdgroups_across_columns: 4,
    stages_rows: true,
    rows_per_pass: 96,
    splits_experts_into_tiles: true,
};

/// `rows` [..., input] bf16, sorted by `sorted_expert_ids` [rows], times
/// each row's expert in `experts`, transposed: [..., output] bf16, the values
/// MLX's `gather_qmm` returns with `transpose = true`, `sorted_indices = true`
/// and right-hand indices only.
pub fn sorted_expert_quantized_matmul(
    runtime: &MlxRuntime,
    kernels: &SortedExpertMatmulKernels,
    rows: &MlxArray,
    experts: &ExpertStack<'_>,
    sorted_expert_ids: &MlxArray,
) -> Result<MlxArray, SortedExpertProjectionError> {
    sorted_expert_quantized_matmul_with_layout(
        runtime,
        kernels,
        (rows, sorted_expert_ids),
        experts,
        SORTED_EXPERT_LAYOUT,
    )
}

/// `sorted_expert_quantized_matmul` with a chosen tile layout, for the
/// measurements that compare layouts; every layout gives the same bits.
pub fn sorted_expert_quantized_matmul_with_layout(
    runtime: &MlxRuntime,
    kernels: &SortedExpertMatmulKernels,
    (rows, sorted_expert_ids): (&MlxArray, &MlxArray),
    experts: &ExpertStack<'_>,
    layout: SortedExpertLayout,
) -> Result<MlxArray, SortedExpertProjectionError> {
    let (expert_count, output_width, input_width) = check_expert_stack(experts)?;
    let (rows_shape, row_count) = bf16_row_count(rows, input_width, "rows")?;
    let plan =
        SortedExpertDispatchPlan::plan(row_count, output_width, input_width, expert_count, layout)?;
    let flat_rows = runtime.reshape(rows, &[row_count, input_width])?;
    let row_count_array = kernels.row_count_array(runtime, row_count)?;
    let first_rows = expert_first_rows(
        runtime,
        kernels,
        sorted_expert_ids,
        &row_count_array,
        expert_count,
        &plan,
    )?;
    let launch = MatmulLaunch {
        plan,
        layout,
        widths: (expert_count, output_width, input_width),
        row_count,
        template_extras: Vec::new(),
    };
    let projected = launch.run(
        runtime,
        &kernels.matmul,
        &[
            &flat_rows,
            experts.packed_weight,
            experts.scales,
            experts.biases,
            &first_rows,
            &row_count_array,
        ],
    )?;
    let mut output_shape = rows_shape;
    *output_shape.last_mut().expect("rows have a width") = output_width;
    Ok(runtime.reshape(&projected, &output_shape)?)
}

/// `token_rows` [tokens, ..., input] bf16, one row per token, projected for
/// every one of `sorted_order`'s assignments [tokens x `experts_per_token`]
/// (the argsort of the flat expert ids), each by its expert in
/// `sorted_expert_ids`: [assignments, 1, output] bf16 in sorted order. The
/// same values as taking the token rows in sorted order and projecting them
/// with `sorted_expert_quantized_matmul`, without making that sorted copy.
pub fn sorted_expert_quantized_matmul_from_token_rows(
    runtime: &MlxRuntime,
    kernels: &SortedExpertMatmulKernels,
    (token_rows, sorted_order, sorted_expert_ids): (&MlxArray, &MlxArray, &MlxArray),
    experts_per_token: i32,
    experts: &ExpertStack<'_>,
) -> Result<MlxArray, SortedExpertProjectionError> {
    let (expert_count, output_width, input_width) = check_expert_stack(experts)?;
    let (_, token_count) = bf16_row_count(token_rows, input_width, "token rows")?;
    let row_count = token_count * experts_per_token;
    let plan = SortedExpertDispatchPlan::plan(
        row_count,
        output_width,
        input_width,
        expert_count,
        TOKEN_ROWS_LAYOUT,
    )?;
    let flat_tokens = runtime.reshape(token_rows, &[token_count, input_width])?;
    let row_count_array = kernels.row_count_array(runtime, row_count)?;
    let first_rows = expert_first_rows(
        runtime,
        kernels,
        sorted_expert_ids,
        &row_count_array,
        expert_count,
        &plan,
    )?;
    let launch = MatmulLaunch {
        plan,
        layout: TOKEN_ROWS_LAYOUT,
        widths: (expert_count, output_width, input_width),
        row_count,
        template_extras: vec![integer("TOPK", experts_per_token)],
    };
    let projected = launch.run(
        runtime,
        &kernels.token_rows_matmul,
        &[
            &flat_tokens,
            experts.packed_weight,
            experts.scales,
            experts.biases,
            &first_rows,
            sorted_order,
            &row_count_array,
        ],
    )?;
    Ok(runtime.reshape(&projected, &[row_count, 1, output_width])?)
}

/// The token-row variant reads its rows through the sort only while staging
/// them, so it always stages, whatever `SORTED_EXPERT_LAYOUT` becomes. Served
/// on the 35B-A3B (`cargo measure-sorted-expert-matmul-served-chunk`), 96-row
/// tiles split across threadgroups read a 2,048-token chunk 1.0% faster than
/// MLX's route (11 of 12 rounds); 64-row tiles were within the spread and a
/// two-simdgroup tile 4% slower.
pub const TOKEN_ROWS_LAYOUT: SortedExpertLayout = SortedExpertLayout {
    simdgroups_across_columns: 4,
    stages_rows: true,
    rows_per_pass: 96,
    splits_experts_into_tiles: true,
};

/// One launch of a multiply kernel: its plan, layout and widths, and the
/// template integers a variant adds (the token-row variant's `TOPK`).
struct MatmulLaunch {
    plan: SortedExpertDispatchPlan,
    layout: SortedExpertLayout,
    /// (experts, output, input).
    widths: (i32, i32, i32),
    row_count: i32,
    template_extras: Vec<MlxMetalKernelTemplateArgument>,
}

impl MatmulLaunch {
    fn run(
        self,
        runtime: &MlxRuntime,
        kernel: &MlxMetalKernel,
        inputs: &[&MlxArray],
    ) -> Result<MlxArray, SortedExpertProjectionError> {
        let (expert_count, output_width, input_width) = self.widths;
        let mut template_arguments = vec![
            integer("N", output_width),
            integer("K", input_width),
            integer("E", expert_count),
            integer("BM", self.layout.rows_per_pass),
            integer("WN", self.layout.simdgroups_across_columns),
            integer("STAGE_ROWS", i32::from(self.layout.stages_rows)),
            integer(
                "TILE_SCHEDULE",
                i32::from(self.layout.splits_experts_into_tiles),
            ),
        ];
        template_arguments.extend(self.template_extras);
        let mut outputs = runtime.apply_metal_kernel(
            kernel,
            inputs,
            &[MlxMetalKernelOutput::new(
                vec![self.row_count, output_width],
                MlxDtype::BFloat16,
            )],
            self.plan.grid,
            self.plan.threadgroup,
            &template_arguments,
        )?;
        outputs
            .pop()
            .ok_or(SortedExpertProjectionError::MissingOutput)
    }
}

/// Each expert's first row in the sorted order, `[experts]` int32.
fn expert_first_rows(
    runtime: &MlxRuntime,
    kernels: &SortedExpertMatmulKernels,
    sorted_expert_ids: &MlxArray,
    row_count_array: &MlxArray,
    expert_count: i32,
    plan: &SortedExpertDispatchPlan,
) -> Result<MlxArray, SortedExpertProjectionError> {
    let mut outputs = runtime.apply_metal_kernel(
        &kernels.expert_first_rows,
        &[sorted_expert_ids, row_count_array],
        &[MlxMetalKernelOutput::new(
            vec![expert_count],
            MlxDtype::Int32,
        )],
        plan.first_row_grid,
        plan.first_row_threadgroup,
        &[integer("E", expert_count)],
    )?;
    outputs
        .pop()
        .ok_or(SortedExpertProjectionError::MissingOutput)
}

/// The shape of `rows` and how many rows it holds, when they are bf16 rows
/// `input_width` wide; the kernel refuses anything else rather than misread it.
fn bf16_row_count(
    rows: &MlxArray,
    input_width: i32,
    what_they_are: &str,
) -> Result<(Vec<i32>, i32), SortedExpertProjectionError> {
    let shape = rows.shape();
    match shape.split_last() {
        Some((width, leading)) if *width == input_width && rows.dtype() == MlxDtype::BFloat16 => {
            let row_count = leading.iter().product();
            Ok((shape, row_count))
        }
        _ => Err(unsupported(format!(
            "{what_they_are} are {:?} {shape:?}, expected bf16 rows {input_width} wide",
            rows.dtype()
        ))),
    }
}

/// Why a sorted-expert multiply failed.
#[derive(Debug, thiserror::Error)]
pub enum SortedExpertProjectionError {
    #[error(transparent)]
    Refused(#[from] SortedExpertDispatchRefusal),
    #[error(transparent)]
    Mlx(#[from] mlx_bridge::MlxCError),
    #[error(transparent)]
    Runtime(#[from] MlxRuntimeError),
    #[error("the sorted-expert kernel returned no output")]
    MissingOutput,
    #[error(
        "the sorted-expert kernel reads only 4-bit, group-64 expert stacks with bf16 scales: {description}"
    )]
    UnsupportedQuantization { description: String },
}

fn unsupported(description: String) -> SortedExpertProjectionError {
    SortedExpertProjectionError::UnsupportedQuantization { description }
}

/// The `(experts, output, input)` widths of a 4-bit, group-64 expert stack
/// with bf16 scales and biases. The word count alone cannot tell 4-bit
/// group 64 from other packings, so the scales' shape and every dtype are
/// checked and anything else is refused rather than misread.
fn check_expert_stack(
    experts: &ExpertStack<'_>,
) -> Result<(i32, i32, i32), SortedExpertProjectionError> {
    let weight_shape = experts.packed_weight.shape();
    let scale_shape = experts.scales.shape();
    let [expert_count, output_width, packed_words] = weight_shape[..] else {
        return Err(unsupported(format!(
            "packed weight is {weight_shape:?}, expected [experts, output, input / 8]"
        )));
    };
    let input_width = packed_words * 8;
    let expected_scales = vec![
        expert_count,
        output_width,
        input_width / QUANTIZATION_GROUP_SIZE,
    ];
    let dtypes_match = experts.packed_weight.dtype() == MlxDtype::UInt32
        && experts.scales.dtype() == MlxDtype::BFloat16
        && experts.biases.dtype() == MlxDtype::BFloat16;
    if !dtypes_match || scale_shape != expected_scales || experts.biases.shape() != expected_scales
    {
        return Err(unsupported(format!(
            "weight {weight_shape:?}, scales {scale_shape:?} {:?}, expected 4-bit group-64 scales {expected_scales:?} bf16",
            experts.scales.dtype()
        )));
    }
    Ok((expert_count, output_width, input_width))
}

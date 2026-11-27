//! The tiled sorted-expert multiplies a prompt chunk's routed experts take:
//! gate and up with their SwiGLU in one kernel reading the token rows in
//! place, and the down projection over the sorted activations, both with
//! MLX's bits (see `astro_primitive_sorted_expert_tiled_matmul.h` for the
//! schedule and why it is faster).
//!
//! Each multiply first cuts its sorted rows into single-expert row tiles
//! (`astro_primitive_sorted_expert_tile_list`); the multiply then runs one
//! threadgroup per (row tile, 64-column block).

use std::cell::RefCell;
use std::collections::HashMap;

use mlx_bridge::{MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::metal_kernel_declaration::declare_metal_kernel;
use crate::metal_kernel_template::integer;
use crate::metal_kernels::{
    ASTRO_PRIMITIVE_SORTED_EXPERT_GATE_UP_SWIGLU_4BIT_TOKEN_ROWS,
    ASTRO_PRIMITIVE_SORTED_EXPERT_TILE_LIST, ASTRO_PRIMITIVE_SORTED_EXPERT_TILED_MATMUL_4BIT,
};

use super::sorted_expert_quantized_matmul::{
    ExpertStack, SortedExpertProjectionError, bf16_row_count, check_expert_stack, last_output,
    unsupported, uploaded_i32_array,
};

/// Output columns a threadgroup computes (two 32-column simdgroups).
const TILE_COLUMNS: i32 = 64;
/// Threads of the one threadgroup that builds a tile list.
const TILE_LIST_THREADS: i32 = 1_024;
/// Weight steps are one 64-value quantization group deep.
const STEP_DEPTH: i32 = 64;

/// How a multiply's rows are tiled and its threadgroups laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledExpertSchedule {
    /// Rows of one expert a tile holds: 64 or 96 (32 a simdgroup).
    pub rows_per_tile: i32,
    /// 0: grid (column block, tile). Otherwise the tile on the grid's x
    /// axis in groups of this many tiles, so a row tile's column blocks run
    /// next to each other and share its rows in the GPU's caches.
    pub tiles_per_grid_group: i32,
}

/// Gate and up over a 2,048-token chunk of a 35B-A3B layer (64 rows an
/// expert, K 2,048): 96-row tiles on the grid's x axis, the configuration
/// oMLX's planner picks at 48-95 rows an expert and K of 1,024 or more.
pub const GATE_UP_SCHEDULE: TiledExpertSchedule = TiledExpertSchedule {
    rows_per_tile: 96,
    tiles_per_grid_group: 32,
};

/// The down projection (K 512): 64-row tiles in the plain layout, where
/// streaming the weights dominates below 120 rows an expert.
pub const DOWN_SCHEDULE: TiledExpertSchedule = TiledExpertSchedule {
    rows_per_tile: 64,
    tiles_per_grid_group: 0,
};

/// The three kernels, declared once and shared by every expert layer.
pub struct TiledExpertKernels {
    tile_list: MlxMetalKernel,
    gate_up_swiglu: MlxMetalKernel,
    matmul: MlxMetalKernel,
    /// `[rows, max tiles]` int32 arrays already uploaded: a chunk makes the
    /// same lists every layer, and each upload is a host round the GPU waits on.
    bounds_arrays: RefCell<HashMap<(i32, i32), MlxArray>>,
}

impl std::fmt::Debug for TiledExpertKernels {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("TiledExpertKernels")
    }
}

impl TiledExpertKernels {
    /// Declares the kernels. MLX compiles a custom kernel on its first
    /// evaluation, so serving proves them first through the
    /// `sorted_expert_tiled_matmul` probe.
    pub fn declare() -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            tile_list: declare_metal_kernel(&ASTRO_PRIMITIVE_SORTED_EXPERT_TILE_LIST, &[])?,
            gate_up_swiglu: declare_metal_kernel(
                &ASTRO_PRIMITIVE_SORTED_EXPERT_GATE_UP_SWIGLU_4BIT_TOKEN_ROWS,
                &[],
            )?,
            matmul: declare_metal_kernel(&ASTRO_PRIMITIVE_SORTED_EXPERT_TILED_MATMUL_4BIT, &[])?,
            bounds_arrays: RefCell::default(),
        })
    }

    fn bounds_array(
        &self,
        runtime: &MlxRuntime,
        row_count: i32,
        maximum_tile_count: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        uploaded_i32_array(
            &self.bounds_arrays,
            runtime,
            (row_count, maximum_tile_count),
            &[row_count, maximum_tile_count],
        )
    }
}

/// A multiply's row tiles: the list, its length, and the most it can hold.
struct RowTiles {
    tiles: MlxArray,
    tile_count: MlxArray,
    maximum_tile_count: i32,
}

/// `token_rows` [tokens, ..., input] bf16 projected through `gate` and `up`
/// for every one of `sorted_order`'s assignments, each by its expert in
/// `sorted_expert_ids`, and joined as MLX's compiled
/// `gate * sigmoid(gate) * up`: [assignments, 1, output] bf16 in sorted
/// order, the bits of MLX's two sorted `gather_qmm`s on the sorted copy of
/// the rows followed by the compiled SwiGLU.
pub fn sorted_expert_swiglu_from_token_rows(
    runtime: &MlxRuntime,
    kernels: &TiledExpertKernels,
    (token_rows, sorted_order, sorted_expert_ids): (&MlxArray, &MlxArray, &MlxArray),
    experts_per_token: i32,
    (gate, up): (&ExpertStack<'_>, &ExpertStack<'_>),
) -> Result<MlxArray, SortedExpertProjectionError> {
    let widths = check_expert_stack(gate)?;
    if check_expert_stack(up)? != widths {
        return Err(unsupported(
            "the gate and up stacks differ in shape".to_owned(),
        ));
    }
    let (expert_count, output_width, input_width) = widths;
    check_tileable(output_width * 2, input_width)?;
    let (_, token_count) = bf16_row_count(token_rows, input_width, "token rows")?;
    let row_count = token_count * experts_per_token;
    let flat_tokens = runtime.reshape(token_rows, &[token_count, input_width])?;
    let row_tiles = row_tiles(
        runtime,
        kernels,
        sorted_expert_ids,
        (row_count, expert_count),
        GATE_UP_SCHEDULE,
    )?;
    let activated = run_tiled_kernel(
        runtime,
        &kernels.gate_up_swiglu,
        &[
            &flat_tokens,
            gate.packed_weight,
            gate.scales,
            gate.biases,
            up.packed_weight,
            up.scales,
            up.biases,
            &row_tiles.tiles,
            &row_tiles.tile_count,
            sorted_order,
        ],
        &TiledLaunch {
            schedule: GATE_UP_SCHEDULE,
            row_count,
            output_width,
            input_width,
            column_blocks: output_width * 2 / TILE_COLUMNS,
            maximum_tile_count: row_tiles.maximum_tile_count,
            experts_per_token: Some(experts_per_token),
        },
    )?;
    Ok(runtime.reshape(&activated, &[row_count, 1, output_width])?)
}

/// `rows` [..., input] bf16, sorted by `sorted_expert_ids`, times each row's
/// expert in `experts`, transposed: [..., output] bf16 with the bits of
/// MLX's sorted `gather_qmm`.
pub fn tiled_sorted_expert_matmul(
    runtime: &MlxRuntime,
    kernels: &TiledExpertKernels,
    rows: &MlxArray,
    experts: &ExpertStack<'_>,
    sorted_expert_ids: &MlxArray,
) -> Result<MlxArray, SortedExpertProjectionError> {
    let (expert_count, output_width, input_width) = check_expert_stack(experts)?;
    check_tileable(output_width, input_width)?;
    let (rows_shape, row_count) = bf16_row_count(rows, input_width, "rows")?;
    let flat_rows = runtime.reshape(rows, &[row_count, input_width])?;
    let row_tiles = row_tiles(
        runtime,
        kernels,
        sorted_expert_ids,
        (row_count, expert_count),
        DOWN_SCHEDULE,
    )?;
    let down_inputs = [
        &flat_rows,
        experts.packed_weight,
        experts.scales,
        experts.biases,
        &row_tiles.tiles,
        &row_tiles.tile_count,
    ];
    let projected = run_tiled_kernel(
        runtime,
        &kernels.matmul,
        &down_inputs,
        &TiledLaunch {
            schedule: DOWN_SCHEDULE,
            row_count,
            output_width,
            input_width,
            column_blocks: output_width / TILE_COLUMNS,
            maximum_tile_count: row_tiles.maximum_tile_count,
            experts_per_token: None,
        },
    )?;
    let mut output_shape = rows_shape;
    *output_shape.last_mut().expect("rows have a width") = output_width;
    Ok(runtime.reshape(&projected, &output_shape)?)
}

/// Refuses widths the tiles cannot cover exactly: `weight_rows` (2 x output
/// for a paired gate/up) a multiple of 64 and the input a positive multiple
/// of the 64-deep step.
fn check_tileable(weight_rows: i32, input_width: i32) -> Result<(), SortedExpertProjectionError> {
    if weight_rows <= 0
        || weight_rows % TILE_COLUMNS != 0
        || input_width <= 0
        || input_width % STEP_DEPTH != 0
    {
        return Err(unsupported(format!(
            "{weight_rows} weight rows and {input_width} inputs: the tiled kernels need multiples of 64"
        )));
    }
    Ok(())
}

/// The row tiles of `row_count` rows sorted over `expert_count` experts:
/// at most one partial tile an expert, so `ceil(rows / BM) + min(E, rows)`
/// bounds the list.
fn row_tiles(
    runtime: &MlxRuntime,
    kernels: &TiledExpertKernels,
    sorted_expert_ids: &MlxArray,
    (row_count, expert_count): (i32, i32),
    schedule: TiledExpertSchedule,
) -> Result<RowTiles, SortedExpertProjectionError> {
    if row_count <= 0 || expert_count <= 0 {
        return Err(unsupported(format!(
            "{row_count} rows over {expert_count} experts: both must be positive"
        )));
    }
    let maximum_tile_count = (row_count + schedule.rows_per_tile - 1) / schedule.rows_per_tile
        + expert_count.min(row_count);
    let bounds = kernels.bounds_array(runtime, row_count, maximum_tile_count)?;
    let mut outputs = runtime.apply_metal_kernel(
        &kernels.tile_list,
        &[sorted_expert_ids, &bounds],
        &[
            MlxMetalKernelOutput::new(vec![maximum_tile_count * 4], MlxDtype::UInt32),
            MlxMetalKernelOutput::new(vec![1], MlxDtype::UInt32),
        ],
        [TILE_LIST_THREADS, 1, 1],
        [TILE_LIST_THREADS, 1, 1],
        &[
            integer("BM", schedule.rows_per_tile),
            integer("E", expert_count),
        ],
    )?;
    let tile_count = last_output(&mut outputs)?;
    let tiles = last_output(&mut outputs)?;
    Ok(RowTiles {
        tiles,
        tile_count,
        maximum_tile_count,
    })
}

/// One launch of a tiled multiply.
struct TiledLaunch {
    schedule: TiledExpertSchedule,
    row_count: i32,
    output_width: i32,
    input_width: i32,
    column_blocks: i32,
    maximum_tile_count: i32,
    /// The token-row kernel's experts a token; `None` for sorted rows.
    experts_per_token: Option<i32>,
}

impl TiledLaunch {
    /// Threadgroups (32 lanes x 2 column simdgroups x one row simdgroup a
    /// 32 rows) over every (tile, column block); tiles past the list's
    /// length return at once.
    fn grid_and_threadgroup(&self) -> ([i32; 3], [i32; 3]) {
        let row_simdgroups = self.schedule.rows_per_tile / 32;
        let group = self.schedule.tiles_per_grid_group;
        let (threadgroups_across, threadgroups_down) = if group > 0 {
            (
                group,
                (self.maximum_tile_count + group - 1) / group * self.column_blocks,
            )
        } else {
            (self.column_blocks, self.maximum_tile_count)
        };
        (
            [
                threadgroups_across * 32,
                threadgroups_down * 2,
                row_simdgroups,
            ],
            [32, 2, row_simdgroups],
        )
    }
}

fn run_tiled_kernel(
    runtime: &MlxRuntime,
    kernel: &MlxMetalKernel,
    inputs: &[&MlxArray],
    launch: &TiledLaunch,
) -> Result<MlxArray, SortedExpertProjectionError> {
    let mut template_arguments = vec![
        integer("N", launch.output_width),
        integer("K", launch.input_width),
        integer("GX", launch.schedule.tiles_per_grid_group),
    ];
    if let Some(experts_per_token) = launch.experts_per_token {
        template_arguments.push(integer("TOPK", experts_per_token));
    }
    let (grid, threadgroup) = launch.grid_and_threadgroup();
    let mut outputs = runtime.apply_metal_kernel(
        kernel,
        inputs,
        &[MlxMetalKernelOutput::new(
            vec![launch.row_count, launch.output_width],
            MlxDtype::BFloat16,
        )],
        grid,
        threadgroup,
        &template_arguments,
    )?;
    last_output(&mut outputs)
}

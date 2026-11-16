//! How one row-batched matmul call is laid out on the GPU: pure arithmetic,
//! so every shape rule is unit-tested without a GPU.

use thiserror::Error;

/// Output columns one simdgroup produces: two 16-wide matrix-unit fragments.
pub(crate) const OUTPUT_COLUMNS_PER_SIMDGROUP: i32 = 32;
/// Rows one threadgroup covers at most; more rows take more threadgroups.
const ROWS_PER_THREADGROUP_LIMIT: i32 = 32;
/// Rows the kernel accepts in one call. Drafting verifies at most 16 rows.
/// Measured on an M5 Pro over the 27B's projections, 32 rows already cost
/// 1.66x one row here, so prompt chunks (hundreds of rows) stay on MLX's
/// prefill kernels rather than looping this one.
pub const ROW_BATCHED_MATMUL_MAXIMUM_ROWS: i32 = 32;
pub(crate) const QUANTIZATION_GROUP_SIZE: i32 = 64;
const SIMDGROUP_WIDTH: i32 = 32;
/// Input slices stop doubling here: the slices after the first each park two
/// fragments' partial sums in threadgroup memory, 7 x 2 x 8 x 32 floats =
/// 14 KB at eight slices, within the 32 KB a threadgroup may use.
const MAXIMUM_INPUT_SLICES: i32 = 8;
/// Slicing stops once a shape already has this many simdgroups in flight.
/// This is TensorFold's figure for the same kernel; on the M5 Pro it gave
/// the 1.27x 16-row result, and it is not tuned per chip.
const SATURATING_SIMDGROUP_COUNT: i32 = 1024;
/// Each slice keeps at least this many groups, so the per-slice setup and the
/// final reduction stay small against the streamed weights.
const MINIMUM_GROUPS_PER_SLICE: i32 = 8;

/// Why a shape or row count cannot use the row-batched kernel. Callers fall
/// back to MLX's own quantized matmul for these.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RowBatchedDispatchRefusal {
    #[error("{row_count} rows is outside 1 to {ROW_BATCHED_MATMUL_MAXIMUM_ROWS}")]
    RowCount { row_count: i32 },
    #[error(
        "input width {input_width} is not a positive multiple of four {QUANTIZATION_GROUP_SIZE}-value groups"
    )]
    InputWidth { input_width: i32 },
    #[error(
        "output width {output_width} is not a positive multiple of {OUTPUT_COLUMNS_PER_SIMDGROUP}"
    )]
    OutputWidth { output_width: i32 },
}

/// Everything one dispatch needs, derived from the shape and row count only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowBatchedDispatchPlan {
    pub(crate) row_count: i32,
    /// Rows rounded up to whole 16-row matrix-unit blocks.
    pub(crate) padded_row_count: i32,
    /// 16-row blocks one threadgroup handles.
    pub(crate) row_blocks_per_threadgroup: i32,
    pub(crate) input_slice_count: i32,
    /// True when the last threadgroup's rows pass the padded row count.
    pub(crate) last_block_passes_padding: bool,
    pub(crate) grid: [i32; 3],
    pub(crate) threadgroup: [i32; 3],
    pub(crate) group_sum_grid: [i32; 3],
    pub(crate) group_sum_threadgroup: [i32; 3],
}

impl RowBatchedDispatchPlan {
    pub(crate) fn plan(
        row_count: i32,
        output_width: i32,
        input_width: i32,
    ) -> Result<Self, RowBatchedDispatchRefusal> {
        if !(1..=ROW_BATCHED_MATMUL_MAXIMUM_ROWS).contains(&row_count) {
            return Err(RowBatchedDispatchRefusal::RowCount { row_count });
        }
        check_weight_shape(output_width, input_width)?;
        let padded_row_count = 16 * ((row_count + 15) / 16);
        let rows_per_threadgroup = padded_row_count.min(ROWS_PER_THREADGROUP_LIMIT);
        let input_slice_count = input_slice_count(output_width, input_width);
        let column_tile_count = output_width / OUTPUT_COLUMNS_PER_SIMDGROUP;
        let row_threadgroup_count =
            (padded_row_count + rows_per_threadgroup - 1) / rows_per_threadgroup;
        let group_count = input_width / QUANTIZATION_GROUP_SIZE;
        Ok(Self {
            row_count,
            padded_row_count,
            row_blocks_per_threadgroup: rows_per_threadgroup / 16,
            input_slice_count,
            last_block_passes_padding: padded_row_count % rows_per_threadgroup != 0,
            grid: [
                column_tile_count * SIMDGROUP_WIDTH * input_slice_count,
                row_threadgroup_count,
                1,
            ],
            threadgroup: [SIMDGROUP_WIDTH * input_slice_count, 1, 1],
            group_sum_grid: [group_count, padded_row_count, 1],
            group_sum_threadgroup: [group_count.min(256), 1, 1],
        })
    }
}

/// Whether the kernel can tile an `[output, input]` weight at all: whole
/// 32-column tiles and whole 64-value groups. The 27B's 48-wide DeltaNet gate
/// projections fail this and stay on MLX's matmul, where they cost little.
pub(crate) fn check_weight_shape(
    output_width: i32,
    input_width: i32,
) -> Result<(), RowBatchedDispatchRefusal> {
    if input_width <= 0 || input_width % QUANTIZATION_GROUP_SIZE != 0 {
        return Err(RowBatchedDispatchRefusal::InputWidth { input_width });
    }
    if output_width <= 0 || output_width % OUTPUT_COLUMNS_PER_SIMDGROUP != 0 {
        return Err(RowBatchedDispatchRefusal::OutputWidth { output_width });
    }
    Ok(())
}

/// Input slices for an `[output, input]` weight. It depends on the shape only,
/// never on the row count, so a row's result is the same at every row count.
/// Narrow, long weights (the 5,120-wide down projection reading 17,408 inputs)
/// have few column tiles, so they get the most slices to fill the GPU.
pub(crate) fn input_slice_count(output_width: i32, input_width: i32) -> i32 {
    let column_tile_count =
        (output_width + OUTPUT_COLUMNS_PER_SIMDGROUP - 1) / OUTPUT_COLUMNS_PER_SIMDGROUP;
    let group_count = input_width / QUANTIZATION_GROUP_SIZE;
    let mut slice_count = 1;
    while slice_count < MAXIMUM_INPUT_SLICES
        && column_tile_count * slice_count < SATURATING_SIMDGROUP_COUNT
        && group_count / (slice_count * 2) >= MINIMUM_GROUPS_PER_SLICE
    {
        slice_count *= 2;
    }
    slice_count
}

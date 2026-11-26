//! How one register-ordered projection call is laid out on the GPU: pure
//! arithmetic, unit-tested without a GPU.

use super::dispatch_plan::{
    OUTPUT_COLUMNS_PER_SIMDGROUP, QUANTIZATION_GROUP_SIZE, RowBatchedDispatchRefusal,
    check_weight_shape, input_slice_count,
};

/// Rows one dispatch takes: eight 16-row blocks on the grid's second axis.
/// TensorFold's figure (`reg_rows`); a prompt chunk with more rows takes
/// several dispatches, each reading the weight once for its 128 rows.
pub const REGISTER_ORDERED_ROWS_PER_DISPATCH: i32 = 128;
/// The fewest rows for which a pass restores MLX's layout for itself and
/// multiplies with MLX's own kernel (`restored_layout_matmul`).
///
/// A prompt chunk's bits need not match a reply token's, so a chunk is free
/// to take whichever kernel is fastest, and for many rows that is MLX's on
/// MLX's layout. Restoring costs one read and one write of the projection,
/// the same whatever the rows, so it pays once the rows are many. Dense 27B
/// shapes on MLX 0.32.5, against MLX's layout
/// (`cargo measure-register-order-verdict`, twelve interleaved rounds):
///
/// | Rows  | Register-ordered kernel | Restored for the pass |
/// |-------|-------------------------|-----------------------|
/// | 256   | +13 to +29%             | +16 to +22%           |
/// | 512   | +13 to +33%             | +7 to +11%            |
/// | 1,024 | +12 to +38%             | +4 to +7%             |
/// | 2,048 | +11 to +38%             | +1 to +4%             |
pub const FEWEST_ROWS_A_PASS_RESTORES_MLX_LAYOUT_FOR: i32 = 512;

/// Whether a pass of `row_count` rows restores MLX's layout for itself.
#[must_use]
pub const fn pass_restores_mlx_layout(row_count: i32) -> bool {
    row_count >= FEWEST_ROWS_A_PASS_RESTORES_MLX_LAYOUT_FOR
}

const ROW_BLOCK_HEIGHT: i32 = 16;
const SIMDGROUP_WIDTH: i32 = 32;
/// Words of register-ordered codes per 32-column tile and 64-value group.
pub const REGISTER_BLOCK_WORDS: i32 = 256;

/// Input slices TensorFold measured fastest on an M5 for the Qwen 27B's
/// projection shapes `(output, input)`, streamed over every layer. Shapes not
/// listed take `input_slice_count`. Fused shapes are listed by their fused
/// width: 6,240 is the DeltaNet z | b | a stack and 2,048 attention's k | v.
const MEASURED_SLICE_COUNTS: [(i32, i32, i32); 6] = [
    (10240, 5120, 2),
    (6240, 5120, 4),
    (5120, 6144, 4),
    (12288, 5120, 2),
    (2048, 5120, 8),
    (5120, 17408, 4),
];

/// The input slices a register-ordered weight of this shape always takes.
/// It depends on the shape only, so a row's bits are the same at every row
/// count, one-row decode steps included.
pub fn register_ordered_slice_count(output_width: i32, input_width: i32) -> i32 {
    for (measured_output, measured_input, slice_count) in MEASURED_SLICE_COUNTS {
        if measured_output == output_width && measured_input == input_width {
            return slice_count;
        }
    }
    input_slice_count(output_width, input_width)
}

/// Everything one dispatch needs, from the shape, the slices and the rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterOrderedDispatchPlan {
    /// Rows rounded up to whole 16-row blocks; the group sums are this tall.
    pub padded_row_count: i32,
    pub grid: [i32; 3],
    pub threadgroup: [i32; 3],
    pub group_sum_grid: [i32; 3],
    pub group_sum_threadgroup: [i32; 3],
}

impl RegisterOrderedDispatchPlan {
    pub fn plan(
        row_count: i32,
        output_width: i32,
        input_width: i32,
        input_slice_count: i32,
    ) -> Result<Self, RowBatchedDispatchRefusal> {
        if !(1..=REGISTER_ORDERED_ROWS_PER_DISPATCH).contains(&row_count) {
            return Err(RowBatchedDispatchRefusal::RowCount { row_count });
        }
        check_weight_shape(output_width, input_width)?;
        let row_block_count = (row_count + ROW_BLOCK_HEIGHT - 1) / ROW_BLOCK_HEIGHT;
        let column_tile_count = output_width / OUTPUT_COLUMNS_PER_SIMDGROUP;
        let group_count = input_width / QUANTIZATION_GROUP_SIZE;
        Ok(Self {
            padded_row_count: row_block_count * ROW_BLOCK_HEIGHT,
            grid: [
                column_tile_count * SIMDGROUP_WIDTH * input_slice_count,
                row_block_count,
                1,
            ],
            threadgroup: [SIMDGROUP_WIDTH * input_slice_count, 1, 1],
            group_sum_grid: [group_count, row_block_count * ROW_BLOCK_HEIGHT, 1],
            group_sum_threadgroup: [group_count.min(256), 1, 1],
        })
    }
}

/// The conversion's grid: one 256-thread threadgroup per tile and group.
pub fn register_order_conversion_grid(output_width: i32, input_width: i32) -> ([i32; 3], [i32; 3]) {
    let block_count =
        (output_width / OUTPUT_COLUMNS_PER_SIMDGROUP) * (input_width / QUANTIZATION_GROUP_SIZE);
    (
        [block_count * REGISTER_BLOCK_WORDS, 1, 1],
        [REGISTER_BLOCK_WORDS, 1, 1],
    )
}

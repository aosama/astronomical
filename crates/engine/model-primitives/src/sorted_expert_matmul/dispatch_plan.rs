//! Which sorted-expert multiplies the kernel takes, and the grid it runs on.
//! Pure arithmetic, so it is tested without a GPU.

/// Output columns one simdgroup covers.
pub const OUTPUT_COLUMNS_PER_SIMDGROUP: i32 = 32;

/// How a threadgroup's tile is laid out. Every layout gives the same bits;
/// they differ in how often a token row or a weight block is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortedExpertLayout {
    /// Simdgroups across the output columns: the tile is 32 of these wide.
    pub simdgroups_across_columns: i32,
    /// Whether a step's token rows are staged once in threadgroup memory for
    /// every simdgroup, rather than read by each simdgroup from device memory.
    pub stages_rows: bool,
    /// Rows of its expert a threadgroup takes a pass, 64 or 96: an expert
    /// with no more rows than this has its weights dequantized once.
    pub rows_per_pass: i32,
    /// Whether each threadgroup takes one row tile of an expert (MLX's
    /// schedule: a popular expert's rows run on several threadgroups at once)
    /// rather than all of one expert's rows. A real router's experts are
    /// uneven, and one threadgroup working through a popular expert alone
    /// holds the whole multiply up.
    pub splits_experts_into_tiles: bool,
}

impl SortedExpertLayout {
    pub fn output_columns_per_threadgroup(self) -> i32 {
        OUTPUT_COLUMNS_PER_SIMDGROUP * self.simdgroups_across_columns
    }

    /// Two simdgroups down the rows, of 32 lanes each.
    pub fn threads_per_threadgroup(self) -> i32 {
        64 * self.simdgroups_across_columns
    }
}
/// The input depth of one step, and MLX's quantization group.
pub const QUANTIZATION_GROUP_SIZE: i32 = 64;
/// The expert search runs one thread per expert in groups this wide.
pub const FIRST_ROW_THREADS_PER_THREADGROUP: i32 = 64;

/// The launch of one sorted-expert multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortedExpertDispatchPlan {
    /// Threads, as MLX's custom kernels take a grid: one threadgroup per
    /// tile of output columns across, one per expert down.
    pub grid: [i32; 3],
    pub threadgroup: [i32; 3],
    pub first_row_grid: [i32; 3],
    pub first_row_threadgroup: [i32; 3],
}

/// Why the kernel does not take a multiply; the caller then runs MLX's.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SortedExpertDispatchRefusal {
    #[error("{output_width} output columns are not a multiple of the {tile_width}-column tile")]
    OutputWidth { output_width: i32, tile_width: i32 },
    #[error(
        "{simdgroups_across_columns} simdgroups across and {rows_per_pass} rows a pass: the kernel takes 2 or 4, and 64 or 96"
    )]
    Layout {
        simdgroups_across_columns: i32,
        rows_per_pass: i32,
    },
    #[error("{input_width} input values are not a positive multiple of {QUANTIZATION_GROUP_SIZE}")]
    InputWidth { input_width: i32 },
    #[error("{row_count} rows over {expert_count} experts: both must be positive")]
    EmptyMultiply { row_count: i32, expert_count: i32 },
}

impl SortedExpertDispatchPlan {
    pub fn plan(
        row_count: i32,
        output_width: i32,
        input_width: i32,
        expert_count: i32,
        layout: SortedExpertLayout,
    ) -> Result<Self, SortedExpertDispatchRefusal> {
        if row_count <= 0 || expert_count <= 0 {
            return Err(SortedExpertDispatchRefusal::EmptyMultiply {
                row_count,
                expert_count,
            });
        }
        if !matches!(layout.simdgroups_across_columns, 2 | 4)
            || !matches!(layout.rows_per_pass, 64 | 96)
        {
            return Err(SortedExpertDispatchRefusal::Layout {
                simdgroups_across_columns: layout.simdgroups_across_columns,
                rows_per_pass: layout.rows_per_pass,
            });
        }
        let tile_width = layout.output_columns_per_threadgroup();
        if output_width <= 0 || output_width % tile_width != 0 {
            return Err(SortedExpertDispatchRefusal::OutputWidth {
                output_width,
                tile_width,
            });
        }
        if input_width <= 0 || input_width % QUANTIZATION_GROUP_SIZE != 0 {
            return Err(SortedExpertDispatchRefusal::InputWidth { input_width });
        }
        Ok(Self {
            grid: [
                output_width / tile_width * layout.threads_per_threadgroup(),
                if layout.splits_experts_into_tiles {
                    row_tile_bound(row_count, expert_count, layout.rows_per_pass)
                } else {
                    expert_count
                },
                1,
            ],
            threadgroup: [layout.threads_per_threadgroup(), 1, 1],
            first_row_grid: [expert_count, 1, 1],
            first_row_threadgroup: [expert_count.min(FIRST_ROW_THREADS_PER_THREADGROUP), 1, 1],
        })
    }
}

/// The most row tiles any routing can need, as MLX bounds it: every expert
/// boundary can split one tile in two, and there are never more tiles than
/// rows. Threadgroups past the routing's real tile count return at once.
pub fn row_tile_bound(row_count: i32, expert_count: i32, rows_per_tile: i32) -> i32 {
    let whole_tiles = (row_count + rows_per_tile - 1) / rows_per_tile;
    row_count.min(whole_tiles + expert_count - 1)
}

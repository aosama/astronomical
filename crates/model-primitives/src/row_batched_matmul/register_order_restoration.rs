//! MLX's packed layout rebuilt from a register-ordered weight, for the few
//! readers that need it: Neural Engine programs dequantize the MLP weights
//! once when they are built, and decision readout reads a handful of output
//! head rows. Neither is on a decode step, so restoring costs nothing there.

use mlx_bridge::{MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput};
use runtime_integration::MlxRuntime;

use super::dispatch_plan::{OUTPUT_COLUMNS_PER_SIMDGROUP, QUANTIZATION_GROUP_SIZE};
use super::register_order_source::REGISTER_ORDER_RESTORATION_SOURCE;
use super::register_ordered_matmul::RegisterOrderedWeight;
use super::register_ordered_plan::REGISTER_BLOCK_WORDS;
use super::row_batched_quantized_matmul::{RowBatchedProjectionError, integer};

/// MLX's `(packed_weight, scales, biases)` for some rows of a 4-bit,
/// group-64 projection: [rows, input / 8] uint32 and [rows, input / 64] bf16.
pub struct RestoredMlxRows {
    pub packed_weight: MlxArray,
    pub scales: MlxArray,
    pub biases: MlxArray,
}

impl RestoredMlxRows {
    /// The given rows of all three arrays, in that order.
    fn rows(
        &self,
        runtime: &MlxRuntime,
        row_ids: &[i32],
    ) -> Result<Self, RowBatchedProjectionError> {
        let row_indices = runtime.array_from_i32(row_ids, &[row_ids.len() as i32])?;
        Ok(Self {
            packed_weight: runtime.take_axis(&self.packed_weight, &row_indices, 0)?,
            scales: runtime.take_axis(&self.scales, &row_indices, 0)?,
            biases: runtime.take_axis(&self.biases, &row_indices, 0)?,
        })
    }
}

impl RegisterOrderedWeight {
    /// The whole weight back in MLX's layout. Lazy; it holds a second copy
    /// of the weight while it lives, so callers drop it once used.
    pub fn restored_mlx_layout(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<RestoredMlxRows, RowBatchedProjectionError> {
        let tile_count = self.output_width / OUTPUT_COLUMNS_PER_SIMDGROUP;
        let tile_ids: Vec<i32> = (0..tile_count).collect();
        let restored = self.restored_tiles(runtime, &tile_ids)?;
        if self.logical_output_width == self.output_width {
            return Ok(restored);
        }
        let logical_row_ids: Vec<i32> = (0..self.logical_output_width).collect();
        restored.rows(runtime, &logical_row_ids)
    }

    /// The output rows `row_ids` (output columns of the projection) in MLX's
    /// layout, in the order given. Each row's whole 32-row tile is restored
    /// and the row taken from it.
    pub fn restored_rows(
        &self,
        runtime: &MlxRuntime,
        row_ids: &[i32],
    ) -> Result<RestoredMlxRows, RowBatchedProjectionError> {
        let mut tile_ids = Vec::with_capacity(row_ids.len());
        let mut rows_within_restored = Vec::with_capacity(row_ids.len());
        for (listed_index, row_id) in (0_i32..).zip(row_ids) {
            if !(0..self.logical_output_width).contains(row_id) {
                return Err(RowBatchedProjectionError::UnsupportedQuantization {
                    description: format!(
                        "row {row_id} is outside {} rows",
                        self.logical_output_width
                    ),
                });
            }
            tile_ids.push(row_id / OUTPUT_COLUMNS_PER_SIMDGROUP);
            rows_within_restored.push(
                listed_index * OUTPUT_COLUMNS_PER_SIMDGROUP + row_id % OUTPUT_COLUMNS_PER_SIMDGROUP,
            );
        }
        self.restored_tiles(runtime, &tile_ids)?
            .rows(runtime, &rows_within_restored)
    }

    fn restored_tiles(
        &self,
        runtime: &MlxRuntime,
        tile_ids: &[i32],
    ) -> Result<RestoredMlxRows, RowBatchedProjectionError> {
        // Declared per call: MLX compiles a kernel's source once per process
        // and reuses it, and restoring is rare enough that holding the handle
        // in every model would buy nothing.
        let restoration = MlxMetalKernel::new(
            "astro_register_order_restoration_4bit",
            &["R", "P", "tile_ids"],
            &["Wq", "S", "B"],
            REGISTER_ORDER_RESTORATION_SOURCE,
        )?;
        let listed_tile_count = tile_ids.len() as i32;
        let restored_row_count = listed_tile_count * OUTPUT_COLUMNS_PER_SIMDGROUP;
        let group_count = self.input_width / QUANTIZATION_GROUP_SIZE;
        let tile_id_array = runtime.array_from_i32(tile_ids, &[listed_tile_count])?;
        let mut outputs = runtime.apply_metal_kernel(
            &restoration,
            &[&self.codes, &self.scale_bias_words, &tile_id_array],
            &[
                MlxMetalKernelOutput::new(
                    vec![restored_row_count, self.input_width / 8],
                    MlxDtype::UInt32,
                ),
                MlxMetalKernelOutput::new(
                    vec![restored_row_count, group_count],
                    MlxDtype::BFloat16,
                ),
                MlxMetalKernelOutput::new(
                    vec![restored_row_count, group_count],
                    MlxDtype::BFloat16,
                ),
            ],
            [listed_tile_count * group_count * REGISTER_BLOCK_WORDS, 1, 1],
            [REGISTER_BLOCK_WORDS, 1, 1],
            &[
                integer("N", self.output_width),
                integer("K", self.input_width),
            ],
        )?;
        let biases = outputs
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)?;
        let scales = outputs
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)?;
        let packed_weight = outputs
            .pop()
            .ok_or(RowBatchedProjectionError::MissingOutput)?;
        Ok(RestoredMlxRows {
            packed_weight,
            scales,
            biases,
        })
    }
}

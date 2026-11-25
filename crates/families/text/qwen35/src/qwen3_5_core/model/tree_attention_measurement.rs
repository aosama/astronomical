//! Measurements of a draft tree's full attention, kept apart from the
//! serving path: MLX's attention run in float32 (the exact reference), and a
//! record of how many output elements the row-exact kernel and MLX's bf16
//! attention each leave from it. The acceptance journeys use them to tell a
//! kernel's precision from the spread between sampled texts (`cargo
//! measure-dflash2-row-exactness`, `cargo measure-dflash2-tree-rounds`);
//! serving never turns them on.

use mlx_bridge::MlxArray;

use super::base::Qwen3_5ModelBase;
use super::tree_layer_attention::TreeRowInputs;
use text_generation::execution_error::TextExecutionError;

/// One full-attention layer's tree attention against float32, rounded to bf16.
#[derive(Debug, Clone, Copy, Default)]
pub struct TreeAttentionErrorCount {
    pub element_count: usize,
    /// Elements where the row-exact kernel's bf16 output differs from
    /// float32 attention rounded to bf16.
    pub kernel_differing_elements: usize,
    /// The same for MLX's bf16 attention, the serving path.
    pub mlx_differing_elements: usize,
    pub kernel_largest_difference: f32,
    pub mlx_largest_difference: f32,
    /// The kernel's differing elements by tree row (first four rows).
    pub kernel_differing_by_row: [usize; 4],
    /// The kernel's differing elements in dimensions 0..128 and 128..256,
    /// each half the output of one simdgroup of a pair.
    pub kernel_differing_by_half: [usize; 2],
    /// Of the kernel's differing elements, those smaller in magnitude than
    /// float32's: about half for rounding noise, most for a shrinking bias.
    pub kernel_smaller_elements: usize,
}

impl Qwen3_5ModelBase {
    /// Runs MLX's tree attention in float32 (true) or in the inputs' bf16
    /// (false, as serving does), for measurements of attention precision.
    #[doc(hidden)]
    pub fn set_tree_attention_in_float32_for_measurement(&mut self, in_float32: bool) {
        self.optional_accelerators
            .tree_attention_in_float32_for_measurement = in_float32;
    }

    /// Starts recording, for every tree attention through the row-exact
    /// kernel, how its output and MLX's bf16 output each differ from float32
    /// attention. Measurement only: every layer is evaluated on the spot.
    #[doc(hidden)]
    pub fn start_recording_tree_attention_errors_for_measurement(&mut self) {
        self.optional_accelerators.tree_attention_error_record =
            Some(std::cell::RefCell::new(Vec::new()));
    }

    /// The counts recorded so far, oldest first, leaving the record empty.
    #[doc(hidden)]
    pub fn take_tree_attention_errors_for_measurement(&self) -> Vec<TreeAttentionErrorCount> {
        self.optional_accelerators
            .tree_attention_error_record
            .as_ref()
            .map(|record| record.take())
            .unwrap_or_default()
    }

    /// Counts, against float32 attention rounded to bf16, the elements the
    /// kernel's output and MLX's bf16 output each differ in.
    pub(super) fn record_tree_attention_errors(
        &self,
        kernel_attended: &MlxArray,
        rotated_queries: &MlxArray,
        (active_keys, active_values): (&MlxArray, &MlxArray),
        tree: &TreeRowInputs,
        scale: f32,
    ) -> Result<(), TextExecutionError> {
        let mlx_attended = self.runtime.masked_scaled_dot_product_attention(
            rotated_queries,
            active_keys,
            active_values,
            scale,
            &tree.attention_mask,
        )?;
        let exact_attended = self.tree_attention_in_float32(
            rotated_queries,
            (active_keys, active_values),
            tree,
            scale,
        )?;
        let kernel_values = self.host_float32_values(kernel_attended)?;
        let mlx_values = self.host_float32_values(&mlx_attended)?;
        let exact_values = self.host_float32_values(&exact_attended)?;
        let mut count = TreeAttentionErrorCount {
            element_count: exact_values.len(),
            ..Default::default()
        };
        let row_count = rotated_queries.shape()[2] as usize;
        for (element_index, ((kernel_value, mlx_value), exact_value)) in kernel_values
            .iter()
            .zip(&mlx_values)
            .zip(&exact_values)
            .enumerate()
        {
            let kernel_difference = (kernel_value - exact_value).abs();
            let mlx_difference = (mlx_value - exact_value).abs();
            if kernel_difference > 0.0 {
                let row = (element_index / 256) % row_count;
                if row < 4 {
                    count.kernel_differing_by_row[row] += 1;
                }
                count.kernel_differing_by_half[(element_index % 256) / 128] += 1;
                count.kernel_smaller_elements +=
                    usize::from(kernel_value.abs() < exact_value.abs());
            }
            count.kernel_differing_elements += usize::from(kernel_difference > 0.0);
            count.mlx_differing_elements += usize::from(mlx_difference > 0.0);
            count.kernel_largest_difference =
                count.kernel_largest_difference.max(kernel_difference);
            count.mlx_largest_difference = count.mlx_largest_difference.max(mlx_difference);
        }
        if let Some(record) = self
            .optional_accelerators
            .tree_attention_error_record
            .as_ref()
        {
            record.borrow_mut().push(count);
        }
        Ok(())
    }

    fn host_float32_values(&self, array: &MlxArray) -> Result<Vec<f32>, TextExecutionError> {
        let single_precision = self.runtime.astype(array, mlx_bridge::MlxDtype::Float32)?;
        Ok(self.runtime.array_to_vec_f32(&single_precision)?)
    }

    /// MLX's masked attention on float32 copies of the bf16 inputs, cast
    /// back to the queries' dtype: the exact reference for measurements.
    pub(super) fn tree_attention_in_float32(
        &self,
        rotated_queries: &MlxArray,
        (active_keys, active_values): (&MlxArray, &MlxArray),
        tree: &TreeRowInputs,
        scale: f32,
    ) -> Result<MlxArray, TextExecutionError> {
        let float32 = mlx_bridge::MlxDtype::Float32;
        let attended = self.runtime.masked_scaled_dot_product_attention(
            &self.runtime.astype(rotated_queries, float32)?,
            &self.runtime.astype(active_keys, float32)?,
            &self.runtime.astype(active_values, float32)?,
            scale,
            &self.runtime.astype(&tree.attention_mask, float32)?,
        )?;
        Ok(self.runtime.astype(&attended, rotated_queries.dtype())?)
    }
}

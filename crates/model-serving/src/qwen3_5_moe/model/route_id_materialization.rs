//! Host materialization of routed expert identifiers for paging and demand evidence.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};

impl Qwen3_5Model {
    /// Copies and normalizes one route for compact page planning.
    pub(super) fn copy_sorted_unique_expert_ids(
        &self,
        selected_indices: &MlxArray,
    ) -> Result<Vec<usize>, Qwen3_5ExecutionError> {
        let mut selected_expert_ids = self.copy_selected_expert_ids(selected_indices)?;
        selected_expert_ids.sort_unstable();
        selected_expert_ids.dedup();
        Ok(selected_expert_ids)
    }

    /// Copies every routed assignment so demand learning preserves frequency.
    pub(super) fn copy_selected_expert_ids(
        &self,
        selected_indices: &MlxArray,
    ) -> Result<Vec<usize>, Qwen3_5ExecutionError> {
        let contiguous_ids = self
            .runtime
            .build_contiguous_row_major_copy(selected_indices)?;
        contiguous_ids.evaluate()?;
        Ok(contiguous_ids
            .copy_evaluated_u32_values()?
            .into_iter()
            .map(|expert_id| expert_id as usize)
            .collect())
    }
}

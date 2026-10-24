//! Host materialization of routed expert identifiers for paging and demand evidence.

use mlx_c_rust::MlxArray;

use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{PerformanceAttribution, PerformanceOperation};

impl Qwen3_5Model {
    /// Copies and normalizes one route for compact page planning.
    pub(super) fn copy_sorted_unique_expert_ids(
        &self,
        selected_indices: &MlxArray,
    ) -> Result<Vec<usize>, Qwen3_5ExecutionError> {
        let mut selected_expert_ids = self.copy_selected_expert_ids(selected_indices, None)?;
        selected_expert_ids.sort_unstable();
        selected_expert_ids.dedup();
        Ok(selected_expert_ids)
    }

    /// Copies every routed assignment so demand learning preserves frequency.
    pub(super) fn copy_selected_expert_ids(
        &self,
        selected_indices: &MlxArray,
        mut performance_attribution: Option<&mut PerformanceAttribution>,
    ) -> Result<Vec<usize>, Qwen3_5ExecutionError> {
        let contiguous_ids =
            if let Some(performance_attribution) = performance_attribution.as_deref_mut() {
                performance_attribution.measure_operation(
                    PerformanceOperation::DecodeExpertRouteContiguousArrayConstruction,
                    |_performance_attribution| {
                        self.runtime
                            .build_contiguous_row_major_copy(selected_indices)
                    },
                )?
            } else {
                self.runtime
                    .build_contiguous_row_major_copy(selected_indices)?
            };
        if let Some(performance_attribution) = performance_attribution.as_deref_mut() {
            performance_attribution.measure_operation(
                PerformanceOperation::DecodeExpertRouteArrayEvaluation,
                |_performance_attribution| contiguous_ids.evaluate(),
            )?;
        } else {
            contiguous_ids.evaluate()?;
        }
        let expert_ids =
            if let Some(performance_attribution) = performance_attribution.as_deref_mut() {
                performance_attribution.measure_operation(
                    PerformanceOperation::DecodeExpertRouteHostCopy,
                    |_performance_attribution| contiguous_ids.copy_evaluated_u32_values(),
                )?
            } else {
                contiguous_ids.copy_evaluated_u32_values()?
            };
        Ok(expert_ids
            .into_iter()
            .map(|expert_id| expert_id as usize)
            .collect())
    }
}

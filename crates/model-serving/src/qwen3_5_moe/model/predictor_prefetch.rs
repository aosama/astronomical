//! Predictor-driven leftover prefetch (issue #540).
//!
//! After sampling the next token, the predictor names experts that token is
//! likely to route. Missing names are streamed into leftover warm slots only.
//! Occupied pages are never evicted. Fail-open: a stream error skips the layer.

use crate::qwen3_5::model::Qwen3_5Model;
use crate::{PerformanceAttribution, PerformanceCounter};

impl Qwen3_5Model {
    pub(crate) fn prefetch_predicted_experts(
        &self,
        predicted_experts_by_layer: &[Vec<usize>],
        performance_attribution: &mut PerformanceAttribution,
    ) {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return;
        };
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return;
        };
        let mut prefetched_expert_count = 0_u64;
        let mut prefetched_payload_bytes = 0_u64;
        for (layer_index, predicted_expert_ids) in predicted_experts_by_layer.iter().enumerate() {
            if predicted_expert_ids.is_empty() {
                continue;
            }
            let missing_expert_ids: Vec<usize> = {
                let retained_experts = retained_experts.borrow();
                predicted_expert_ids
                    .iter()
                    .copied()
                    .filter(|expert_id| !retained_experts.is_expert_warm(layer_index, *expert_id))
                    .collect()
            };
            if missing_expert_ids.is_empty() {
                continue;
            }
            let Ok((streamed_weights, streamed_manifest)) = self
                .stream_operation_local_routed_experts(
                    expert_pager,
                    layer_index,
                    1,
                    &missing_expert_ids,
                    true,
                    performance_attribution,
                )
            else {
                continue;
            };
            let written_expert_count = retained_experts.borrow_mut().insert_into_free_slots_only(
                &self.runtime,
                layer_index,
                &missing_expert_ids,
                &streamed_weights,
                missing_expert_ids.len(),
            );
            let Ok(written_expert_count) = written_expert_count else {
                continue;
            };
            if written_expert_count == 0 {
                continue;
            }
            prefetched_expert_count =
                prefetched_expert_count.saturating_add(written_expert_count as u64);
            let per_expert_bytes = streamed_manifest
                .payload_byte_count
                .checked_div(missing_expert_ids.len() as u64)
                .unwrap_or(0);
            prefetched_payload_bytes = prefetched_payload_bytes
                .saturating_add(per_expert_bytes.saturating_mul(written_expert_count as u64));
        }
        if prefetched_expert_count > 0 {
            performance_attribution.record_counter(
                PerformanceCounter::PredictorPrefetchIssueCount,
                prefetched_expert_count,
            );
            performance_attribution.record_counter(
                PerformanceCounter::PredictorPrefetchByteCount,
                prefetched_payload_bytes,
            );
        }
    }
}

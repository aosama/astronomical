//! Enacts `memory/`'s decode-handoff seating decision.
//!
//! The planner may name complete layers after an atomic demote. Decode never
//! streams complete layers, so this pass loads those indexes into retained RAM
//! before the first generate token. SSD reads stay in the pager; this file only
//! walks the decided indexes.

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{PerformanceAttribution, complete_layer_indexes_required_before_decode};

impl Qwen3_5Model {
    /// Complete-layer indexes `memory/` requires seated before decode.
    pub(crate) fn planned_complete_layer_indexes_to_seat_before_decode(&self) -> Vec<usize> {
        self.active_expert_residency_plan
            .borrow()
            .as_ref()
            .map(complete_layer_indexes_required_before_decode)
            .unwrap_or_default()
    }

    fn has_seated_complete_layer(&self, layer_index: usize, expert_capacity: usize) -> bool {
        self.retained_experts
            .as_ref()
            .is_some_and(|retained_experts| {
                retained_experts
                    .borrow()
                    .has_complete_layer(layer_index, expert_capacity)
            })
    }

    /// Loads planned complete layers into the retained cache until leftover fills.
    pub(crate) fn seat_complete_layers_before_decode(
        &self,
        layer_indexes: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(0);
        };
        let mut seated_payload_bytes = 0_u64;
        for &layer_index in layer_indexes {
            let Some(layer_plan) = expert_pager.layer_plans().get(layer_index) else {
                continue;
            };
            let expert_capacity = layer_plan.expert_capacity;
            if self.has_seated_complete_layer(layer_index, expert_capacity) {
                continue;
            }
            let (_streamed_weights, streamed_manifest) = self.stream_complete_expert_layer(
                expert_pager,
                layer_index,
                2_048,
                expert_capacity,
                expert_capacity,
                true,
                performance_attribution,
            )?;
            if self.has_seated_complete_layer(layer_index, expert_capacity) {
                seated_payload_bytes =
                    seated_payload_bytes.saturating_add(streamed_manifest.payload_byte_count);
            } else {
                break;
            }
        }
        Ok(seated_payload_bytes)
    }
}

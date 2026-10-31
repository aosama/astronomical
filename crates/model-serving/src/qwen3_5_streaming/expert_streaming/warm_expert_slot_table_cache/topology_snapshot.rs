//! Residency topology snapshots: the occupancy truth the residency planner
//! arbitrates, with each seated layer's covered demand across both phases.
//!
//! A layer's coverage score sums prefill and decode evidence on purpose:
//! prefill demand measures what the current request needs admitted now,
//! decode demand measures what the conversation keeps re-using, and a layer
//! seating prompt-needed and conversation-hot experts covers strictly more
//! real demand than either stream measures alone. Summing also preserves
//! the planner's historical behavior, where a single fused counter played
//! both roles.

use crate::memory::{CurrentExpertLayerResidency, RetainedExpertPageClass};

use super::{ExpertSlotTable, WarmExpertSlotTableCache};

impl WarmExpertSlotTableCache {
    #[must_use]
    pub fn topology_snapshot(&self, expert_capacity: usize) -> Vec<CurrentExpertLayerResidency> {
        self.tables_by_layer
            .iter()
            .enumerate()
            .filter_map(|(layer_index, table)| {
                self.current_layer_residency(layer_index, table.as_ref(), expert_capacity)
            })
            .collect()
    }

    /// One layer's current residency classification, or `None` when the
    /// layer holds no warm table.
    fn current_layer_residency(
        &self,
        layer_index: usize,
        warm_table: Option<&ExpertSlotTable>,
        expert_capacity: usize,
    ) -> Option<CurrentExpertLayerResidency> {
        let warm_table = warm_table?;
        let mut expert_ids: Vec<usize> = warm_table.slot_by_expert_id.keys().copied().collect();
        expert_ids.sort_unstable();
        // A slot table that has grown to hold every expert is effectively
        // a complete layer and must be classified as StableCompleteLayer
        // so the residency validation passes (ElasticRoutedExperts requires
        // retained_count < expert_capacity, which is false when all are held).
        let class = if expert_ids.len() >= expert_capacity {
            RetainedExpertPageClass::StableCompleteLayer
        } else {
            RetainedExpertPageClass::ElasticRoutedExperts
        };
        let covered_weighted_demand = self.covered_demand(layer_index, &expert_ids);
        Some(CurrentExpertLayerResidency {
            layer_index,
            class,
            retained_expert_ids: expert_ids,
            payload_bytes: warm_table.payload_bytes,
            covered_weighted_demand,
        })
    }

    /// Pure: the prefill-plus-decode demand a layer's seated experts cover.
    /// Missing phases or unsized layers contribute zero for an expert.
    fn covered_demand(&self, layer_index: usize, seated_expert_ids: &[usize]) -> u64 {
        let prefill_layer_counts = self.prefill_demand_counts_by_layer.get(layer_index);
        let decode_layer_counts = self.decode_demand_counts_by_layer.get(layer_index);
        let mut covered_weighted_demand = 0_u64;
        for expert_id in seated_expert_ids {
            let prefill_demand_count = prefill_layer_counts
                .and_then(|layer_counts| layer_counts.get(*expert_id))
                .copied()
                .unwrap_or(0);
            let decode_demand_count = decode_layer_counts
                .and_then(|layer_counts| layer_counts.get(*expert_id))
                .copied()
                .unwrap_or(0);
            covered_weighted_demand = covered_weighted_demand
                .saturating_add(prefill_demand_count)
                .saturating_add(decode_demand_count);
        }
        covered_weighted_demand
    }
}

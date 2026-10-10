//! Route-demand evidence and topology reporting for the retained expert
//! slot-table cache.
//!
//! Demand counters feed the residency planner's coverage scoring, and the
//! topology snapshot is the single source the planner reads current ownership
//! from.

use crate::memory::{CurrentExpertLayerResidency, RetainedExpertPageClass};

use super::RetainedExpertCache;

impl RetainedExpertCache {
    pub fn record_expert_demand(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_expert_ids: &[usize],
    ) {
        let Some(demand) = self.expert_demand_counts_by_layer.get_mut(layer_index) else {
            return;
        };
        if demand.len() < expert_capacity {
            demand.resize(expert_capacity, 0);
        }
        let weight = self.demand_assignment_weight.max(1);
        for expert_id in selected_expert_ids {
            if let Some(count) = demand.get_mut(*expert_id) {
                *count = count.saturating_add(weight);
            }
        }
    }

    pub fn clear_expert_demand(&mut self) {
        for demand in &mut self.expert_demand_counts_by_layer {
            demand.fill(0);
        }
        self.demand_assignment_weight = 1;
    }

    pub fn record_disk_load(&mut self, expert_count: usize, batch_count: usize) {
        self.disk_page_load_count = self
            .disk_page_load_count
            .saturating_add(u64::try_from(expert_count).unwrap_or(u64::MAX));
        self.disk_batch_load_count = self
            .disk_batch_load_count
            .saturating_add(u64::try_from(batch_count).unwrap_or(u64::MAX));
    }

    #[must_use]
    pub fn topology_snapshot(&self, expert_capacity: usize) -> Vec<CurrentExpertLayerResidency> {
        self.tables_by_layer
            .iter()
            .enumerate()
            .filter_map(|(layer_index, table)| {
                let table = table.as_ref()?;
                let mut expert_ids: Vec<usize> = table.slot_by_expert_id.keys().copied().collect();
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
                let covered_weighted_demand = expert_ids
                    .iter()
                    .filter_map(|expert_id| {
                        self.expert_demand_counts_by_layer[layer_index].get(*expert_id)
                    })
                    .copied()
                    .fold(0_u64, u64::saturating_add);
                Some(CurrentExpertLayerResidency {
                    layer_index,
                    class,
                    retained_expert_ids: expert_ids,
                    payload_bytes: table.payload_bytes,
                    covered_weighted_demand,
                })
            })
            .collect()
    }
}

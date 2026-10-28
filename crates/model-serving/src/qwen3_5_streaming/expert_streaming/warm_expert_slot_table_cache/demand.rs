//! Route-demand evidence and topology reporting for the retained expert
//! slot-table cache.
//!
//! Demand counters feed the residency planner's coverage scoring, and the
//! topology snapshot is the single source the planner reads current ownership
//! from.

use crate::memory::{CurrentExpertLayerResidency, RetainedExpertPageClass};

use super::{ExpertSlotTable, WarmExpertSlotTableCache};

impl WarmExpertSlotTableCache {
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

    /// Raises or restores the per-assignment demand multiplier.
    ///
    /// The request's final prefill chunk records under
    /// `last_prefill_chunk_demand_weight` so prompt-tail routes keep decode
    /// relevant coverage in the handoff residency plan.
    pub fn set_demand_assignment_weight(&mut self, assignment_weight: u64) {
        self.demand_assignment_weight = assignment_weight.max(1);
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
        let covered_weighted_demand = expert_ids
            .iter()
            .filter_map(|expert_id| self.expert_demand_counts_by_layer[layer_index].get(*expert_id))
            .copied()
            .fold(0_u64, u64::saturating_add);
        Some(CurrentExpertLayerResidency {
            layer_index,
            class,
            retained_expert_ids: expert_ids,
            payload_bytes: warm_table.payload_bytes,
            covered_weighted_demand,
        })
    }
}

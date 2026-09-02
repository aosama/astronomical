//! Reclamation and ceiling enforcement for the retained expert slot-table
//! cache: request-pressure freezes, whole-table eviction order, and the
//! effective-ceiling arithmetic the insert path and the residency planner
//! share.

use crate::expert_paging::{ExpertWeightMemoryCacheStatistics, RetainedExpertReclamation};

use super::RetainedExpertCache;

impl RetainedExpertCache {
    pub fn limit_for_request_pressure(&mut self, reclamation_target_bytes: u64) -> bool {
        let pressure_maximum = self
            .resident_payload_bytes
            .saturating_sub(reclamation_target_bytes);
        self.limit_for_request_pressure_to_maximum(pressure_maximum)
    }

    pub fn limit_for_request_pressure_to_maximum(
        &mut self,
        pressure_maximum_resident_payload_bytes: u64,
    ) -> bool {
        self.request_pressure_maximum_resident_payload_bytes =
            Some(pressure_maximum_resident_payload_bytes);
        self.reclaim_to_effective_ceiling().released_payload_bytes() > 0
    }

    pub fn resume_after_request_pressure(&mut self) -> bool {
        self.request_pressure_maximum_resident_payload_bytes
            .take()
            .is_some()
    }

    pub fn release_all(&mut self) -> bool {
        let had_experts = self.resident_payload_bytes > 0;
        for layer_index in 0..self.tables_by_layer.len() {
            self.remove_layer(layer_index);
        }
        had_experts
    }

    #[must_use]
    pub fn statistics(&self) -> ExpertWeightMemoryCacheStatistics {
        let occupied_layer_count = self
            .tables_by_layer
            .iter()
            .filter(|table| table.is_some())
            .count();
        let total_expert_count = self
            .tables_by_layer
            .iter()
            .filter_map(|table| table.as_ref())
            .map(|table| table.occupied_slot_count)
            .sum::<usize>();
        ExpertWeightMemoryCacheStatistics {
            entry_count: total_expert_count,
            resident_payload_byte_count: self.resident_payload_bytes,
            maximum_resident_payload_byte_count: self.effective_maximum_resident_payload_bytes(),
            eviction_count: self.eviction_count,
            disk_page_load_count: self.disk_page_load_count,
            disk_batch_load_count: self.disk_batch_load_count,
            complete_layer_count: 0,
            complete_layer_payload_byte_count: 0,
            partial_layer_count: occupied_layer_count,
            partial_layer_payload_byte_count: self.resident_payload_bytes,
            mandatory_read_promotion_count: 0,
            complete_layer_eviction_count: 0,
            partial_layer_eviction_count: self.eviction_count,
        }
    }

    /// Returns the plan-composed long-lived ceiling (without any request-
    /// pressure override). Hot-expert warming must respect this too: the
    /// composer already subtracted activation workspace, context reserve, and
    /// the stream slot from the active ceiling when composing it.
    #[must_use]
    pub(crate) fn normal_maximum_resident_payload_bytes(&self) -> u64 {
        self.normal_maximum_resident_payload_bytes
    }

    pub(super) fn can_admit(&self, new_payload_bytes: u64) -> bool {
        self.resident_payload_bytes
            .saturating_add(new_payload_bytes)
            <= self.effective_maximum_resident_payload_bytes()
    }

    fn evict_least_used_table(&mut self) -> bool {
        let mut fewest: Option<(usize, usize)> = None;
        for (layer_index, table) in self.tables_by_layer.iter().enumerate() {
            if let Some(table) = table.as_ref() {
                let is_smaller = fewest
                    .is_none_or(|(_, smallest_count)| table.occupied_slot_count < smallest_count);
                if is_smaller {
                    fewest = Some((layer_index, table.occupied_slot_count));
                }
            }
        }
        let Some((layer_index, _)) = fewest else {
            return false;
        };
        self.remove_layer(layer_index)
    }

    pub(super) fn reclaim_to_effective_ceiling(&mut self) -> RetainedExpertReclamation {
        let mut released_bytes = 0_u64;
        let mut released_count = 0_usize;
        tracing::debug!(
            resident_payload_bytes = self.resident_payload_bytes,
            effective_maximum = self.effective_maximum_resident_payload_bytes(),
            table_count = self.tables_by_layer.iter().filter(|t| t.is_some()).count(),
            "reclaim_to_effective_ceiling starting"
        );
        while self.resident_payload_bytes > self.effective_maximum_resident_payload_bytes() {
            let before = self.resident_payload_bytes;
            if !self.evict_least_used_table() {
                break;
            }
            released_bytes =
                released_bytes.saturating_add(before.saturating_sub(self.resident_payload_bytes));
            released_count += 1;
        }
        if released_count > 0 {
            tracing::info!(
                released_count,
                released_bytes,
                resident_payload_bytes = self.resident_payload_bytes,
                table_count = self.tables_by_layer.iter().filter(|t| t.is_some()).count(),
                "reclaim_to_effective_ceiling evicted tables"
            );
        }
        RetainedExpertReclamation {
            released_partial_layer_count: released_count,
            released_partial_payload_bytes: released_bytes,
            released_complete_layer_count: 0,
            released_complete_payload_bytes: 0,
        }
    }

    pub(super) fn effective_maximum_resident_payload_bytes(&self) -> u64 {
        self.request_pressure_maximum_resident_payload_bytes
            .unwrap_or(self.normal_maximum_resident_payload_bytes)
    }
}

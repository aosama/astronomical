//! Capacity enforcement for the retained experts of the astro expert cache: the
//! whole-table eviction order and the admission check the insert path uses.

use mlx_bridge::MlxArray;

use engine_core::expert_paging::{AstroExpertCacheStatistics, RetainedExpertReclamation};

use super::AstroExpertCache;

impl AstroExpertCache {
    #[must_use]
    pub fn statistics(&self) -> AstroExpertCacheStatistics {
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
        AstroExpertCacheStatistics {
            entry_count: total_expert_count,
            resident_payload_byte_count: self.resident_payload_bytes,
            maximum_resident_payload_byte_count: self.capacity_bytes,
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

    /// How many bytes of experts the astro expert cache may keep.
    #[must_use]
    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub fn append_resident_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        for table in self.tables_by_layer.iter().flatten() {
            table.weights.append_array_references(arrays);
        }
    }

    pub(super) fn can_admit(&self, new_payload_bytes: u64) -> bool {
        self.resident_payload_bytes
            .saturating_add(new_payload_bytes)
            <= self.capacity_bytes
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
        let Some((layer_index, occupied_slot_count)) = fewest else {
            return false;
        };
        let victim_payload_bytes = self.tables_by_layer[layer_index]
            .as_ref()
            .map_or(0, |table| table.payload_bytes);
        let victim_is_complete = self.tables_by_layer[layer_index]
            .as_ref()
            .is_some_and(|table| {
                table.occupied_slot_count == table.slot_count() && table.slot_count() > 0
            });
        tracing::info!(
            layer_index,
            occupied_slot_count,
            victim_payload_bytes,
            victim_is_complete,
            "warm astro_expert_cache evicted least-used table"
        );
        self.remove_layer(layer_index)
    }

    pub(super) fn trim_to_capacity(&mut self) -> RetainedExpertReclamation {
        let mut released_bytes = 0_u64;
        let mut released_count = 0_usize;
        tracing::debug!(
            resident_payload_bytes = self.resident_payload_bytes,
            capacity_bytes = self.capacity_bytes,
            table_count = self.tables_by_layer.iter().filter(|t| t.is_some()).count(),
            "trim_to_capacity starting"
        );
        while self.resident_payload_bytes > self.capacity_bytes {
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
                "trim_to_capacity evicted tables"
            );
        }
        RetainedExpertReclamation {
            released_partial_layer_count: released_count,
            released_partial_payload_bytes: released_bytes,
            released_complete_layer_count: 0,
            released_complete_payload_bytes: 0,
        }
    }
}

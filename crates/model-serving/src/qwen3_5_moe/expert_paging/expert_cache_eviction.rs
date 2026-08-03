use super::expert_cache::ExpertWeightMemoryCache;

impl ExpertWeightMemoryCache {
    pub(super) fn evict_oldest_unprotected_partial_experts_to_fit_global_maximum(
        &mut self,
        protected_layer_index: Option<usize>,
        protected_selected_expert_ids: &[usize],
    ) {
        while self.resident_payload_byte_count > self.maximum_resident_payload_byte_count {
            let partial_expert_eviction_candidate = self
                .cached_experts_by_layer
                .iter()
                .enumerate()
                .flat_map(|(layer_index, cached_experts_for_layer)| {
                    cached_experts_for_layer
                        .iter()
                        .filter_map(move |(expert_id, cached_expert)| {
                            let is_protected_in_flight_expert =
                                protected_layer_index.is_some_and(|protected_layer_index| {
                                    layer_index == protected_layer_index
                                        && protected_selected_expert_ids.contains(expert_id)
                                });
                            (!is_protected_in_flight_expert).then_some((
                                layer_index,
                                *expert_id,
                                cached_expert.last_access_sequence_number,
                            ))
                        })
                })
                .min_by_key(
                    |(
                        partial_expert_layer_index,
                        partial_expert_id,
                        partial_expert_last_access_sequence_number,
                    )| {
                        (
                            *partial_expert_last_access_sequence_number,
                            *partial_expert_layer_index,
                            *partial_expert_id,
                        )
                    },
                )
                .map(|(partial_expert_layer_index, partial_expert_id, _)| {
                    (partial_expert_layer_index, partial_expert_id)
                });
            let Some((partial_expert_layer_index, partial_expert_id)) =
                partial_expert_eviction_candidate
            else {
                break;
            };
            let Some(evicted_partial_expert) =
                self.cached_experts_by_layer[partial_expert_layer_index].remove(&partial_expert_id)
            else {
                break;
            };
            self.resident_payload_byte_count = self
                .resident_payload_byte_count
                .saturating_sub(evicted_partial_expert.resident_payload_byte_count);
            self.resident_payload_byte_count_by_layer[partial_expert_layer_index] = self
                .resident_payload_byte_count_by_layer[partial_expert_layer_index]
                .saturating_sub(evicted_partial_expert.resident_payload_byte_count);
            self.eviction_count = self.eviction_count.saturating_add(1);
        }
    }
}

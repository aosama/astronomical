//! In-memory cache for complete-layer and one-expert MLX pages. Complete layers
//! preserve global-expert order; partial layers own independent expert pages.

use std::collections::HashMap;

use super::ExpertWeightPage;
use super::expert_cache_statistics::{
    ExpertWeightMemoryCacheStatistics, paged_expert_payload_byte_count,
};
use super::memory_budget::{
    MemoryBudgetSnapshot, automatic_expert_weight_memory_cache_maximum_size_bytes,
};

/// Complete one-expert weights retained for reuse by later decode tokens.
#[derive(Debug)]
pub(crate) struct CachedExpertWeights<ExpertPage> {
    pub(crate) paged_expert_weights: ExpertPage,
    pub(super) resident_payload_byte_count: u64,
    pub(super) last_access_sequence_number: u64,
}

/// One complete layer page retained in its original global-expert order.
#[derive(Debug)]
pub(super) struct CachedCompleteLayerExpertWeights<ExpertPage> {
    pub(super) paged_expert_weights: ExpertPage,
    pub(super) resident_payload_byte_count: u64,
    pub(super) last_access_sequence_number: u64,
}

/// Budgets complete layers and model-derived decode-route floors within one ceiling.
#[derive(Debug)]
pub struct ExpertWeightMemoryCache<ExpertPage> {
    pub(super) cached_experts_by_layer: Vec<HashMap<usize, CachedExpertWeights<ExpertPage>>>,
    pub(super) complete_layer_expert_weights:
        Vec<Option<CachedCompleteLayerExpertWeights<ExpertPage>>>,
    pub(super) minimum_decode_route_payload_byte_count_by_layer: Vec<u64>,
    pub(super) resident_payload_byte_count_by_layer: Vec<u64>,
    pub(super) resident_payload_byte_count: u64,
    pub(super) maximum_resident_payload_byte_count: u64,
    pub(super) request_memory_pressure_maximum_resident_payload_byte_count: Option<u64>,
    next_access_sequence_number: u64,
    pub(super) eviction_count: u64,
    cache_hit_count: u64,
    complete_layer_hit_count: u64,
    cache_miss_count: u64,
    disk_page_load_count: u64,
    disk_batch_load_count: u64,
}

impl<ExpertPage> ExpertWeightMemoryCache<ExpertPage>
where
    ExpertPage: ExpertWeightPage,
{
    /// Returns whether every decoder layer retains one complete sparse-expert page.
    #[must_use]
    pub fn has_complete_expert_layers_for_every_decoder_layer(&self) -> bool {
        !self.complete_layer_expert_weights.is_empty()
            && self
                .complete_layer_expert_weights
                .iter()
                .all(Option::is_some)
    }

    pub(crate) fn has_complete_expert_layer(&self, layer_index: usize) -> bool {
        self.complete_layer_expert_weights
            .get(layer_index)
            .is_some_and(Option::is_some)
    }
    #[must_use]
    pub fn new(
        layer_count: usize,
        mut minimum_decode_route_payload_byte_count_by_layer: Vec<u64>,
    ) -> Self {
        minimum_decode_route_payload_byte_count_by_layer.resize(layer_count, 0);
        minimum_decode_route_payload_byte_count_by_layer.truncate(layer_count);
        Self {
            cached_experts_by_layer: (0..layer_count).map(|_| HashMap::new()).collect(),
            complete_layer_expert_weights: (0..layer_count).map(|_| None).collect(),
            minimum_decode_route_payload_byte_count_by_layer,
            resident_payload_byte_count_by_layer: vec![0; layer_count],
            resident_payload_byte_count: 0,
            // The first automatic miss replaces this provisional sentinel.
            maximum_resident_payload_byte_count: u64::MAX,
            request_memory_pressure_maximum_resident_payload_byte_count: None,
            next_access_sequence_number: 0,
            eviction_count: 0,
            cache_hit_count: 0,
            complete_layer_hit_count: 0,
            cache_miss_count: 0,
            disk_page_load_count: 0,
            disk_batch_load_count: 0,
        }
    }

    pub(crate) fn record_expert_access(&mut self, layer_index: usize, expert_id: usize) -> bool {
        // Sequence ordering provides deterministic LRU without clock reads.
        self.next_access_sequence_number = self.next_access_sequence_number.saturating_add(1);
        let Some(cached_expert) = self
            .cached_experts_by_layer
            .get_mut(layer_index)
            .and_then(|cached_experts| cached_experts.get_mut(&expert_id))
        else {
            self.cache_miss_count = self.cache_miss_count.saturating_add(1);
            return false;
        };
        cached_expert.last_access_sequence_number = self.next_access_sequence_number;
        self.cache_hit_count = self.cache_hit_count.saturating_add(1);
        true
    }

    pub(crate) fn record_complete_layer_hit(&mut self, layer_index: usize) -> Option<&ExpertPage> {
        let cached_complete_layer = self
            .complete_layer_expert_weights
            .get_mut(layer_index)
            .and_then(Option::as_mut)?;
        self.next_access_sequence_number = self.next_access_sequence_number.saturating_add(1);
        self.complete_layer_hit_count = self.complete_layer_hit_count.saturating_add(1);
        cached_complete_layer.last_access_sequence_number = self.next_access_sequence_number;
        Some(&cached_complete_layer.paged_expert_weights)
    }

    /// Applies a live retention ceiling to globally held complete layers and
    /// equal layer shares for partial one-expert pages.
    pub fn update_maximum_resident_payload_byte_count(
        &mut self,
        maximum_resident_payload_byte_count: u64,
    ) {
        self.maximum_resident_payload_byte_count = self
            .maximum_resident_payload_byte_count_under_pressure_limits(
                maximum_resident_payload_byte_count,
            );
        self.evict_oldest_unprotected_partial_experts_to_fit_global_maximum(None, &[]);
        self.evict_oldest_complete_layers_to_fit_global_maximum();
    }

    pub(crate) fn maximum_resident_payload_byte_count_for_memory_budget_snapshot(
        &self,
        memory_budget_snapshot: &MemoryBudgetSnapshot,
    ) -> u64 {
        automatic_expert_weight_memory_cache_maximum_size_bytes(
            memory_budget_snapshot,
            self.resident_payload_byte_count,
            0,
        )
    }

    pub(crate) fn update_from_memory_budget_snapshot_while_protecting_selected_experts(
        &mut self,
        memory_budget_snapshot: &MemoryBudgetSnapshot,
        protected_layer_index: usize,
        protected_selected_expert_ids: &[usize],
        pending_retained_expert_payload_bytes: u64,
    ) {
        let maximum_resident_payload_byte_count =
            automatic_expert_weight_memory_cache_maximum_size_bytes(
                memory_budget_snapshot,
                self.resident_payload_byte_count,
                pending_retained_expert_payload_bytes,
            );
        self.maximum_resident_payload_byte_count = self
            .maximum_resident_payload_byte_count_under_pressure_limits(
                maximum_resident_payload_byte_count,
            );
        // Protect the in-flight selection from immediate reload while reclaiming
        // finer-grained partial pages before a complete layer.
        self.evict_oldest_unprotected_partial_experts_to_fit_global_maximum(
            Some(protected_layer_index),
            protected_selected_expert_ids,
        );
        self.evict_oldest_complete_layers_to_fit_global_maximum();
    }

    /// Reconciles retained experts before a request allocates a temporary page.
    pub fn reconcile_retention_before_temporary_expert_page(
        &mut self,
        memory_budget_snapshot: &MemoryBudgetSnapshot,
        protected_layer_index: usize,
        protected_selected_expert_ids: &[usize],
    ) {
        self.update_from_memory_budget_snapshot_while_protecting_selected_experts(
            memory_budget_snapshot,
            protected_layer_index,
            protected_selected_expert_ids,
            0,
        );
    }
    pub(crate) fn cached_expert(
        &self,
        layer_index: usize,
        expert_id: usize,
    ) -> Option<&CachedExpertWeights<ExpertPage>> {
        self.cached_experts_by_layer
            .get(layer_index)
            .and_then(|cached_experts| cached_experts.get(&expert_id))
    }

    pub(crate) fn evict_oldest_unselected_experts_to_fit(
        &mut self,
        layer_index: usize,
        selected_expert_ids: &[usize],
        additional_resident_payload_byte_count: u64,
    ) -> bool {
        if !selected_expert_ids.is_empty() || additional_resident_payload_byte_count > 0 {
            self.evict_complete_layer(layer_index);
        }
        let maximum_layer_resident_payload_byte_count =
            self.maximum_layer_resident_payload_byte_count(layer_index);
        if additional_resident_payload_byte_count > maximum_layer_resident_payload_byte_count {
            return false;
        }
        let Some(layer_resident_payload_byte_count_including_complete_layer) = self
            .resident_payload_byte_count_by_layer
            .get(layer_index)
            .copied()
        else {
            return false;
        };
        let complete_layer_resident_payload_byte_count = self.complete_layer_expert_weights
            [layer_index]
            .as_ref()
            .map_or(0, |complete_layer| {
                complete_layer.resident_payload_byte_count
            });
        let mut projected_layer_resident_payload_byte_count =
            layer_resident_payload_byte_count_including_complete_layer
                .saturating_sub(complete_layer_resident_payload_byte_count)
                .saturating_add(additional_resident_payload_byte_count);

        // Remove oldest unselected experts until the incoming page fits.
        while projected_layer_resident_payload_byte_count
            > maximum_layer_resident_payload_byte_count
        {
            let eviction_expert_id = self.cached_experts_by_layer[layer_index]
                .iter()
                .filter(|(expert_id, _)| !selected_expert_ids.contains(expert_id))
                .min_by_key(|(expert_id, cached_expert)| {
                    (cached_expert.last_access_sequence_number, **expert_id)
                })
                .map(|(expert_id, _)| *expert_id);
            let Some(eviction_expert_id) = eviction_expert_id else {
                return false;
            };
            let Some(evicted_expert) =
                self.cached_experts_by_layer[layer_index].remove(&eviction_expert_id)
            else {
                return false;
            };
            self.resident_payload_byte_count = self
                .resident_payload_byte_count
                .saturating_sub(evicted_expert.resident_payload_byte_count);
            self.resident_payload_byte_count_by_layer[layer_index] = self
                .resident_payload_byte_count_by_layer[layer_index]
                .saturating_sub(evicted_expert.resident_payload_byte_count);
            projected_layer_resident_payload_byte_count =
                projected_layer_resident_payload_byte_count
                    .saturating_sub(evicted_expert.resident_payload_byte_count);
            self.eviction_count = self.eviction_count.saturating_add(1);
        }
        true
    }

    pub(crate) fn remember_expert(
        &mut self,
        layer_index: usize,
        expert_id: usize,
        paged_expert_weights: ExpertPage,
    ) {
        let resident_payload_byte_count = paged_expert_payload_byte_count(&paged_expert_weights);
        // Insertion counts as access so the next turnover keeps this expert.
        self.next_access_sequence_number = self.next_access_sequence_number.saturating_add(1);
        let Some(cached_experts_for_layer) = self.cached_experts_by_layer.get_mut(layer_index)
        else {
            return;
        };
        let replaced_expert_weights = cached_experts_for_layer.insert(
            expert_id,
            CachedExpertWeights {
                paged_expert_weights,
                resident_payload_byte_count,
                last_access_sequence_number: self.next_access_sequence_number,
            },
        );
        if let Some(replaced_expert_weights) = replaced_expert_weights {
            self.resident_payload_byte_count = self
                .resident_payload_byte_count
                .saturating_sub(replaced_expert_weights.resident_payload_byte_count);
            self.resident_payload_byte_count_by_layer[layer_index] = self
                .resident_payload_byte_count_by_layer[layer_index]
                .saturating_sub(replaced_expert_weights.resident_payload_byte_count);
        }
        self.resident_payload_byte_count = self
            .resident_payload_byte_count
            .saturating_add(resident_payload_byte_count);
        self.resident_payload_byte_count_by_layer[layer_index] = self
            .resident_payload_byte_count_by_layer[layer_index]
            .saturating_add(resident_payload_byte_count);
    }

    pub(crate) fn remember_complete_layer_expert_weights(
        &mut self,
        layer_index: usize,
        paged_expert_weights: ExpertPage,
    ) {
        let resident_payload_byte_count = paged_expert_payload_byte_count(&paged_expert_weights);
        self.next_access_sequence_number = self.next_access_sequence_number.saturating_add(1);
        let Some(cached_experts_for_layer) = self.cached_experts_by_layer.get_mut(layer_index)
        else {
            return;
        };
        for (_, removed_expert) in cached_experts_for_layer.drain() {
            self.resident_payload_byte_count = self
                .resident_payload_byte_count
                .saturating_sub(removed_expert.resident_payload_byte_count);
            self.resident_payload_byte_count_by_layer[layer_index] = self
                .resident_payload_byte_count_by_layer[layer_index]
                .saturating_sub(removed_expert.resident_payload_byte_count);
        }
        if let Some(replaced_complete_layer) =
            self.complete_layer_expert_weights[layer_index].take()
        {
            self.resident_payload_byte_count = self
                .resident_payload_byte_count
                .saturating_sub(replaced_complete_layer.resident_payload_byte_count);
            self.resident_payload_byte_count_by_layer[layer_index] = self
                .resident_payload_byte_count_by_layer[layer_index]
                .saturating_sub(replaced_complete_layer.resident_payload_byte_count);
        }
        self.complete_layer_expert_weights[layer_index] = Some(CachedCompleteLayerExpertWeights {
            paged_expert_weights,
            resident_payload_byte_count,
            last_access_sequence_number: self.next_access_sequence_number,
        });
        self.resident_payload_byte_count = self
            .resident_payload_byte_count
            .saturating_add(resident_payload_byte_count);
        self.resident_payload_byte_count_by_layer[layer_index] = self
            .resident_payload_byte_count_by_layer[layer_index]
            .saturating_add(resident_payload_byte_count);
    }

    pub(crate) fn record_disk_page_load(&mut self) {
        self.disk_page_load_count += 1;
    }

    pub(crate) fn record_cache_bypass_misses(&mut self, cache_miss_count: usize) {
        self.cache_miss_count = self
            .cache_miss_count
            .saturating_add(cache_miss_count as u64);
    }

    pub(crate) fn record_disk_page_loads(&mut self, disk_page_load_count: usize) {
        self.disk_page_load_count = self
            .disk_page_load_count
            .saturating_add(disk_page_load_count as u64);
    }

    pub(crate) fn record_disk_batch_loads(&mut self, disk_batch_load_count: usize) {
        self.disk_batch_load_count += disk_batch_load_count as u64;
    }

    #[must_use]
    pub fn statistics(&self) -> ExpertWeightMemoryCacheStatistics {
        let complete_layer_count = self
            .complete_layer_expert_weights
            .iter()
            .filter(|complete_layer| complete_layer.is_some())
            .count();
        ExpertWeightMemoryCacheStatistics {
            entry_count: self
                .cached_experts_by_layer
                .iter()
                .map(HashMap::len)
                .sum::<usize>()
                .saturating_add(complete_layer_count),
            complete_layer_count,
            resident_payload_byte_count: self.resident_payload_byte_count,
            maximum_resident_payload_byte_count: self.maximum_resident_payload_byte_count,
            eviction_count: self.eviction_count,
            cache_hit_count: self.cache_hit_count,
            complete_layer_hit_count: self.complete_layer_hit_count,
            cache_miss_count: self.cache_miss_count,
            disk_page_load_count: self.disk_page_load_count,
            disk_batch_load_count: self.disk_batch_load_count,
        }
    }

    fn evict_oldest_complete_layers_to_fit_global_maximum(&mut self) {
        while self.resident_payload_byte_count > self.maximum_resident_payload_byte_count {
            let eviction_layer_index = self
                .complete_layer_expert_weights
                .iter()
                .enumerate()
                .filter_map(|(layer_index, cached_complete_layer)| {
                    cached_complete_layer.as_ref().map(|cached_complete_layer| {
                        (
                            layer_index,
                            cached_complete_layer.last_access_sequence_number,
                        )
                    })
                })
                .min_by_key(|(layer_index, last_access_sequence_number)| {
                    (*last_access_sequence_number, *layer_index)
                })
                .map(|(layer_index, _)| layer_index);
            let Some(eviction_layer_index) = eviction_layer_index else {
                break;
            };
            if !self.evict_complete_layer(eviction_layer_index) {
                break;
            }
        }
    }

    /// Demotes oldest complete layers until every paged layer can retain one decode route.
    pub(crate) fn reconcile_complete_layers_for_decode_route_floors(&mut self) -> bool {
        let mut did_evict_complete_layer = false;
        while self
            .current_hybrid_retention_payload_byte_count()
            .is_some_and(|hybrid_retention_payload_byte_count| {
                hybrid_retention_payload_byte_count > self.maximum_resident_payload_byte_count
            })
        {
            let eviction_layer_index = self
                .complete_layer_expert_weights
                .iter()
                .enumerate()
                .filter_map(|(layer_index, complete_layer)| {
                    complete_layer.as_ref().map(|complete_layer| {
                        (layer_index, complete_layer.last_access_sequence_number)
                    })
                })
                .min_by_key(|(layer_index, last_access_sequence_number)| {
                    (*last_access_sequence_number, *layer_index)
                })
                .map(|(layer_index, _)| layer_index);
            let Some(eviction_layer_index) = eviction_layer_index else {
                break;
            };
            if !self.evict_complete_layer(eviction_layer_index) {
                break;
            }
            did_evict_complete_layer = true;
        }
        did_evict_complete_layer
    }

    fn evict_complete_layer(&mut self, layer_index: usize) -> bool {
        let Some(evicted_complete_layer) = self
            .complete_layer_expert_weights
            .get_mut(layer_index)
            .and_then(Option::take)
        else {
            return false;
        };
        self.resident_payload_byte_count = self
            .resident_payload_byte_count
            .saturating_sub(evicted_complete_layer.resident_payload_byte_count);
        self.resident_payload_byte_count_by_layer[layer_index] = self
            .resident_payload_byte_count_by_layer[layer_index]
            .saturating_sub(evicted_complete_layer.resident_payload_byte_count);
        self.eviction_count = self.eviction_count.saturating_add(1);
        true
    }
}

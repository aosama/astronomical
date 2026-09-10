//! Decode/prefill residency status: mode, telemetry, and cache statistics.

use astronomical_ipc_protocol::ExpertMemoryMode;

use crate::ExpertResidencyTelemetry;
use crate::expert_paging::{ExpertWeightMemoryCacheStatistics, RetainedExpertPageCache};
use crate::laguna::normalization::{LagunaFeedForwardDescriptor, LagunaTargetContract};
use crate::memory::DecodeExpertCache;

use super::expert_coverage::{
    resident_complete_payload_bytes, resident_sparse_layer_count, sparse_layer_count,
    sparse_layer_counts,
};
use super::expert_residency::{LagunaExpertResidencyState, LagunaLastExpertForward};
use super::weights::LagunaNativeWeights;

impl LagunaExpertResidencyState {
    pub(super) fn expert_memory_mode(
        &self,
        contract: &LagunaTargetContract,
        weights: &LagunaNativeWeights,
    ) -> ExpertMemoryMode {
        let (sparse_layer_count, resident_layer_count) = sparse_layer_counts(contract, weights);
        let retained_complete_layer_count = self
            .retained_layers
            .borrow()
            .as_ref()
            .map(|retained_layers| retained_layers.statistics().complete_layer_count)
            .unwrap_or(0);
        if sparse_layer_count == 0
            || resident_layer_count.saturating_add(retained_complete_layer_count)
                == sparse_layer_count
        {
            return ExpertMemoryMode::Resident;
        }
        let decode_expert_count = self
            .decode_cache
            .borrow()
            .as_ref()
            .map(DecodeExpertCache::resident_expert_count)
            .unwrap_or(0);
        let expected_decode_experts = self.paging_plan.as_ref().map(|paging_plan| {
            paging_plan
                .sparse_layers()
                .iter()
                .map(|sparse_layer| sparse_layer.expert_capacity())
                .sum::<usize>()
        });
        if expected_decode_experts == Some(decode_expert_count) && decode_expert_count > 0 {
            return ExpertMemoryMode::Resident;
        }
        let retained_payload_bytes = self
            .retained_layers
            .borrow()
            .as_ref()
            .map(|retained_layers| retained_layers.statistics().resident_payload_byte_count)
            .unwrap_or(0);
        let decode_payload_bytes = self
            .decode_cache
            .borrow()
            .as_ref()
            .map(DecodeExpertCache::total_payload_bytes)
            .unwrap_or(0);
        if retained_payload_bytes > 0 || decode_payload_bytes > 0 {
            return ExpertMemoryMode::Hybrid;
        }
        if resident_layer_count == 0 && self.paging_plan.is_some() {
            return ExpertMemoryMode::Paged;
        }
        ExpertMemoryMode::Hybrid
    }

    pub(super) fn expert_residency_telemetry(
        &self,
        contract: &LagunaTargetContract,
        weights: &LagunaNativeWeights,
    ) -> ExpertResidencyTelemetry {
        let statistics = self.expert_weight_memory_cache_statistics(contract, weights);
        let total_layer_count = u32::try_from(sparse_layer_count(contract)).unwrap_or(u32::MAX);
        let last_forward = *self.last_forward.borrow();
        let retained_expert_count =
            if self.expert_memory_mode(contract, weights) == ExpertMemoryMode::Resident {
                // Natively resident experts live in the bound weights, which a
                // fully resident model carries without any paging plan; the
                // roster therefore comes from the normalized geometry.
                contract
                    .layers()
                    .iter()
                    .filter_map(|layer_descriptor| match layer_descriptor.feed_forward() {
                        LagunaFeedForwardDescriptor::Moe(moe_descriptor) => {
                            Some(u64::from(moe_descriptor.expert_count()))
                        }
                        LagunaFeedForwardDescriptor::Dense(_) => None,
                    })
                    .sum::<u64>()
            } else {
                let decode_expert_count: u64 = self
                    .decode_cache
                    .borrow()
                    .as_ref()
                    .map(DecodeExpertCache::resident_expert_count)
                    .unwrap_or(0) as u64;
                let retained_cache_expert_count: u64 = self
                    .retained_layers
                    .borrow()
                    .as_ref()
                    .map_or(0, RetainedExpertPageCache::resident_expert_count)
                    as u64;
                if decode_expert_count > 0 {
                    decode_expert_count
                } else if retained_cache_expert_count > 0 {
                    retained_cache_expert_count
                } else {
                    match last_forward {
                        // A streamed page that nothing retained is still the
                        // resident expert payload this forward materialized,
                        // which is the same grain the payload bytes report.
                        LagunaLastExpertForward::StreamedCompleteLayer { expert_count, .. }
                        | LagunaLastExpertForward::StreamedRoutedPage { expert_count, .. } => {
                            u64::from(expert_count)
                        }
                        LagunaLastExpertForward::None => 0,
                    }
                }
            };
        ExpertResidencyTelemetry {
            total_layer_count,
            resident_expert_count: u32::try_from(retained_expert_count).unwrap_or(u32::MAX),
            resident_expert_payload_bytes: statistics.resident_payload_byte_count,
        }
    }

    pub(super) fn expert_weight_memory_cache_statistics(
        &self,
        contract: &LagunaTargetContract,
        weights: &LagunaNativeWeights,
    ) -> ExpertWeightMemoryCacheStatistics {
        let mode = self.expert_memory_mode(contract, weights);
        let last_forward = *self.last_forward.borrow();
        let cache_statistics = self
            .retained_layers
            .borrow()
            .as_ref()
            .map(RetainedExpertPageCache::statistics)
            .unwrap_or_default();
        let decode_cache = self.decode_cache.borrow();
        let decode_payload_bytes = decode_cache
            .as_ref()
            .map(DecodeExpertCache::total_payload_bytes)
            .unwrap_or(0);
        let decode_expert_count = decode_cache
            .as_ref()
            .map(DecodeExpertCache::resident_expert_count)
            .unwrap_or(0);
        let decode_disk_expert_load_count = decode_cache
            .as_ref()
            .map(DecodeExpertCache::disk_expert_load_count)
            .unwrap_or(0);
        let decode_disk_batch_load_count = decode_cache
            .as_ref()
            .map(DecodeExpertCache::disk_batch_load_count)
            .unwrap_or(0);
        let decode_eviction_count = decode_cache
            .as_ref()
            .map(DecodeExpertCache::eviction_count)
            .unwrap_or(0);
        let (
            complete_layer_count,
            complete_layer_payload_byte_count,
            partial_layer_count,
            partial_layer_payload_byte_count,
        ) = match (mode, last_forward) {
            (ExpertMemoryMode::Resident, _) => {
                let complete_layer_count = resident_sparse_layer_count(contract, weights);
                let native_complete_layer_payload_byte_count = match self.paging_plan.as_ref() {
                    Some(plan) => {
                        resident_complete_payload_bytes(plan, contract, weights).unwrap_or(0)
                    }
                    // Without a paging plan every routed payload is the bound
                    // weight ownership itself.
                    None => weights.resident_routed_expert_payload_bytes(),
                };
                (
                    complete_layer_count.saturating_add(cache_statistics.complete_layer_count),
                    native_complete_layer_payload_byte_count
                        .saturating_add(cache_statistics.complete_layer_payload_byte_count),
                    0,
                    0,
                )
            }
            (ExpertMemoryMode::Hybrid, _) if decode_payload_bytes > 0 => {
                (0, 0, decode_expert_count, decode_payload_bytes)
            }
            (ExpertMemoryMode::Hybrid, _) => (
                cache_statistics.complete_layer_count,
                cache_statistics.complete_layer_payload_byte_count,
                cache_statistics.partial_layer_count,
                cache_statistics.partial_layer_payload_byte_count,
            ),
            (
                _,
                LagunaLastExpertForward::StreamedCompleteLayer {
                    layer_count,
                    payload_bytes,
                    ..
                },
            ) => (layer_count as usize, payload_bytes, 0, 0),
            (
                _,
                LagunaLastExpertForward::StreamedRoutedPage {
                    layer_count,
                    payload_bytes,
                    ..
                },
            ) => (0, 0, layer_count as usize, payload_bytes),
            _ => (0, 0, 0, 0),
        };
        ExpertWeightMemoryCacheStatistics {
            entry_count: complete_layer_count.saturating_add(partial_layer_count),
            resident_payload_byte_count: complete_layer_payload_byte_count
                .saturating_add(partial_layer_payload_byte_count),
            maximum_resident_payload_byte_count: cache_statistics
                .maximum_resident_payload_byte_count
                .max(
                    complete_layer_payload_byte_count
                        .saturating_add(partial_layer_payload_byte_count),
                ),
            eviction_count: cache_statistics
                .eviction_count
                .saturating_add(decode_eviction_count),
            disk_page_load_count: cache_statistics
                .disk_page_load_count
                .saturating_add(decode_disk_expert_load_count),
            disk_batch_load_count: cache_statistics
                .disk_batch_load_count
                .saturating_add(decode_disk_batch_load_count),
            complete_layer_count,
            complete_layer_payload_byte_count,
            partial_layer_count,
            partial_layer_payload_byte_count,
            mandatory_read_promotion_count: cache_statistics.mandatory_read_promotion_count,
            complete_layer_eviction_count: cache_statistics.complete_layer_eviction_count,
            partial_layer_eviction_count: cache_statistics.partial_layer_eviction_count,
        }
    }
}

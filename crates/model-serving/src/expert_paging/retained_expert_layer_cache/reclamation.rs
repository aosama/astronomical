//! Deterministic partial-first reclamation for retained expert ownership.

use super::{RetainedExpertLayerCache, RetainedExpertLayerEntry, RetainedExpertReclamation};
use crate::{ExpertWeightPage, RetainedExpertPageClass};

impl<ExpertPage> RetainedExpertLayerCache<ExpertPage>
where
    ExpertPage: ExpertWeightPage,
{
    /// Releases elastic pages before stable complete layers for an exact deficit.
    pub fn reclaim_for_request_pressure(
        &mut self,
        required_payload_bytes: u64,
    ) -> RetainedExpertReclamation {
        let target_payload_bytes = self
            .resident_payload_bytes
            .saturating_sub(required_payload_bytes);
        self.reclaim_to_payload_ceiling(target_payload_bytes)
    }

    pub(super) fn reclaim_to_effective_ceiling(&mut self) -> RetainedExpertReclamation {
        let effective_maximum_resident_payload_bytes =
            self.effective_maximum_resident_payload_bytes();
        self.reclaim_to_payload_ceiling(effective_maximum_resident_payload_bytes)
    }

    fn reclaim_to_payload_ceiling(
        &mut self,
        payload_ceiling_bytes: u64,
    ) -> RetainedExpertReclamation {
        let mut reclamation = RetainedExpertReclamation::default();
        while self.resident_payload_bytes > payload_ceiling_bytes {
            let Some(layer_index) = self.lowest_coverage_partial_layer_index() else {
                break;
            };
            let released_payload_bytes = self.retained_layers[layer_index]
                .as_ref()
                .map_or(0, |entry| entry.payload_bytes);
            self.remove_layer(layer_index);
            reclamation.released_partial_layer_count =
                reclamation.released_partial_layer_count.saturating_add(1);
            reclamation.released_partial_payload_bytes = reclamation
                .released_partial_payload_bytes
                .saturating_add(released_payload_bytes);
        }
        for layer_index in (0..self.retained_layers.len()).rev() {
            if self.resident_payload_bytes <= payload_ceiling_bytes {
                break;
            }
            let Some(retained_entry) = self.retained_layers[layer_index].as_ref() else {
                continue;
            };
            if retained_entry.class != RetainedExpertPageClass::StableCompleteLayer {
                continue;
            }
            let released_payload_bytes = retained_entry.payload_bytes;
            self.remove_layer(layer_index);
            reclamation.released_complete_layer_count =
                reclamation.released_complete_layer_count.saturating_add(1);
            reclamation.released_complete_payload_bytes = reclamation
                .released_complete_payload_bytes
                .saturating_add(released_payload_bytes);
        }
        reclamation
    }

    fn lowest_coverage_partial_layer_index(&self) -> Option<usize> {
        self.retained_layers
            .iter()
            .enumerate()
            .filter_map(|(layer_index, retained_entry)| {
                let retained_entry = retained_entry.as_ref()?;
                (retained_entry.class == RetainedExpertPageClass::ElasticRoutedExperts)
                    .then_some(layer_index)
            })
            .min_by(|left_layer_index, right_layer_index| {
                let Some(left_entry) = self.retained_layers[*left_layer_index].as_ref() else {
                    return std::cmp::Ordering::Equal;
                };
                let Some(right_entry) = self.retained_layers[*right_layer_index].as_ref() else {
                    return std::cmp::Ordering::Equal;
                };
                let left_demand = self.covered_demand(*left_layer_index, left_entry);
                let right_demand = self.covered_demand(*right_layer_index, right_entry);
                let left_score = u128::from(left_demand) * u128::from(right_entry.payload_bytes);
                let right_score = u128::from(right_demand) * u128::from(left_entry.payload_bytes);
                left_score
                    .cmp(&right_score)
                    .then_with(|| left_layer_index.cmp(right_layer_index))
            })
    }

    fn covered_demand(
        &self,
        layer_index: usize,
        retained_entry: &RetainedExpertLayerEntry<ExpertPage>,
    ) -> u64 {
        retained_entry
            .expert_ids
            .iter()
            .filter_map(|expert_id| self.expert_demand_counts_by_layer[layer_index].get(*expert_id))
            .copied()
            .fold(0_u64, u64::saturating_add)
    }

    pub(super) fn record_eviction(&mut self, class: RetainedExpertPageClass) {
        self.eviction_count = self.eviction_count.saturating_add(1);
        match class {
            RetainedExpertPageClass::StableCompleteLayer => {
                self.complete_layer_eviction_count =
                    self.complete_layer_eviction_count.saturating_add(1);
            }
            RetainedExpertPageClass::ElasticRoutedExperts => {
                self.partial_layer_eviction_count =
                    self.partial_layer_eviction_count.saturating_add(1);
            }
        }
    }
}

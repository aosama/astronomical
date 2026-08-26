//! Deterministic RAM ownership and demand evidence for Rust-loaded expert pages.
//!
//! This is a byte-accounting and ownership container, not a loader. The model
//! performs SafeTensors input/output and MLX evaluation before offering a page
//! to this cache. Consequently, `Some(page)` means the complete selected page is usable;
//! there is no observable partially loaded state.
//!
//! The production policy uses observed route frequency to preserve or reclaim
//! already-owned partial pages. It never turns demand into speculative I/O. The
//! last prefill chunk may count each assignment more than once so prompt-tail
//! coverage remains useful at the transition to decode.

mod contract;
mod reclamation;

pub use contract::{
    RetainedExpertLayerCommit, RetainedExpertLayerCommitDelta, RetainedExpertLayerCommitError,
    RetainedExpertLayerCommitOutcome, RetainedExpertReclamation,
};

use super::{ExpertWeightMemoryCacheStatistics, ExpertWeightPage};
use crate::memory::{CurrentExpertLayerResidency, RetainedExpertPageClass};
#[derive(Debug)]
struct RetainedExpertLayerEntry<ExpertPage> {
    page: ExpertPage,
    class: RetainedExpertPageClass,
    expert_ids: Vec<usize>,
    payload_bytes: u64,
}

/// Keeps one deterministic retained expert page per layer within one byte ceiling.
#[derive(Debug)]
pub struct RetainedExpertPageCache<ExpertPage> {
    /// One stable slot per decoder layer. `None` means execution must stream it.
    retained_layers: Vec<Option<RetainedExpertLayerEntry<ExpertPage>>>,
    /// Cumulative routed demand used to choose a useful page for every layer.
    expert_demand_counts_by_layer: Vec<Vec<u64>>,
    /// Multiplier applied to each recorded assignment. Last-chunk prefill
    /// raises this so tail routes outrank earlier prompt routes of equal count.
    demand_assignment_weight: u64,
    /// Sum of payload bytes for every `Some` slot; metadata overhead is excluded.
    resident_payload_bytes: u64,
    /// Long-lived limit supplied by the composed MLX RAM budget.
    normal_maximum_resident_payload_bytes: u64,
    /// Temporary upper bound installed while one request needs expert bytes back.
    ///
    /// This is separate from the long-lived maximum so finalization can remove
    /// request pressure and refill without guessing the original machine budget.
    request_pressure_maximum_resident_payload_bytes: Option<u64>,
    eviction_count: u64,
    disk_page_load_count: u64,
    disk_batch_load_count: u64,
    mandatory_read_promotion_count: u64,
    complete_layer_eviction_count: u64,
    partial_layer_eviction_count: u64,
}

impl<ExpertPage> RetainedExpertPageCache<ExpertPage>
where
    ExpertPage: ExpertWeightPage,
{
    #[must_use]
    pub fn new(layer_count: usize) -> Self {
        Self {
            retained_layers: (0..layer_count).map(|_| None).collect(),
            expert_demand_counts_by_layer: (0..layer_count).map(|_| Vec::new()).collect(),
            demand_assignment_weight: 1,
            resident_payload_bytes: 0,
            normal_maximum_resident_payload_bytes: 0,
            request_pressure_maximum_resident_payload_bytes: None,
            eviction_count: 0,
            disk_page_load_count: 0,
            disk_batch_load_count: 0,
            mandatory_read_promotion_count: 0,
            complete_layer_eviction_count: 0,
            partial_layer_eviction_count: 0,
        }
    }

    pub fn retained_layer(&self, layer_index: usize) -> Option<&ExpertPage> {
        self.retained_layers
            .get(layer_index)?
            .as_ref()
            .map(|entry| &entry.page)
    }

    pub fn retained_layer_mut(&mut self, layer_index: usize) -> Option<&mut ExpertPage> {
        self.retained_layers
            .get_mut(layer_index)?
            .as_mut()
            .map(|entry| &mut entry.page)
    }

    /// Re-reads the page after an in-place overlay grow.
    pub fn sync_page_payload_bytes(&mut self, layer_index: usize) {
        let Some(entry) = self
            .retained_layers
            .get_mut(layer_index)
            .and_then(|layer_slot| layer_slot.as_mut())
        else {
            return;
        };
        let updated_payload_bytes = entry.page.resident_payload_byte_count();
        self.resident_payload_bytes = self
            .resident_payload_bytes
            .saturating_sub(entry.payload_bytes)
            .saturating_add(updated_payload_bytes);
        entry.payload_bytes = updated_payload_bytes;
    }

    /// Repeats exact projected-byte accounting before a caller transfers ownership.
    #[must_use]
    pub fn can_commit_materialized_page(
        &self,
        layer_index: usize,
        candidate_payload_bytes: u64,
    ) -> bool {
        if candidate_payload_bytes == 0 {
            return false;
        }
        let Some(layer_slot) = self.retained_layers.get(layer_index) else {
            return false;
        };
        let existing_payload_bytes = layer_slot.as_ref().map_or(0, |entry| entry.payload_bytes);
        self.resident_payload_bytes
            .checked_sub(existing_payload_bytes)
            .and_then(|payload_without_replaced_page| {
                payload_without_replaced_page.checked_add(candidate_payload_bytes)
            })
            .is_some_and(|projected_payload_bytes| {
                projected_payload_bytes <= self.effective_maximum_resident_payload_bytes()
            })
    }

    pub fn update_maximum_resident_payload_bytes(
        &mut self,
        maximum_payload_bytes: u64,
    ) -> RetainedExpertReclamation {
        // Store the normal limit independently from temporary request pressure.
        // A live pressure cap still wins through `effective_maximum...`, but the
        // normal value survives so finalization can resume retention immediately
        // without waiting for another budget publication as an accidental repair.
        self.normal_maximum_resident_payload_bytes = maximum_payload_bytes;
        self.reclaim_to_effective_ceiling()
    }

    /// Commits a complete layer loaded by a mandatory prefill read.
    pub fn commit_materialized_complete_layer(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        expert_page: ExpertPage,
    ) -> Result<RetainedExpertLayerCommit<ExpertPage>, RetainedExpertLayerCommitError> {
        if self.retained_layers.get(layer_index).is_none() {
            return Err(RetainedExpertLayerCommitError::LayerOutOfRange { layer_index });
        }
        if expert_capacity == 0 {
            return Err(RetainedExpertLayerCommitError::ZeroExpertCapacity { layer_index });
        }
        if expert_page.resident_payload_byte_count() == 0 {
            return Err(RetainedExpertLayerCommitError::ZeroPayload { layer_index });
        }
        if self.retained_layers[layer_index]
            .as_ref()
            .is_some_and(|entry| entry.class == RetainedExpertPageClass::StableCompleteLayer)
        {
            return Ok(RetainedExpertLayerCommit {
                outcome: RetainedExpertLayerCommitOutcome::PreservedExisting,
                uncommitted_page: Some(expert_page),
            });
        }
        let expert_ids = (0..expert_capacity).collect();
        let commit_outcome = self.commit_entry(
            layer_index,
            RetainedExpertPageClass::StableCompleteLayer,
            expert_ids,
            expert_page,
        )?;
        if matches!(
            commit_outcome.outcome,
            RetainedExpertLayerCommitOutcome::Committed(_)
        ) {
            self.mandatory_read_promotion_count =
                self.mandatory_read_promotion_count.saturating_add(1);
        }
        Ok(commit_outcome)
    }

    /// Commits the first exact routed page loaded by mandatory decode execution.
    pub fn commit_materialized_routed_page(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        expert_ids: Vec<usize>,
        expert_page: ExpertPage,
    ) -> Result<RetainedExpertLayerCommit<ExpertPage>, RetainedExpertLayerCommitError> {
        self.validate_routed_page_metadata(layer_index, expert_capacity, &expert_ids)?;
        if expert_page.resident_payload_byte_count() == 0 {
            return Err(RetainedExpertLayerCommitError::ZeroPayload { layer_index });
        }
        if let Some(existing_entry) = self.retained_layers[layer_index].as_ref() {
            if existing_entry.class == RetainedExpertPageClass::StableCompleteLayer
                || !routed_expert_ids_are_strict_superset(&existing_entry.expert_ids, &expert_ids)
            {
                // Keep the owner unless the caller already merged a strictly
                // larger expert set. Replacing with a disjoint miss page would
                // drop the experts the next token still needs.
                return Ok(RetainedExpertLayerCommit {
                    outcome: RetainedExpertLayerCommitOutcome::PreservedExisting,
                    uncommitted_page: Some(expert_page),
                });
            }
        }
        self.commit_entry(
            layer_index,
            RetainedExpertPageClass::ElasticRoutedExperts,
            expert_ids,
            expert_page,
        )
    }

    fn commit_entry(
        &mut self,
        layer_index: usize,
        class: RetainedExpertPageClass,
        expert_ids: Vec<usize>,
        expert_page: ExpertPage,
    ) -> Result<RetainedExpertLayerCommit<ExpertPage>, RetainedExpertLayerCommitError> {
        let replacement_payload_bytes = expert_page.resident_payload_byte_count();
        let existing_payload_bytes = self.retained_layers[layer_index]
            .as_ref()
            .map_or(0, |entry| entry.payload_bytes);
        let projected_payload_bytes = self
            .resident_payload_bytes
            .checked_sub(existing_payload_bytes)
            .ok_or(RetainedExpertLayerCommitError::InconsistentPayloadAccounting { layer_index })?
            .checked_add(replacement_payload_bytes)
            .ok_or(RetainedExpertLayerCommitError::PayloadByteCountOverflow { layer_index })?;
        if projected_payload_bytes > self.effective_maximum_resident_payload_bytes() {
            return Ok(RetainedExpertLayerCommit {
                outcome: RetainedExpertLayerCommitOutcome::RejectedByCurrentCeiling,
                uncommitted_page: Some(expert_page),
            });
        }
        let replacement_entry = RetainedExpertLayerEntry {
            page: expert_page,
            class,
            expert_ids,
            payload_bytes: replacement_payload_bytes,
        };
        if let Some(replaced_entry) = self.retained_layers[layer_index].replace(replacement_entry) {
            self.record_eviction(replaced_entry.class);
        }
        self.resident_payload_bytes = projected_payload_bytes;
        Ok(RetainedExpertLayerCommit {
            outcome: RetainedExpertLayerCommitOutcome::Committed(RetainedExpertLayerCommitDelta {
                released_payload_bytes: existing_payload_bytes,
                committed_payload_bytes: replacement_payload_bytes,
            }),
            uncommitted_page: None,
        })
    }

    fn validate_routed_page_metadata(
        &self,
        layer_index: usize,
        expert_capacity: usize,
        expert_ids: &[usize],
    ) -> Result<(), RetainedExpertLayerCommitError> {
        if self.retained_layers.get(layer_index).is_none() {
            return Err(RetainedExpertLayerCommitError::LayerOutOfRange { layer_index });
        }
        if expert_capacity == 0 {
            return Err(RetainedExpertLayerCommitError::ZeroExpertCapacity { layer_index });
        }
        if expert_ids.is_empty()
            || expert_ids.len() >= expert_capacity
            || expert_ids.windows(2).any(|ids| ids[0] >= ids[1])
            || expert_ids
                .iter()
                .any(|expert_id| *expert_id >= expert_capacity)
        {
            return Err(RetainedExpertLayerCommitError::InvalidExpertIds { layer_index });
        }
        Ok(())
    }

    /// Removes one stale page before a barrier-safe topology rebuild.
    pub fn remove_layer(&mut self, layer_index: usize) -> bool {
        let Some(layer_slot) = self.retained_layers.get_mut(layer_index) else {
            return false;
        };
        let Some(removed_entry) = layer_slot.take() else {
            return false;
        };
        self.resident_payload_bytes = self
            .resident_payload_bytes
            .saturating_sub(removed_entry.payload_bytes);
        self.record_eviction(removed_entry.class);
        true
    }

    /// Records routed experts without retaining request-owned arrays.
    pub fn record_expert_demand(
        &mut self,
        layer_index: usize,
        expert_capacity: usize,
        selected_expert_ids: &[usize],
    ) {
        let Some(layer_demand_counts) = self.expert_demand_counts_by_layer.get_mut(layer_index)
        else {
            return;
        };
        if layer_demand_counts.len() < expert_capacity {
            layer_demand_counts.resize(expert_capacity, 0);
        }
        let assignment_weight = self.demand_assignment_weight.max(1);
        for selected_expert_id in selected_expert_ids {
            if let Some(demand_count) = layer_demand_counts.get_mut(*selected_expert_id) {
                *demand_count = demand_count.saturating_add(assignment_weight);
            }
        }
    }

    /// Raises or restores the per-assignment demand multiplier.
    pub fn set_demand_assignment_weight(&mut self, assignment_weight: u64) {
        self.demand_assignment_weight = assignment_weight.max(1);
    }

    /// Starts a fresh evidence window after one topology plan consumes demand.
    pub fn clear_expert_demand(&mut self) {
        for layer_demand_counts in &mut self.expert_demand_counts_by_layer {
            layer_demand_counts.fill(0);
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

    /// Returns planner-ready ownership metadata without exposing page arrays.
    #[must_use]
    pub fn topology_snapshot(&self, _expert_capacity: usize) -> Vec<CurrentExpertLayerResidency> {
        self.retained_layers
            .iter()
            .enumerate()
            .filter_map(|(layer_index, retained_entry)| {
                let retained_entry = retained_entry.as_ref()?;
                Some(CurrentExpertLayerResidency {
                    layer_index,
                    class: retained_entry.class,
                    retained_expert_ids: retained_entry.expert_ids.clone(),
                    payload_bytes: retained_entry.payload_bytes,
                    covered_weighted_demand: retained_entry
                        .expert_ids
                        .iter()
                        .filter_map(|expert_id| {
                            self.expert_demand_counts_by_layer[layer_index].get(*expert_id)
                        })
                        .copied()
                        .fold(0_u64, u64::saturating_add),
                })
            })
            .collect()
    }

    /// Freezes retained pages at a smaller ceiling so the remaining prompt fits.
    ///
    /// `reclamation_target_bytes` means "please free about this many expert
    /// bytes". This method turns that request into an absolute cap:
    /// current owned payload minus the requested release. Whole layers are
    /// the only eviction unit, so the cache may free more than asked and must
    /// never free less when enough payload exists.
    ///
    /// The long-lived normal maximum stays stored. Decode handoff later calls
    /// `resume_after_request_pressure` to make that normal budget visible
    /// again. Leaving this freeze in place is what used to keep generation at
    /// about one gigabyte of experts.
    pub fn limit_for_request_pressure(&mut self, reclamation_target_bytes: u64) -> bool {
        let pressure_maximum = self
            .resident_payload_bytes
            .saturating_sub(reclamation_target_bytes);
        self.limit_for_request_pressure_to_maximum(pressure_maximum)
    }

    /// Installs an absolute request-scoped ceiling derived from one admitted forward.
    ///
    /// Unlike deficit-based reclamation, this can leave room for mandatory reads
    /// to become retained while preventing those reads from consuming the exact
    /// context, streaming-page, and transient reserve already admitted.
    pub fn limit_for_request_pressure_to_maximum(
        &mut self,
        pressure_maximum_resident_payload_bytes: u64,
    ) -> bool {
        self.request_pressure_maximum_resident_payload_bytes =
            Some(pressure_maximum_resident_payload_bytes);
        self.reclaim_to_effective_ceiling().released_payload_bytes() > 0
    }

    /// Lifts the temporary request-pressure freeze.
    ///
    /// This does not load pages. It only forgets the smaller cap so
    /// `effective_maximum_resident_payload_bytes` returns the long-lived
    /// normal budget again. Returns `true` when a freeze was actually present.
    pub fn resume_after_request_pressure(&mut self) -> bool {
        self.request_pressure_maximum_resident_payload_bytes
            .take()
            .is_some()
    }

    pub fn release_all(&mut self) -> bool {
        let had_retained_layers = self.resident_payload_bytes > 0;
        for layer_index in 0..self.retained_layers.len() {
            self.remove_layer(layer_index);
        }
        self.resident_payload_bytes = 0;
        had_retained_layers
    }

    #[must_use]
    pub fn resident_expert_count(&self) -> usize {
        self.retained_layers
            .iter()
            .flatten()
            .map(|entry| entry.expert_ids.len())
            .sum()
    }

    #[must_use]
    pub fn statistics(&self) -> ExpertWeightMemoryCacheStatistics {
        let complete_layer_count = self
            .retained_layers
            .iter()
            .flatten()
            .filter(|entry| entry.class == RetainedExpertPageClass::StableCompleteLayer)
            .count();
        let complete_layer_payload_byte_count = self
            .retained_layers
            .iter()
            .flatten()
            .filter(|entry| entry.class == RetainedExpertPageClass::StableCompleteLayer)
            .map(|entry| entry.payload_bytes)
            .fold(0_u64, u64::saturating_add);
        let partial_layer_count = self
            .retained_layers
            .iter()
            .flatten()
            .filter(|entry| entry.class == RetainedExpertPageClass::ElasticRoutedExperts)
            .count();
        ExpertWeightMemoryCacheStatistics {
            entry_count: self
                .retained_layers
                .iter()
                .filter(|layer| layer.is_some())
                .count(),
            resident_payload_byte_count: self.resident_payload_bytes,
            maximum_resident_payload_byte_count: self.effective_maximum_resident_payload_bytes(),
            eviction_count: self.eviction_count,
            disk_page_load_count: self.disk_page_load_count,
            disk_batch_load_count: self.disk_batch_load_count,
            complete_layer_count,
            complete_layer_payload_byte_count,
            partial_layer_count,
            partial_layer_payload_byte_count: self
                .resident_payload_bytes
                .saturating_sub(complete_layer_payload_byte_count),
            mandatory_read_promotion_count: self.mandatory_read_promotion_count,
            complete_layer_eviction_count: self.complete_layer_eviction_count,
            partial_layer_eviction_count: self.partial_layer_eviction_count,
        }
    }

    /// The tighter of the long-lived budget and any live request-pressure freeze.
    ///
    /// While a freeze is present, every plan, replace, and evict uses the
    /// smaller number. After `resume_after_request_pressure`, only the
    /// long-lived budget remains.
    fn effective_maximum_resident_payload_bytes(&self) -> u64 {
        self.normal_maximum_resident_payload_bytes.min(
            self.request_pressure_maximum_resident_payload_bytes
                .unwrap_or(u64::MAX),
        )
    }
}

fn routed_expert_ids_are_strict_superset(
    existing_expert_ids: &[usize],
    proposed_expert_ids: &[usize],
) -> bool {
    if proposed_expert_ids.len() <= existing_expert_ids.len() {
        return false;
    }
    existing_expert_ids.iter().all(|existing_expert_id| {
        proposed_expert_ids
            .binary_search(existing_expert_id)
            .is_ok()
    })
}

/// Scales last-chunk assignments so their token density matches earlier prefill.
///
/// Decode continues from the prompt tail. One last-chunk token therefore counts
/// as many assignments as `earlier / last_chunk`, with a floor of one.
#[must_use]
pub fn last_prefill_chunk_demand_weight(
    earlier_prefill_token_count: u64,
    last_prefill_chunk_token_count: u64,
) -> u64 {
    if earlier_prefill_token_count == 0 || last_prefill_chunk_token_count == 0 {
        return 1;
    }
    (earlier_prefill_token_count / last_prefill_chunk_token_count).max(1)
}

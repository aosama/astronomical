//! Deterministic RAM ownership and demand evidence for Rust-loaded expert pages.
//!
//! This is a byte-accounting and ownership container, not a loader. The model
//! performs SafeTensors input/output and MLX evaluation before offering a page
//! to this cache. Consequently, `Some(page)` means the complete selected page is usable;
//! there is no observable partially loaded state.
//!
//! The production policy ranks experts globally by observed route frequency.
//! The last prefill chunk may count each assignment more than once so decode
//! pages follow the prompt tail. Stable identifiers break ties.

use super::{ExpertWeightMemoryCacheStatistics, ExpertWeightPage};
use thiserror::Error;

#[derive(Clone, Copy)]
struct ExpertCandidate {
    layer_index: usize,
    expert_id: usize,
    demand_count: u64,
    payload_bytes: u64,
}

/// Invalid immutable geometry supplied to the retained-page planner.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RetainedExpertPagePlanError {
    #[error(
        "retained-page geometry has {payload_layer_count} payload layers and {expert_capacity_layer_count} capacity layers, expected {retained_layer_count}"
    )]
    LayerCountMismatch {
        retained_layer_count: usize,
        payload_layer_count: usize,
        expert_capacity_layer_count: usize,
    },
    #[error("retained-page geometry has zero expert capacity at layer {layer_index}")]
    ZeroExpertCapacity { layer_index: usize },
    #[error("retained-page geometry has zero payload at layer {layer_index}")]
    ZeroLayerPayload { layer_index: usize },
}

/// Keeps one deterministic retained expert page per layer within one byte ceiling.
#[derive(Debug)]
pub struct RetainedExpertLayerCache<ExpertPage> {
    /// One stable slot per decoder layer. `None` means execution must stream it.
    retained_layers: Vec<Option<ExpertPage>>,
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
}

impl<ExpertPage> RetainedExpertLayerCache<ExpertPage>
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
        }
    }

    pub fn retained_layer(&self, layer_index: usize) -> Option<&ExpertPage> {
        self.retained_layers.get(layer_index)?.as_ref()
    }

    pub fn update_maximum_resident_payload_bytes(&mut self, maximum_payload_bytes: u64) {
        // Store the normal limit independently from temporary request pressure.
        // A live pressure cap still wins through `effective_maximum...`, but the
        // normal value survives so finalization can resume retention immediately
        // without waiting for another budget publication as an accidental repair.
        self.normal_maximum_resident_payload_bytes = maximum_payload_bytes;
        self.evict_highest_layers_to_fit();
    }

    /// Replaces one layer atomically when the replacement fits the shared ceiling.
    pub fn replace_layer(&mut self, layer_index: usize, expert_page: ExpertPage) -> bool {
        let replacement_payload_bytes = expert_page.resident_payload_byte_count();
        let effective_maximum_resident_payload_bytes =
            self.effective_maximum_resident_payload_bytes();
        let Some(layer_slot) = self.retained_layers.get_mut(layer_index) else {
            return false;
        };
        let existing_payload_bytes = layer_slot
            .as_ref()
            .map_or(0, ExpertWeightPage::resident_payload_byte_count);
        let projected_payload_bytes = self
            .resident_payload_bytes
            .saturating_sub(existing_payload_bytes)
            .saturating_add(replacement_payload_bytes);
        if projected_payload_bytes > effective_maximum_resident_payload_bytes {
            return false;
        }
        if layer_slot.replace(expert_page).is_some() {
            self.eviction_count = self.eviction_count.saturating_add(1);
        }
        self.resident_payload_bytes = projected_payload_bytes;
        true
    }

    /// Removes one stale page before a barrier-safe topology rebuild.
    pub fn remove_layer(&mut self, layer_index: usize) -> bool {
        let Some(layer_slot) = self.retained_layers.get_mut(layer_index) else {
            return false;
        };
        let Some(removed_page) = layer_slot.take() else {
            return false;
        };
        self.resident_payload_bytes = self
            .resident_payload_bytes
            .saturating_sub(removed_page.resident_payload_byte_count());
        self.eviction_count = self.eviction_count.saturating_add(1);
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

    /// Selects experts globally by observed route frequency.
    #[must_use]
    pub fn preferred_expert_ids_for_global_budget(
        &self,
        complete_layer_payload_bytes: &[u64],
        expert_capacities: &[usize],
    ) -> Result<Vec<Vec<usize>>, RetainedExpertPagePlanError> {
        let layer_count = self.expert_demand_counts_by_layer.len();
        if complete_layer_payload_bytes.len() != layer_count
            || expert_capacities.len() != layer_count
        {
            return Err(RetainedExpertPagePlanError::LayerCountMismatch {
                retained_layer_count: layer_count,
                payload_layer_count: complete_layer_payload_bytes.len(),
                expert_capacity_layer_count: expert_capacities.len(),
            });
        }
        let mut expert_candidates = Vec::new();
        for layer_index in 0..layer_count {
            let expert_capacity = expert_capacities[layer_index];
            if expert_capacity == 0 {
                return Err(RetainedExpertPagePlanError::ZeroExpertCapacity { layer_index });
            }
            if complete_layer_payload_bytes[layer_index] == 0 {
                return Err(RetainedExpertPagePlanError::ZeroLayerPayload { layer_index });
            }
            // Round upward so planning never admits more payload than the cache
            // ceiling when unusual tensor geometry is not exactly divisible.
            let expert_capacity_bytes = u64::try_from(expert_capacity).unwrap_or(u64::MAX);
            let payload_bytes_per_expert = complete_layer_payload_bytes[layer_index]
                .saturating_add(expert_capacity_bytes.saturating_sub(1))
                / expert_capacity_bytes.max(1);
            for expert_id in 0..expert_capacity {
                let demand_count = self.expert_demand_counts_by_layer[layer_index]
                    .get(expert_id)
                    .copied()
                    .unwrap_or(0);
                expert_candidates.push(ExpertCandidate {
                    layer_index,
                    expert_id,
                    demand_count,
                    payload_bytes: payload_bytes_per_expert,
                });
            }
        }
        expert_candidates.sort_unstable_by(|left, right| {
            right
                .demand_count
                .cmp(&left.demand_count)
                .then_with(|| left.layer_index.cmp(&right.layer_index))
                .then_with(|| left.expert_id.cmp(&right.expert_id))
        });

        let mut preferred_expert_ids_by_layer =
            (0..layer_count).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut remaining_payload_bytes = self.effective_maximum_resident_payload_bytes();
        for expert_candidate in expert_candidates {
            if expert_candidate.payload_bytes > remaining_payload_bytes {
                continue;
            }
            preferred_expert_ids_by_layer[expert_candidate.layer_index]
                .push(expert_candidate.expert_id);
            remaining_payload_bytes =
                remaining_payload_bytes.saturating_sub(expert_candidate.payload_bytes);
        }
        for preferred_expert_ids in &mut preferred_expert_ids_by_layer {
            preferred_expert_ids.sort_unstable();
        }
        Ok(preferred_expert_ids_by_layer)
    }

    pub fn record_disk_load(&mut self, expert_count: usize, batch_count: usize) {
        self.disk_page_load_count = self
            .disk_page_load_count
            .saturating_add(expert_count as u64);
        self.disk_batch_load_count = self
            .disk_batch_load_count
            .saturating_add(batch_count as u64);
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
        self.request_pressure_maximum_resident_payload_bytes = Some(pressure_maximum);
        let payload_before_eviction = self.resident_payload_bytes;
        self.evict_highest_layers_to_fit();
        self.resident_payload_bytes < payload_before_eviction
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
        for retained_layer in &mut self.retained_layers {
            if retained_layer.take().is_some() {
                self.eviction_count = self.eviction_count.saturating_add(1);
            }
        }
        self.resident_payload_bytes = 0;
        had_retained_layers
    }

    #[must_use]
    pub fn statistics(&self) -> ExpertWeightMemoryCacheStatistics {
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
        }
    }

    fn evict_highest_layers_to_fit(&mut self) {
        // Reverse order is policy, not an arbitrary implementation detail. It
        // leaves the lowest contiguous prefix resident after every shrink.
        let effective_maximum_resident_payload_bytes =
            self.effective_maximum_resident_payload_bytes();
        for layer_slot in self.retained_layers.iter_mut().rev() {
            if self.resident_payload_bytes <= effective_maximum_resident_payload_bytes {
                break;
            }
            if let Some(evicted_layer) = layer_slot.take() {
                self.resident_payload_bytes = self
                    .resident_payload_bytes
                    .saturating_sub(evicted_layer.resident_payload_byte_count());
                self.eviction_count = self.eviction_count.saturating_add(1);
            }
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

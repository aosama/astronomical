//! Deterministic RAM ownership for complete Rust-loaded expert layers.
//!
//! This is a byte-accounting and ownership container, not a loader. The model
//! performs SafeTensors I/O and MLX evaluation before offering a complete layer
//! to this cache. Consequently, `Some(page)` means the entire layer is usable;
//! there is no observable partially loaded state.
//!
//! The production policy currently keeps an ascending layer prefix. Shrinking a
//! budget evicts from the highest index downward, preserving that invariant. A
//! future evidence-backed layer-selection policy must change this owner and its
//! tests explicitly rather than relying on incidental `Vec` ordering.

use super::{ExpertWeightMemoryCacheStatistics, ExpertWeightPage};

/// Keeps complete layers in ascending layer order within one byte ceiling.
#[derive(Debug)]
pub struct RetainedExpertLayerCache<ExpertPage> {
    /// One stable slot per decoder layer. `None` means execution must stream it.
    retained_layers: Vec<Option<ExpertPage>>,
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

    pub fn retain_complete_layer(&mut self, layer_index: usize, expert_page: ExpertPage) -> bool {
        let effective_maximum_resident_payload_bytes =
            self.effective_maximum_resident_payload_bytes();
        let Some(layer_slot) = self.retained_layers.get_mut(layer_index) else {
            return false;
        };
        if layer_slot.is_some() {
            // Retention is idempotent. Never replace a valid layer with a second
            // owner, because old lazy graphs may still reference the first page.
            return true;
        }
        let page_payload_bytes = expert_page.resident_payload_byte_count();
        if self
            .resident_payload_bytes
            .saturating_add(page_payload_bytes)
            > effective_maximum_resident_payload_bytes
        {
            return false;
        }
        *layer_slot = Some(expert_page);
        self.resident_payload_bytes = self
            .resident_payload_bytes
            .saturating_add(page_payload_bytes);
        true
    }

    #[must_use]
    pub fn can_retain_additional_payload_bytes(&self, additional_payload_bytes: u64) -> bool {
        // This pre-I/O predicate mirrors the exact byte check in
        // `retain_complete_layer`. Saturation fails closed: an overflowing sum is
        // larger than every practical budget and therefore cannot be admitted.
        self.resident_payload_bytes
            .saturating_add(additional_payload_bytes)
            <= self.effective_maximum_resident_payload_bytes()
    }

    pub fn record_disk_load(&mut self, expert_count: usize, batch_count: usize) {
        self.disk_page_load_count = self
            .disk_page_load_count
            .saturating_add(expert_count as u64);
        self.disk_batch_load_count = self
            .disk_batch_load_count
            .saturating_add(batch_count as u64);
    }

    pub fn limit_for_request_pressure(&mut self, reclamation_target_bytes: u64) -> bool {
        // Convert "release N bytes" into an absolute cap based on ownership at the
        // decision boundary. Whole-layer granularity may release more than N; it
        // must never release less when enough retained payload exists.
        let pressure_maximum = self
            .resident_payload_bytes
            .saturating_sub(reclamation_target_bytes);
        self.request_pressure_maximum_resident_payload_bytes = Some(pressure_maximum);
        let payload_before_eviction = self.resident_payload_bytes;
        self.evict_highest_layers_to_fit();
        self.resident_payload_bytes < payload_before_eviction
    }

    pub fn resume_after_request_pressure(&mut self) -> bool {
        // Removing the temporary cap does not perform I/O. A barrier-safe caller
        // decides when and how to refill the now-available normal budget.
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

    fn effective_maximum_resident_payload_bytes(&self) -> u64 {
        self.normal_maximum_resident_payload_bytes.min(
            self.request_pressure_maximum_resident_payload_bytes
                .unwrap_or(u64::MAX),
        )
    }
}

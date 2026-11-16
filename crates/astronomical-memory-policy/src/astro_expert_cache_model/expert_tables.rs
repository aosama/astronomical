//! The astro expert cache as the engine can actually hold it: one table per layer.
//!
//! `astro_expert_cache` treats memory as perfectly divisible. An engine cannot: a layer's
//! experts live in one stacked tensor, so an empty slot frees nothing and
//! memory moves between layers only by reallocating a table. This module is
//! the same astro expert cache under that constraint, and every decision it takes is a
//! `TableChange` an engine can carry out on its tensors.
//!
//! A table behaves like a growable vector. A miss takes a free slot if the
//! table has one; otherwise the table grows by a step. When the step does
//! not fit, the layer with the most idle memory is compacted: rebuilt
//! smaller without any expert no token has routed to for a while. One
//! compaction therefore pays for many later misses, and only a few layers
//! ever stop being whole. When no layer has anything idle, the missing
//! expert replaces its own layer's least recently used one in place.
//!
//! Replayed on two recorded replies (`recorded_reply_replay_tests`), this
//! reads about 1.2 experts per token on a first reply and 0.1 when the same
//! text is served again, where dividing memory by layer in advance read 10
//! to 13 on every token. It beats even the ideal divisible astro expert cache on a first
//! reply (2.7), because a batch compaction takes the never-used experts of
//! one layer at once instead of guessing among experts that all look alike.

use super::astro_expert_cache::{AstroExpertCacheLayerGeometry, ExpertKey};
use super::astro_expert_cache_record::AstroExpertCacheRecord;

/// How tables grow and what counts as idle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AstroExpertCacheTableRules {
    /// Slots a table grows by. Measured on the recorded replies: 8 doubled
    /// the reallocations for the same reads, 32 left more memory allocated
    /// and empty and read more.
    pub growth_step_slots: usize,
    /// An expert no token has routed to for this many tokens may be dropped
    /// in a compaction. Longer was better at every length tried, up to
    /// nearly a whole reply; shorter drops experts the reply comes back to.
    pub idle_window_tokens: u64,
}

impl Default for AstroExpertCacheTableRules {
    fn default() -> Self {
        Self {
            growth_step_slots: 16,
            idle_window_tokens: 200,
        }
    }
}

/// One thing the engine must do to its tables, in the order given.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TableChange {
    /// Reallocate the layer's table larger, keeping what it holds.
    Grow {
        layer_index: usize,
        slot_capacity: usize,
    },
    /// Reallocate the layer's table smaller, without the dropped experts.
    Compact {
        layer_index: usize,
        slot_capacity: usize,
        dropped_expert_ids: Vec<u16>,
    },
    /// Read the expert from storage into a free slot of its layer's table.
    Write { expert: ExpertKey },
    /// Read the expert from storage over an expert of the same layer.
    Replace {
        expert: ExpertKey,
        evicted_expert_id: u16,
    },
    /// Read the expert from storage for this token only: nothing could be
    /// given up for it.
    ServeWithoutKeeping { expert: ExpertKey },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HeldExpert {
    NotHeld,
    /// Held since before any token routed to it.
    NeverUsed,
    UsedAtToken(u64),
}

#[derive(Clone, Debug)]
struct LayerTable {
    slot_capacity: usize,
    held_expert_count: usize,
    held_experts: Vec<HeldExpert>,
}

enum FreedSlot {
    Free,
    TakenFrom(u16),
    None,
}

/// Every layer's table, the bytes they may take together, and the record of
/// what serving each token did to them.
#[derive(Debug)]
pub struct AstroExpertCacheTables {
    layer_geometries: Vec<AstroExpertCacheLayerGeometry>,
    layer_tables: Vec<LayerTable>,
    rules: AstroExpertCacheTableRules,
    capacity_bytes: u64,
    allocated_bytes: u64,
    current_token: u64,
    reallocation_count: u64,
    record: AstroExpertCacheRecord,
}

impl AstroExpertCacheTables {
    #[must_use]
    pub fn new(
        layer_geometries: Vec<AstroExpertCacheLayerGeometry>,
        capacity_bytes: u64,
        rules: AstroExpertCacheTableRules,
    ) -> Self {
        let mut layer_tables = Vec::with_capacity(layer_geometries.len());
        let mut expert_counts = Vec::with_capacity(layer_geometries.len());
        for layer_geometry in &layer_geometries {
            layer_tables.push(LayerTable {
                slot_capacity: 0,
                held_expert_count: 0,
                held_experts: vec![HeldExpert::NotHeld; usize::from(layer_geometry.expert_count)],
            });
            expert_counts.push(layer_geometry.expert_count);
        }
        Self {
            record: AstroExpertCacheRecord::for_layers(expert_counts.into_iter()),
            layer_geometries,
            layer_tables,
            rules,
            capacity_bytes,
            allocated_bytes: 0,
            current_token: 0,
            reallocation_count: 0,
        }
    }

    #[must_use]
    pub const fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    #[must_use]
    pub const fn reallocation_count(&self) -> u64 {
        self.reallocation_count
    }

    #[must_use]
    pub fn slot_capacity(&self, layer_index: usize) -> usize {
        self.layer_tables[layer_index].slot_capacity
    }

    #[must_use]
    pub fn held_expert_count(&self, layer_index: usize) -> usize {
        self.layer_tables[layer_index].held_expert_count
    }

    #[must_use]
    pub fn is_held(&self, expert: ExpertKey) -> bool {
        self.held(expert) != HeldExpert::NotHeld
    }

    #[must_use]
    pub const fn record(&self) -> &AstroExpertCacheRecord {
        &self.record
    }

    /// Takes over a table the engine already holds, such as a layer read in
    /// whole while the prompt was processed. Its experts count as never
    /// used until a token routes to them.
    pub fn adopt_table(
        &mut self,
        layer_index: usize,
        slot_capacity: usize,
        held_expert_ids: &[u16],
    ) {
        let expert_payload_bytes = self.layer_geometries[layer_index].expert_payload_bytes;
        let layer_table = &mut self.layer_tables[layer_index];
        self.allocated_bytes -= slots_as_bytes(layer_table.slot_capacity, expert_payload_bytes);
        self.allocated_bytes += slots_as_bytes(slot_capacity, expert_payload_bytes);
        layer_table.slot_capacity = slot_capacity;
        layer_table.held_expert_count = 0;
        layer_table.held_experts.fill(HeldExpert::NotHeld);
        for &expert_id in held_expert_ids {
            if layer_table.held_experts[usize::from(expert_id)] == HeldExpert::NotHeld {
                layer_table.held_experts[usize::from(expert_id)] = HeldExpert::NeverUsed;
                layer_table.held_expert_count += 1;
            }
        }
        for &expert_id in held_expert_ids {
            self.record.record_seated(ExpertKey {
                layer_index,
                expert_id,
            });
        }
    }

    pub fn begin_token(&mut self) {
        self.current_token += 1;
        self.record.begin_token();
    }

    /// Serves one layer's route for the current token and returns what the
    /// engine must do to its tables for it, in order.
    pub fn serve(&mut self, layer_index: usize, routed_expert_ids: &[u16]) -> Vec<TableChange> {
        self.serve_token(&[(layer_index, routed_expert_ids.to_vec())])
    }

    /// Serves every layer's route of the current token at once.
    ///
    /// Every expert the token used is marked used before anything is given
    /// up. Otherwise a miss in one layer could compact away an expert that a
    /// later layer of the same token was served from, and the token would be
    /// the one that both used it and lost it.
    pub fn serve_token(&mut self, layer_routes: &[(usize, Vec<u16>)]) -> Vec<TableChange> {
        let mut missing_experts = Vec::new();
        for (layer_index, routed_expert_ids) in layer_routes {
            for &expert_id in routed_expert_ids {
                let expert = ExpertKey {
                    layer_index: *layer_index,
                    expert_id,
                };
                let is_hit = self.is_held(expert);
                self.record.record_lookup(expert, is_hit);
                if is_hit {
                    self.mark_used(expert);
                } else {
                    missing_experts.push(expert);
                }
            }
        }
        let mut table_changes = Vec::new();
        for expert in missing_experts {
            // A route may name the same expert twice; the first read holds it.
            if !self.is_held(expert) {
                self.read_missing_expert(expert, &mut table_changes);
            }
        }
        table_changes
    }

    /// Records a route that is already known to have been served wholly from
    /// memory, such as one checked on the graphics processor.
    pub fn record_served_from_memory(&mut self, layer_index: usize, routed_expert_ids: &[u16]) {
        for &expert_id in routed_expert_ids {
            let expert = ExpertKey {
                layer_index,
                expert_id,
            };
            self.record.record_lookup(expert, true);
            self.mark_used(expert);
        }
    }

    /// Changes how many bytes the tables may take and compacts until they
    /// fit, idle experts first.
    pub fn set_capacity(&mut self, capacity_bytes: u64) -> Vec<TableChange> {
        self.capacity_bytes = capacity_bytes;
        let mut table_changes = Vec::new();
        while self.allocated_bytes > self.capacity_bytes {
            if self.compact_layer_with_most_idle_memory(None, None, &mut table_changes) {
                continue;
            }
            if !self.shrink_layer_holding_least_recent_expert(&mut table_changes) {
                break;
            }
        }
        table_changes
    }

    fn read_missing_expert(&mut self, expert: ExpertKey, table_changes: &mut Vec<TableChange>) {
        let freed_slot = if self.has_free_slot(expert.layer_index) {
            FreedSlot::Free
        } else {
            self.free_a_slot_for(expert, table_changes)
        };
        match freed_slot {
            FreedSlot::Free => {
                self.record.record_read(expert, true);
                self.hold(expert);
                table_changes.push(TableChange::Write { expert });
            }
            FreedSlot::TakenFrom(evicted_expert_id) => {
                self.record.record_read(expert, true);
                self.hold(expert);
                table_changes.push(TableChange::Replace {
                    expert,
                    evicted_expert_id,
                });
            }
            FreedSlot::None => {
                self.record.record_read(expert, false);
                table_changes.push(TableChange::ServeWithoutKeeping { expert });
            }
        }
    }

    fn free_a_slot_for(
        &mut self,
        needed: ExpertKey,
        table_changes: &mut Vec<TableChange>,
    ) -> FreedSlot {
        let growth_slots = self.growth_slots(needed.layer_index);
        if growth_slots > 0 && !self.fits(needed.layer_index, growth_slots) {
            self.compact_layer_with_most_idle_memory(
                Some(needed.layer_index),
                Some(needed),
                table_changes,
            );
        }
        if growth_slots > 0 && self.fits(needed.layer_index, growth_slots) {
            self.grow(needed.layer_index, growth_slots, table_changes);
            return FreedSlot::Free;
        }
        match self.least_recently_used_expert_in(needed.layer_index) {
            Some(evicted_expert_id) => {
                self.release(
                    Some(needed),
                    ExpertKey {
                        layer_index: needed.layer_index,
                        expert_id: evicted_expert_id,
                    },
                );
                FreedSlot::TakenFrom(evicted_expert_id)
            }
            None => FreedSlot::None,
        }
    }

    fn growth_slots(&self, layer_index: usize) -> usize {
        let expert_count = usize::from(self.layer_geometries[layer_index].expert_count);
        let slot_capacity = self.layer_tables[layer_index].slot_capacity;
        self.rules
            .growth_step_slots
            .min(expert_count.saturating_sub(slot_capacity))
    }

    fn fits(&self, layer_index: usize, growth_slots: usize) -> bool {
        let growth_bytes = slots_as_bytes(
            growth_slots,
            self.layer_geometries[layer_index].expert_payload_bytes,
        );
        self.allocated_bytes.saturating_add(growth_bytes) <= self.capacity_bytes
    }

    fn grow(
        &mut self,
        layer_index: usize,
        growth_slots: usize,
        table_changes: &mut Vec<TableChange>,
    ) {
        let layer_table = &mut self.layer_tables[layer_index];
        layer_table.slot_capacity += growth_slots;
        self.allocated_bytes += slots_as_bytes(
            growth_slots,
            self.layer_geometries[layer_index].expert_payload_bytes,
        );
        self.reallocation_count += 1;
        table_changes.push(TableChange::Grow {
            layer_index,
            slot_capacity: layer_table.slot_capacity,
        });
    }

    /// Compacts the layer whose idle experts free the most memory. Returns
    /// whether any layer had memory to give.
    fn compact_layer_with_most_idle_memory(
        &mut self,
        spared_layer_index: Option<usize>,
        needed: Option<ExpertKey>,
        table_changes: &mut Vec<TableChange>,
    ) -> bool {
        let mut most_freed: Option<(u64, usize)> = None;
        for layer_index in 0..self.layer_tables.len() {
            if spared_layer_index == Some(layer_index) {
                continue;
            }
            let freed_bytes = self.bytes_freed_by_compacting(layer_index);
            if freed_bytes > 0 && most_freed.is_none_or(|(most_bytes, _)| freed_bytes > most_bytes)
            {
                most_freed = Some((freed_bytes, layer_index));
            }
        }
        let Some((_freed_bytes, layer_index)) = most_freed else {
            return false;
        };
        let idle_expert_ids = self.idle_expert_ids(layer_index);
        self.compact(layer_index, &idle_expert_ids, needed, table_changes);
        true
    }

    fn bytes_freed_by_compacting(&self, layer_index: usize) -> u64 {
        let layer_table = &self.layer_tables[layer_index];
        let idle_expert_count = self.idle_expert_ids(layer_index).len();
        let compacted_slot_capacity = self.slots_for(
            layer_index,
            layer_table.held_expert_count - idle_expert_count,
        );
        slots_as_bytes(
            layer_table
                .slot_capacity
                .saturating_sub(compacted_slot_capacity),
            self.layer_geometries[layer_index].expert_payload_bytes,
        )
    }

    /// The held experts no token has routed to within the idle window.
    fn idle_expert_ids(&self, layer_index: usize) -> Vec<u16> {
        let mut idle_expert_ids = Vec::new();
        for (expert_id, held_expert) in self.layer_tables[layer_index]
            .held_experts
            .iter()
            .enumerate()
        {
            let is_idle = match *held_expert {
                HeldExpert::NotHeld => false,
                HeldExpert::NeverUsed => true,
                HeldExpert::UsedAtToken(used_at_token) => {
                    used_at_token.saturating_add(self.rules.idle_window_tokens) < self.current_token
                }
            };
            if is_idle {
                idle_expert_ids.push(u16::try_from(expert_id).unwrap_or(u16::MAX));
            }
        }
        idle_expert_ids
    }

    /// The slots a table needs for this many experts: whole growth steps, so
    /// the next few misses find a free slot, and never more than the layer.
    fn slots_for(&self, layer_index: usize, held_expert_count: usize) -> usize {
        let growth_step_slots = self.rules.growth_step_slots.max(1);
        let expert_count = usize::from(self.layer_geometries[layer_index].expert_count);
        held_expert_count
            .div_ceil(growth_step_slots)
            .saturating_mul(growth_step_slots)
            .min(expert_count)
    }

    fn compact(
        &mut self,
        layer_index: usize,
        dropped_expert_ids: &[u16],
        needed: Option<ExpertKey>,
        table_changes: &mut Vec<TableChange>,
    ) {
        for &expert_id in dropped_expert_ids {
            self.release(
                needed,
                ExpertKey {
                    layer_index,
                    expert_id,
                },
            );
        }
        let held_expert_count = self.layer_tables[layer_index].held_expert_count;
        let compacted_slot_capacity = self.slots_for(layer_index, held_expert_count);
        let freed_slots = self.layer_tables[layer_index]
            .slot_capacity
            .saturating_sub(compacted_slot_capacity);
        self.layer_tables[layer_index].slot_capacity = compacted_slot_capacity;
        self.allocated_bytes -= slots_as_bytes(
            freed_slots,
            self.layer_geometries[layer_index].expert_payload_bytes,
        );
        self.reallocation_count += 1;
        table_changes.push(TableChange::Compact {
            layer_index,
            slot_capacity: compacted_slot_capacity,
            dropped_expert_ids: dropped_expert_ids.to_vec(),
        });
    }

    /// With nothing idle anywhere, gives back one growth step from the layer
    /// that holds the least recently used expert. Returns whether it could.
    fn shrink_layer_holding_least_recent_expert(
        &mut self,
        table_changes: &mut Vec<TableChange>,
    ) -> bool {
        let mut least_recent: Option<(u64, usize)> = None;
        for (layer_index, layer_table) in self.layer_tables.iter().enumerate() {
            for held_expert in &layer_table.held_experts {
                let HeldExpert::UsedAtToken(used_at_token) = *held_expert else {
                    continue;
                };
                if least_recent.is_none_or(|(least_token, _)| used_at_token < least_token) {
                    least_recent = Some((used_at_token, layer_index));
                }
            }
        }
        let Some((_used_at_token, layer_index)) = least_recent else {
            return false;
        };
        let mut dropped_expert_ids = Vec::new();
        for _dropped_slot in 0..self.rules.growth_step_slots.max(1) {
            let Some(expert_id) =
                self.least_recently_used_expert_among(layer_index, &dropped_expert_ids)
            else {
                break;
            };
            dropped_expert_ids.push(expert_id);
        }
        self.compact(layer_index, &dropped_expert_ids, None, table_changes);
        true
    }

    fn least_recently_used_expert_in(&self, layer_index: usize) -> Option<u16> {
        self.least_recently_used_expert_among(layer_index, &[])
    }

    /// The layer's least recently used expert, sparing the current token's
    /// own experts and any already chosen.
    fn least_recently_used_expert_among(
        &self,
        layer_index: usize,
        already_chosen_expert_ids: &[u16],
    ) -> Option<u16> {
        let mut least_recent: Option<(u64, u16)> = None;
        for (expert_id, held_expert) in self.layer_tables[layer_index]
            .held_experts
            .iter()
            .enumerate()
        {
            let used_at_token = match *held_expert {
                HeldExpert::NotHeld => continue,
                HeldExpert::NeverUsed => 0,
                HeldExpert::UsedAtToken(used_at_token) => used_at_token,
            };
            let expert_id = u16::try_from(expert_id).unwrap_or(u16::MAX);
            if used_at_token == self.current_token || already_chosen_expert_ids.contains(&expert_id)
            {
                continue;
            }
            if least_recent.is_none_or(|(least_token, _)| used_at_token < least_token) {
                least_recent = Some((used_at_token, expert_id));
            }
        }
        least_recent.map(|(_used_at_token, expert_id)| expert_id)
    }

    fn has_free_slot(&self, layer_index: usize) -> bool {
        let layer_table = &self.layer_tables[layer_index];
        layer_table.held_expert_count < layer_table.slot_capacity
    }

    fn held(&self, expert: ExpertKey) -> HeldExpert {
        self.layer_tables[expert.layer_index].held_experts[usize::from(expert.expert_id)]
    }

    fn hold(&mut self, expert: ExpertKey) {
        let layer_table = &mut self.layer_tables[expert.layer_index];
        layer_table.held_experts[usize::from(expert.expert_id)] =
            HeldExpert::UsedAtToken(self.current_token);
        layer_table.held_expert_count += 1;
    }

    fn mark_used(&mut self, expert: ExpertKey) {
        self.layer_tables[expert.layer_index].held_experts[usize::from(expert.expert_id)] =
            HeldExpert::UsedAtToken(self.current_token);
    }

    fn release(&mut self, needed: Option<ExpertKey>, released: ExpertKey) {
        let layer_table = &mut self.layer_tables[released.layer_index];
        if layer_table.held_experts[usize::from(released.expert_id)] == HeldExpert::NotHeld {
            return;
        }
        layer_table.held_experts[usize::from(released.expert_id)] = HeldExpert::NotHeld;
        layer_table.held_expert_count -= 1;
        self.record.record_eviction(needed, released);
    }
}

fn slots_as_bytes(slot_count: usize, expert_payload_bytes: u64) -> u64 {
    u64::try_from(slot_count)
        .unwrap_or(u64::MAX)
        .saturating_mul(expert_payload_bytes)
}

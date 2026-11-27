//! Gives back the least-read experts across every layer when the astro expert
//! cache's capacity falls, rather than whole layers (issue #2681).
//!
//! A layer's table is one block of slots, so emptying a slot frees nothing:
//! giving back part of a layer means building a smaller table that holds only
//! the experts the layer keeps. Dropping whole layers instead, least occupied
//! first, left a cache that had grown under a raised limit with some layers
//! complete and the rest empty after the limit came back down. A reply then
//! read every empty layer from the SSD on every token, and on the 35B-A3B at
//! 10.3 GB that needed more memory than the lowered limit left. Keeping each
//! layer's most-read experts leaves the shape a reply settles into when the
//! limit never moved: every layer holding its hot experts.
//!
//! The slots given back are ranked across all layers by how often the
//! reply's tokens routed to each expert (the cache's decode demand), then by
//! this request's prompt demand, then by how often the slot was read. Read
//! counts alone are not comparable across layers: a layer the prompt seated
//! whole carries the prompt's many reads on experts no reply token uses, and
//! ranking by them emptied other layers' hot experts first. Unused slots go
//! first of all, since they hold no expert. Tables are rebuilt one at a
//! time after the layers left with nothing are dropped, so the memory a
//! rebuild needs beside the table it replaces is at most one layer's table.

use engine_core::expert_paging::{ExpertWeightPage, RetainedExpertReclamation};
use runtime_integration::MlxRuntime;

use super::slot_writes::materialize_and_detach_table_weights;
use super::{AstroExpertCache, ExpertSlotTable};
use crate::qwen3_5_text::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;

/// One layer's table as the trim plan sees it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct TableTrimView {
    /// Bytes one slot of this table occupies, held or not.
    pub(super) slot_bytes: u64,
    pub(super) slot_count: usize,
    /// Every held slot and the demand for its expert.
    pub(super) held_slot_demand: Vec<HeldSlotDemand>,
}

/// A held slot and the demand for its expert, most telling first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HeldSlotDemand {
    pub(super) slot: usize,
    /// Reply tokens that routed to the expert, over the model's lifetime.
    pub(super) decode_demand_count: u64,
    /// This request's prompt routes to the expert, weighted by chunk.
    pub(super) prefill_demand_count: u64,
    pub(super) read_count: u64,
}

/// What the trim does to one layer's table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum LayerTrim {
    Unchanged,
    /// Rebuilt to hold only these slots' experts, in slot order.
    KeepSlots(Vec<usize>),
    Drop,
}

/// One slot the plan may give back: an unused slot, or a held one with the
/// demand for its expert. The field order is the release order: unused
/// slots first because they hold nothing, then the least demanded. An unused
/// slot's `slot` is only its ordinal among the table's unused slots: a
/// rebuild drops every unused slot, so which one is never asked.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReleasableSlot {
    is_held: bool,
    decode_demand_count: u64,
    prefill_demand_count: u64,
    read_count: u64,
    layer_index: usize,
    slot: usize,
}

/// Plans which slots to give back so at least `bytes_to_release` are freed,
/// least read first across every layer. A layer whose every held slot is
/// given back is dropped; a layer that loses only some slots, or only unused
/// ones, is rebuilt holding the rest. Asking for more than every table holds
/// drops every table.
pub(super) fn plan_hot_expert_trim(
    table_views: &[Option<TableTrimView>],
    bytes_to_release: u64,
) -> Vec<LayerTrim> {
    let mut layer_trims = vec![LayerTrim::Unchanged; table_views.len()];
    if bytes_to_release == 0 {
        return layer_trims;
    }
    let mut releasable_slots = releasable_slots(table_views);
    releasable_slots.sort_unstable();
    let mut released_slots_by_layer: Vec<Vec<usize>> = vec![Vec::new(); table_views.len()];
    let mut is_layer_changed = vec![false; table_views.len()];
    let mut released_bytes = 0_u64;
    for releasable_slot in releasable_slots {
        if released_bytes >= bytes_to_release {
            break;
        }
        let Some(Some(table_view)) = table_views.get(releasable_slot.layer_index) else {
            continue;
        };
        released_bytes = released_bytes.saturating_add(table_view.slot_bytes);
        is_layer_changed[releasable_slot.layer_index] = true;
        if releasable_slot.is_held {
            released_slots_by_layer[releasable_slot.layer_index].push(releasable_slot.slot);
        }
    }
    for (layer_index, table_view) in table_views.iter().enumerate() {
        let Some(table_view) = table_view else {
            continue;
        };
        if is_layer_changed[layer_index] {
            layer_trims[layer_index] =
                layer_trim_keeping(table_view, &released_slots_by_layer[layer_index]);
        }
    }
    layer_trims
}

fn releasable_slots(table_views: &[Option<TableTrimView>]) -> Vec<ReleasableSlot> {
    let mut releasable_slots = Vec::new();
    for (layer_index, table_view) in table_views.iter().enumerate() {
        let Some(table_view) = table_view else {
            continue;
        };
        let unused_slot_count = table_view
            .slot_count
            .saturating_sub(table_view.held_slot_demand.len());
        for unused_slot_ordinal in 0..unused_slot_count {
            releasable_slots.push(ReleasableSlot {
                is_held: false,
                decode_demand_count: 0,
                prefill_demand_count: 0,
                read_count: 0,
                layer_index,
                slot: unused_slot_ordinal,
            });
        }
        for held_slot_demand in &table_view.held_slot_demand {
            releasable_slots.push(ReleasableSlot {
                is_held: true,
                decode_demand_count: held_slot_demand.decode_demand_count,
                prefill_demand_count: held_slot_demand.prefill_demand_count,
                read_count: held_slot_demand.read_count,
                layer_index,
                slot: held_slot_demand.slot,
            });
        }
    }
    releasable_slots
}

fn layer_trim_keeping(table_view: &TableTrimView, released_slots: &[usize]) -> LayerTrim {
    let mut kept_slots = Vec::new();
    for held_slot_demand in &table_view.held_slot_demand {
        if !released_slots.contains(&held_slot_demand.slot) {
            kept_slots.push(held_slot_demand.slot);
        }
    }
    if kept_slots.is_empty() {
        return LayerTrim::Drop;
    }
    kept_slots.sort_unstable();
    LayerTrim::KeepSlots(kept_slots)
}

impl AstroExpertCache {
    /// Sets how many bytes of experts the cache may keep and gives back what
    /// no longer fits, keeping every layer's most-demanded experts. A table that
    /// cannot be rebuilt, because the rebuild would not fit beside it or MLX
    /// refused it, is dropped instead, and a rebuild that measured larger than
    /// planned is made up by dropping whole tables, so the cache always ends
    /// within its capacity.
    pub fn set_capacity_keeping_hot_experts(
        &mut self,
        runtime: &MlxRuntime,
        capacity_bytes: u64,
    ) -> RetainedExpertReclamation {
        self.capacity_bytes = capacity_bytes;
        let resident_payload_bytes_before = self.resident_payload_bytes;
        let bytes_to_release = resident_payload_bytes_before.saturating_sub(capacity_bytes);
        if bytes_to_release == 0 {
            return RetainedExpertReclamation::default();
        }
        let layer_trims = plan_hot_expert_trim(&self.table_trim_views(), bytes_to_release);
        let mut changed_layer_count = 0_usize;
        // Drops come first: the memory they free is what a rebuild needs
        // beside the table it replaces.
        for (layer_index, layer_trim) in layer_trims.iter().enumerate() {
            if *layer_trim == LayerTrim::Drop && self.remove_layer(layer_index) {
                changed_layer_count += 1;
            }
        }
        for (layer_index, layer_trim) in layer_trims.into_iter().enumerate() {
            let LayerTrim::KeepSlots(kept_slots) = layer_trim else {
                continue;
            };
            if !self.rebuild_layer_table_keeping(runtime, layer_index, &kept_slots) {
                self.remove_layer(layer_index);
            }
            changed_layer_count += 1;
        }
        let whole_table_fallback = self.trim_to_capacity();
        RetainedExpertReclamation {
            released_partial_layer_count: changed_layer_count
                .saturating_add(whole_table_fallback.released_partial_layer_count),
            released_partial_payload_bytes: resident_payload_bytes_before
                .saturating_sub(self.resident_payload_bytes),
            released_complete_layer_count: 0,
            released_complete_payload_bytes: 0,
        }
    }

    fn table_trim_views(&self) -> Vec<Option<TableTrimView>> {
        let mut table_views = Vec::with_capacity(self.tables_by_layer.len());
        for (layer_index, table) in self.tables_by_layer.iter().enumerate() {
            let Some(table) = table else {
                table_views.push(None);
                continue;
            };
            table_views.push(Some(table_trim_view(
                table,
                layer_demand_counts(&self.decode_demand_counts_by_layer, layer_index),
                layer_demand_counts(&self.prefill_demand_counts_by_layer, layer_index),
            )));
        }
        table_views
    }

    /// Replaces a layer's table with one holding only `kept_slots`; false,
    /// with the table untouched, when the rebuild does not fit beside it or
    /// MLX refuses it.
    fn rebuild_layer_table_keeping(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        kept_slots: &[usize],
    ) -> bool {
        let Some(Some(table)) = self.tables_by_layer.get_mut(layer_index) else {
            return false;
        };
        let rebuilt_table_bytes = (table.full_padded_payload_bytes
            / table.slot_count().max(1) as u64)
            .saturating_mul(kept_slots.len() as u64);
        if !rebuild_fits_beside_the_old_table(runtime, rebuilt_table_bytes) {
            return false;
        }
        let kept_weights = match weights_holding_only(runtime, &table.weights, kept_slots) {
            Ok(kept_weights) => kept_weights,
            Err(rebuild_failure) => {
                tracing::warn!(
                    layer_index,
                    %rebuild_failure,
                    "the astro_expert_cache could not rebuild a smaller table and drops it instead"
                );
                return false;
            }
        };
        let replaced_table_bytes = table.full_padded_payload_bytes;
        install_kept_slots(table, kept_weights, kept_slots);
        let kept_table_bytes = table.full_padded_payload_bytes;
        self.resident_payload_bytes = self
            .resident_payload_bytes
            .saturating_sub(replaced_table_bytes)
            .saturating_add(kept_table_bytes);
        true
    }
}

fn layer_demand_counts(demand_counts_by_layer: &[Vec<u64>], layer_index: usize) -> &[u64] {
    demand_counts_by_layer
        .get(layer_index)
        .map_or(&[], Vec::as_slice)
}

fn table_trim_view(
    table: &ExpertSlotTable,
    decode_demand_count_by_expert: &[u64],
    prefill_demand_count_by_expert: &[u64],
) -> TableTrimView {
    let mut held_slot_demand = Vec::with_capacity(table.occupied_slot_count);
    for (slot, expert_id) in table.expert_id_by_slot.iter().enumerate() {
        let Some(expert_id) = *expert_id else {
            continue;
        };
        held_slot_demand.push(HeldSlotDemand {
            slot,
            decode_demand_count: decode_demand_count_by_expert
                .get(expert_id)
                .copied()
                .unwrap_or(0),
            prefill_demand_count: prefill_demand_count_by_expert
                .get(expert_id)
                .copied()
                .unwrap_or(0),
            read_count: table.read_count_by_slot.get(slot).copied().unwrap_or(0),
        });
    }
    TableTrimView {
        slot_bytes: table.full_padded_payload_bytes / table.slot_count().max(1) as u64,
        slot_count: table.slot_count(),
        held_slot_demand,
    }
}

/// The trim runs under the limit in force when the capacity falls: a live
/// lower takes it before the smaller limit is installed. Both tables exist
/// until the old one is released, so the rebuild has to fit beside it.
fn rebuild_fits_beside_the_old_table(runtime: &MlxRuntime, rebuilt_table_bytes: u64) -> bool {
    let Ok(memory_snapshot) = runtime.memory_snapshot() else {
        return false;
    };
    rebuild_fits_within(
        memory_snapshot.active_memory_bytes() as u64,
        rebuilt_table_bytes,
        runtime.memory_limits().active_memory_limit_bytes() as u64,
    )
}

/// Pure: whether a rebuilt table of `rebuilt_table_bytes`, allocated while
/// `active_memory_bytes` are in use, stays within the installed limit.
pub(super) fn rebuild_fits_within(
    active_memory_bytes: u64,
    rebuilt_table_bytes: u64,
    active_memory_limit_bytes: u64,
) -> bool {
    // A sum past the range would saturate to the largest limit and pass.
    active_memory_bytes
        .checked_add(rebuilt_table_bytes)
        .is_some_and(|needed_memory_bytes| needed_memory_bytes <= active_memory_limit_bytes)
}

/// The kept experts gathered into a table of their own, evaluated and
/// detached so it no longer pins the table it was gathered from.
fn weights_holding_only(
    runtime: &MlxRuntime,
    table_weights: &Qwen3_5PagedExpertWeights,
    kept_slots: &[usize],
) -> Result<Qwen3_5PagedExpertWeights, String> {
    let kept_weights = table_weights
        .select_experts(runtime, kept_slots)
        .map_err(|paging_error| paging_error.to_string())?;
    materialize_and_detach_table_weights(runtime, &kept_weights)
        .map_err(|runtime_error| runtime_error.to_string())?;
    Ok(kept_weights)
}

/// Moves the kept slots' experts and their read history into slots
/// `0..kept_slots.len()` of the rebuilt table.
fn install_kept_slots(
    table: &mut ExpertSlotTable,
    kept_weights: Qwen3_5PagedExpertWeights,
    kept_slots: &[usize],
) {
    let mut expert_id_by_slot = Vec::with_capacity(kept_slots.len());
    let mut read_count_by_slot = Vec::with_capacity(kept_slots.len());
    let mut decode_policy_by_slot = Vec::with_capacity(kept_slots.len());
    table.slot_by_expert_id.clear();
    for (rebuilt_slot, &kept_slot) in kept_slots.iter().enumerate() {
        let expert_id = table.expert_id_by_slot[kept_slot];
        if let Some(expert_id) = expert_id {
            table.slot_by_expert_id.insert(expert_id, rebuilt_slot);
        }
        expert_id_by_slot.push(expert_id);
        read_count_by_slot.push(
            table
                .read_count_by_slot
                .get(kept_slot)
                .copied()
                .unwrap_or(0),
        );
        decode_policy_by_slot.push(
            table
                .decode_policy_by_slot
                .get(kept_slot)
                .copied()
                .unwrap_or_default(),
        );
    }
    table.full_padded_payload_bytes = kept_weights.resident_payload_byte_count();
    table.weights = kept_weights;
    table.expert_id_by_slot = expert_id_by_slot;
    table.read_count_by_slot = read_count_by_slot;
    table.decode_policy_by_slot = decode_policy_by_slot;
    table.occupied_slot_count = kept_slots.len();
    table.payload_bytes = table
        .per_expert_payload_bytes
        .saturating_mul(kept_slots.len() as u64);
}

#[cfg(test)]
#[path = "hot_expert_trim_tests.rs"]
mod hot_expert_trim_tests;

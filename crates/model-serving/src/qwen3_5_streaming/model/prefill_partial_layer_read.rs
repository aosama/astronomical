//! Decisions for a prefill chunk on a layer the warm table only partly holds.
//!
//! A funded completion read is worth its bytes only when the ceiling seats the
//! remainder and later chunks become table hits. Once a ceiling refuses that
//! insert, repeating the remainder read on every chunk re-reads the same
//! unseated experts and, on the fallback, experts already in RAM. These
//! functions keep that refusal sticky until the ceiling rises.

/// Which experts one prefill chunk should load from storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrefillStorageReadScope {
    /// Read every expert the table does not hold, so one insert can finish the layer.
    EntireUnseatedRemainder,
    /// Read only the routed experts the table does not hold.
    RoutedMissesOnly,
}

/// Layers whose completion read the ceiling already refused during this load.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct PrefillLayerCompletionRefusals {
    refused_by_layer: Vec<bool>,
}

/// Retained and missing halves of one routed union, in ascending expert id order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PrefillRouteSplit {
    pub(crate) retained_expert_ids: Vec<usize>,
    pub(crate) missing_expert_ids: Vec<usize>,
}

/// Expert order of a page built as retained rows followed by streamed misses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AssembledPrefillExpertOrder {
    pub(crate) expert_ids: Vec<usize>,
    pub(crate) page_slot_by_global_expert_id: Vec<u32>,
}

impl PrefillLayerCompletionRefusals {
    /// Whether this layer's completion read has already been refused.
    #[must_use]
    pub(crate) fn was_refused(&self, layer_index: usize) -> bool {
        self.refused_by_layer
            .get(layer_index)
            .copied()
            .unwrap_or(false)
    }

    /// Remembers that finishing this layer did not fit the ceiling.
    pub(crate) fn remember(&mut self, layer_index: usize) {
        if self.refused_by_layer.len() <= layer_index {
            self.refused_by_layer
                .resize(layer_index.saturating_add(1), false);
        }
        self.refused_by_layer[layer_index] = true;
    }

    /// Forgets every refusal. A higher ceiling can seat a layer the old one could not.
    pub(crate) fn clear(&mut self) {
        self.refused_by_layer.clear();
    }
}

/// Chooses the storage read for one partial layer.
///
/// A refusal wins over a still-funded plan: the plan does not know the insert
/// already failed, and another full remainder read would not seat anything new.
#[must_use]
pub(crate) fn prefill_storage_read_scope(
    completion_is_funded: bool,
    layer_completion_was_refused: bool,
) -> PrefillStorageReadScope {
    if completion_is_funded && !layer_completion_was_refused {
        PrefillStorageReadScope::EntireUnseatedRemainder
    } else {
        PrefillStorageReadScope::RoutedMissesOnly
    }
}

/// The expert ids to load for `read_scope`.
///
/// An empty remainder falls back to the routed misses so a chunk never issues
/// an empty storage read when the route itself still has holes.
#[must_use]
pub(crate) fn prefill_read_expert_ids(
    read_scope: PrefillStorageReadScope,
    missing_route_expert_ids: &[usize],
    non_resident_expert_ids: Option<&[usize]>,
) -> Vec<usize> {
    match read_scope {
        PrefillStorageReadScope::RoutedMissesOnly => missing_route_expert_ids.to_vec(),
        PrefillStorageReadScope::EntireUnseatedRemainder => match non_resident_expert_ids {
            Some(non_resident_expert_ids) if !non_resident_expert_ids.is_empty() => {
                non_resident_expert_ids.to_vec()
            }
            _ => missing_route_expert_ids.to_vec(),
        },
    }
}

/// Whether this read should suppress later remainder reads on the same layer.
#[must_use]
pub(crate) fn should_remember_prefill_completion_refusal(
    read_scope: PrefillStorageReadScope,
    layer_is_complete_after_insert: bool,
) -> bool {
    read_scope == PrefillStorageReadScope::EntireUnseatedRemainder
        && !layer_is_complete_after_insert
}

/// A raised ceiling is the only change that can make a refused layer seatable again.
#[must_use]
pub(crate) fn should_clear_prefill_completion_refusals(
    previous_ceiling_bytes: u64,
    requested_ceiling_bytes: u64,
) -> bool {
    requested_ceiling_bytes > previous_ceiling_bytes
}

/// Splits a routed union into experts the table holds and experts it does not.
///
/// Ids present only on the missing side are dropped: a completion read may
/// have loaded unseated experts this chunk never routed, and the forward page
/// must stay the routed union or prefill decodes experts it will not multiply.
#[must_use]
pub(crate) fn split_prefill_route(
    routed_expert_ids: &[usize],
    missing_route_expert_ids: &[usize],
) -> PrefillRouteSplit {
    let mut routed_expert_ids = routed_expert_ids.to_vec();
    routed_expert_ids.sort_unstable();
    routed_expert_ids.dedup();
    let mut missing_route_expert_ids = missing_route_expert_ids.to_vec();
    missing_route_expert_ids.sort_unstable();
    missing_route_expert_ids.dedup();
    let mut retained_expert_ids = Vec::new();
    let mut missing_expert_ids = Vec::new();
    let mut routed_index = 0;
    let mut missing_index = 0;
    while routed_index < routed_expert_ids.len() {
        let routed_expert_id = routed_expert_ids[routed_index];
        if missing_index >= missing_route_expert_ids.len() {
            retained_expert_ids.push(routed_expert_id);
            routed_index = routed_index.saturating_add(1);
            continue;
        }
        let missing_expert_id = missing_route_expert_ids[missing_index];
        if missing_expert_id < routed_expert_id {
            missing_index = missing_index.saturating_add(1);
            continue;
        }
        if missing_expert_id == routed_expert_id {
            missing_expert_ids.push(routed_expert_id);
            missing_index = missing_index.saturating_add(1);
        } else {
            retained_expert_ids.push(routed_expert_id);
        }
        routed_index = routed_index.saturating_add(1);
    }
    PrefillRouteSplit {
        retained_expert_ids,
        missing_expert_ids,
    }
}

/// Slot map for a page whose experts are the retained ids followed by the misses.
///
/// `None` when an id is outside the layer. Unassigned slots stay at `u32::MAX`
/// so a gather of an expert that was not placed cannot silently read slot 0.
#[must_use]
pub(crate) fn assembled_prefill_expert_order(
    retained_expert_ids: &[usize],
    missing_expert_ids: &[usize],
    expert_capacity: usize,
) -> Option<AssembledPrefillExpertOrder> {
    let mut expert_ids = Vec::with_capacity(
        retained_expert_ids
            .len()
            .saturating_add(missing_expert_ids.len()),
    );
    let mut retained_index = 0;
    while retained_index < retained_expert_ids.len() {
        expert_ids.push(retained_expert_ids[retained_index]);
        retained_index = retained_index.saturating_add(1);
    }
    let mut missing_index = 0;
    while missing_index < missing_expert_ids.len() {
        expert_ids.push(missing_expert_ids[missing_index]);
        missing_index = missing_index.saturating_add(1);
    }
    let mut page_slot_by_global_expert_id = vec![u32::MAX; expert_capacity];
    let mut page_slot = 0;
    while page_slot < expert_ids.len() {
        let expert_id = expert_ids[page_slot];
        if expert_id >= expert_capacity {
            return None;
        }
        let Ok(page_slot_index) = u32::try_from(page_slot) else {
            return None;
        };
        page_slot_by_global_expert_id[expert_id] = page_slot_index;
        page_slot = page_slot.saturating_add(1);
    }
    Some(AssembledPrefillExpertOrder {
        expert_ids,
        page_slot_by_global_expert_id,
    })
}

/// Page slots of `expert_ids` inside a manifest, in that id order.
///
/// `None` when any id is absent or still the unassigned sentinel.
#[must_use]
pub(crate) fn page_slots_for_expert_ids(
    page_slot_by_global_expert_id: &[u32],
    expert_ids: &[usize],
) -> Option<Vec<usize>> {
    let mut page_slots = Vec::with_capacity(expert_ids.len());
    let mut expert_index = 0;
    while expert_index < expert_ids.len() {
        let expert_id = expert_ids[expert_index];
        let page_slot = page_slot_by_global_expert_id.get(expert_id)?;
        if *page_slot == u32::MAX {
            return None;
        }
        // A slot that does not fit `usize` is the same as an absent expert:
        // handing the gather `usize::MAX` would read a real slot or fail later
        // with a less specific bounds error.
        let Ok(page_slot_index) = usize::try_from(*page_slot) else {
            return None;
        };
        page_slots.push(page_slot_index);
        expert_index = expert_index.saturating_add(1);
    }
    Some(page_slots)
}

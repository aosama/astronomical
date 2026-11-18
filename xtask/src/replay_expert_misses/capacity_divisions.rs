//! Ways the replay lab divides one `astro_expert_cache` capacity between
//! layers: each turns a recorded case into the same case with per-layer slot
//! limits.

use std::collections::BTreeSet;

use astronomical_memory_policy::astro_expert_cache_model::{ReplayCase, SavedReplayCase};

/// The way today's engine divides it: layers held whole when the reply began
/// stay whole, layers that held nothing at either end of the reply keep
/// nothing (the reply's plan left them absent), and every other layer gets an
/// equal table from what the capacity has left, its warm slots.
pub(crate) fn divided_like_todays_engine(saved_replay_case: &SavedReplayCase) -> ReplayCase {
    let replay_case = &saved_replay_case.replay_case;
    let experts_per_layer = usize::from(replay_case.routed_reply.experts_per_layer());
    let layer_count = replay_case.held_expert_ids_by_layer_at_start.len();
    let mut layer_kinds = Vec::with_capacity(layer_count);
    for (layer_index, held_expert_ids_at_start) in replay_case
        .held_expert_ids_by_layer_at_start
        .iter()
        .enumerate()
    {
        let held_at_start = held_expert_ids_at_start.len();
        let held_at_end = saved_replay_case
            .held_expert_ids_by_layer_at_end
            .get(layer_index)
            .map_or(1, Vec::len);
        layer_kinds.push(if held_at_start >= experts_per_layer {
            LayerKind::Whole
        } else if held_at_start == 0 && held_at_end == 0 {
            LayerKind::Absent
        } else {
            LayerKind::Hot
        });
    }
    let whole_layer_count = count_layers(&layer_kinds, LayerKind::Whole);
    let hot_layer_count = count_layers(&layer_kinds, LayerKind::Hot);
    // The engine's own warm slot count when the recording kept it; otherwise
    // what the capacity leaves for each layer not held whole.
    let warm_slot_count = saved_replay_case.warm_slot_count.unwrap_or_else(|| {
        capacity_in_experts(replay_case)
            .saturating_sub(whole_layer_count * experts_per_layer)
            .checked_div(hot_layer_count)
            .unwrap_or(0)
            .min(experts_per_layer)
    });
    let mut layer_slot_limits = Vec::with_capacity(layer_count);
    for (layer_index, layer_kind) in layer_kinds.into_iter().enumerate() {
        // A table's size is what it held after the reply when the recording
        // kept that: the engine's tables stay at the size they were given.
        let held_at_end = saved_replay_case
            .held_expert_ids_by_layer_at_end
            .get(layer_index)
            .map_or(0, Vec::len);
        layer_slot_limits.push(match layer_kind {
            LayerKind::Whole => experts_per_layer,
            LayerKind::Hot if held_at_end > 0 => held_at_end,
            LayerKind::Hot => warm_slot_count,
            LayerKind::Absent => 0,
        });
    }
    with_layer_slot_limits(replay_case, layer_slot_limits)
}

/// Layers held whole when the reply began stay whole, so their routes never
/// stop; the rest share what is left, each free to take any part of it.
pub(crate) fn shared_pinning_whole_layers(replay_case: &ReplayCase) -> ReplayCase {
    let experts_per_layer = usize::from(replay_case.routed_reply.experts_per_layer());
    let mut pinned_layer_indexes = Vec::new();
    for (layer_index, held_expert_ids) in replay_case
        .held_expert_ids_by_layer_at_start
        .iter()
        .enumerate()
    {
        if held_expert_ids.len() >= experts_per_layer {
            pinned_layer_indexes.push(layer_index);
        }
    }
    ReplayCase {
        pinned_layer_indexes,
        ..replay_case.clone()
    }
}

/// Each layer's table sized to its share of the distinct experts the reply
/// used: an oracle, since it reads the whole reply first.
pub(crate) fn divided_by_demand(replay_case: &ReplayCase) -> ReplayCase {
    let routed_reply = &replay_case.routed_reply;
    let mut distinct_experts_by_layer = vec![BTreeSet::new(); routed_reply.layer_count()];
    for token_index in 0..routed_reply.token_count() {
        for (layer_index, distinct_experts) in distinct_experts_by_layer.iter_mut().enumerate() {
            distinct_experts.extend(
                routed_reply
                    .layer_route(token_index, layer_index)
                    .iter()
                    .copied(),
            );
        }
    }
    let total_distinct: usize = distinct_experts_by_layer.iter().map(BTreeSet::len).sum();
    let capacity_experts = capacity_in_experts(replay_case);
    let mut layer_slot_limits = Vec::with_capacity(distinct_experts_by_layer.len());
    for distinct_experts in &distinct_experts_by_layer {
        layer_slot_limits.push(
            (distinct_experts.len() * capacity_experts)
                .checked_div(total_distinct)
                .unwrap_or(0),
        );
    }
    with_layer_slot_limits(replay_case, layer_slot_limits)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LayerKind {
    Whole,
    Hot,
    Absent,
}

fn count_layers(layer_kinds: &[LayerKind], counted_kind: LayerKind) -> usize {
    let mut layer_count = 0;
    for &layer_kind in layer_kinds {
        if layer_kind == counted_kind {
            layer_count += 1;
        }
    }
    layer_count
}

fn capacity_in_experts(replay_case: &ReplayCase) -> usize {
    let expert_bytes = replay_case
        .expert_payload_bytes_by_layer
        .first()
        .copied()
        .unwrap_or(1)
        .max(1);
    usize::try_from(replay_case.capacity_bytes / expert_bytes).unwrap_or(0)
}

fn with_layer_slot_limits(replay_case: &ReplayCase, layer_slot_limits: Vec<usize>) -> ReplayCase {
    ReplayCase {
        layer_slot_limits: Some(layer_slot_limits),
        ..replay_case.clone()
    }
}

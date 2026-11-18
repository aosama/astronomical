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

/// The same case starting with nothing held: what a rule does from a cold
/// start.
pub(crate) fn starting_with_nothing(replay_case: &ReplayCase) -> ReplayCase {
    ReplayCase {
        held_expert_ids_by_layer_at_start: vec![Vec::new(); replay_case.routed_reply.layer_count()],
        ..replay_case.clone()
    }
}

/// The same case starting with the experts the reply uses most, as many as
/// the capacity holds: an oracle for what a perfect warm start is worth.
pub(crate) fn starting_with_the_most_used(replay_case: &ReplayCase) -> ReplayCase {
    let routed_reply = &replay_case.routed_reply;
    let mut use_count_by_expert = std::collections::BTreeMap::new();
    for token_index in 0..routed_reply.token_count() {
        for layer_index in 0..routed_reply.layer_count() {
            for &expert_id in routed_reply.layer_route(token_index, layer_index) {
                *use_count_by_expert
                    .entry((layer_index, expert_id))
                    .or_insert(0_u64) += 1;
            }
        }
    }
    let mut experts_by_use: Vec<((usize, u16), u64)> = use_count_by_expert.into_iter().collect();
    experts_by_use.sort_by_key(|&(expert, use_count)| (std::cmp::Reverse(use_count), expert));
    let mut held_expert_ids_by_layer = vec![Vec::new(); routed_reply.layer_count()];
    for ((layer_index, expert_id), _) in experts_by_use
        .into_iter()
        .take(capacity_in_experts(replay_case))
    {
        held_expert_ids_by_layer[layer_index].push(expert_id);
    }
    ReplayCase {
        held_expert_ids_by_layer_at_start: held_expert_ids_by_layer,
        ..replay_case.clone()
    }
}

/// Every layer an equal table and none held whole: the simplest division.
pub(crate) fn divided_equally(replay_case: &ReplayCase) -> ReplayCase {
    let layer_count = replay_case.routed_reply.layer_count();
    let equal_slot_count = capacity_in_experts(replay_case)
        .checked_div(layer_count)
        .unwrap_or(0)
        .min(usize::from(replay_case.routed_reply.experts_per_layer()));
    with_layer_slot_limits(replay_case, vec![equal_slot_count; layer_count])
}

/// The same case starting with the experts the prompt routed to most, as
/// many as the capacity holds, with `reply_demand_weight` times the earlier
/// replies' demand added: a warm start the engine can make at the reply's
/// start, from evidence it already has.
pub(crate) fn starting_from_demand(
    saved_replay_case: &SavedReplayCase,
    reply_demand_weight: u64,
) -> ReplayCase {
    let replay_case = &saved_replay_case.replay_case;
    let mut experts_by_demand = Vec::new();
    for (layer_index, prompt_layer_demand) in
        saved_replay_case.prompt_demand_by_layer.iter().enumerate()
    {
        for (expert_id, &prompt_demand) in prompt_layer_demand.iter().enumerate() {
            let reply_demand = saved_replay_case
                .reply_demand_by_layer
                .get(layer_index)
                .and_then(|reply_layer_demand| reply_layer_demand.get(expert_id))
                .copied()
                .unwrap_or(0);
            let demand = prompt_demand + reply_demand_weight * reply_demand;
            if demand > 0 {
                experts_by_demand.push((std::cmp::Reverse(demand), layer_index, expert_id));
            }
        }
    }
    experts_by_demand.sort_unstable();
    let mut held_expert_ids_by_layer = vec![Vec::new(); replay_case.routed_reply.layer_count()];
    for (_, layer_index, expert_id) in experts_by_demand
        .into_iter()
        .take(capacity_in_experts(replay_case))
    {
        if let (Some(held_expert_ids), Ok(expert_id)) = (
            held_expert_ids_by_layer.get_mut(layer_index),
            u16::try_from(expert_id),
        ) {
            held_expert_ids.push(expert_id);
        }
    }
    ReplayCase {
        held_expert_ids_by_layer_at_start: held_expert_ids_by_layer,
        ..replay_case.clone()
    }
}

/// `whole_layer_count` of the layers held whole when the reply began stay
/// whole (pinned, so their routes never stop), the ones the prompt demanded
/// most; every other layer gets an equal table from what is left. This is
/// the trade between misses and route stops the engine has to make.
pub(crate) fn whole_layers_then_equal_tables(
    saved_replay_case: &SavedReplayCase,
    whole_layer_count: usize,
) -> ReplayCase {
    let replay_case = &saved_replay_case.replay_case;
    let experts_per_layer = usize::from(replay_case.routed_reply.experts_per_layer());
    let mut whole_candidates = Vec::new();
    for (layer_index, held_expert_ids) in replay_case
        .held_expert_ids_by_layer_at_start
        .iter()
        .enumerate()
    {
        if held_expert_ids.len() >= experts_per_layer {
            let prompt_demand: u64 = saved_replay_case
                .prompt_demand_by_layer
                .get(layer_index)
                .map_or(0, |layer_demand| layer_demand.iter().sum());
            whole_candidates.push((std::cmp::Reverse(prompt_demand), layer_index));
        }
    }
    whole_candidates.sort_unstable();
    let pinned_layer_indexes: Vec<usize> = whole_candidates
        .into_iter()
        .take(whole_layer_count)
        .map(|(_, layer_index)| layer_index)
        .collect();
    let layer_count = replay_case.routed_reply.layer_count();
    let table_layer_count = layer_count - pinned_layer_indexes.len();
    let table_slot_count = capacity_in_experts(replay_case)
        .saturating_sub(pinned_layer_indexes.len() * experts_per_layer)
        .checked_div(table_layer_count)
        .unwrap_or(0)
        .min(experts_per_layer);
    let mut layer_slot_limits = vec![table_slot_count; layer_count];
    let mut held_expert_ids_by_layer = vec![Vec::new(); layer_count];
    for &layer_index in &pinned_layer_indexes {
        layer_slot_limits[layer_index] = experts_per_layer;
        held_expert_ids_by_layer[layer_index]
            .clone_from(&replay_case.held_expert_ids_by_layer_at_start[layer_index]);
    }
    ReplayCase {
        held_expert_ids_by_layer_at_start: held_expert_ids_by_layer,
        layer_slot_limits: Some(layer_slot_limits),
        pinned_layer_indexes,
        ..replay_case.clone()
    }
}

/// `whole_layer_count` layers kept whole as in `whole_layers_then_equal_tables`,
/// and every other layer's table sized in proportion to how many of its
/// experts cover nine tenths of its prompt demand: tables sized at the
/// reply's start from evidence the engine already has.
pub(crate) fn whole_layers_then_tables_by_prompt_demand(
    saved_replay_case: &SavedReplayCase,
    whole_layer_count: usize,
) -> ReplayCase {
    let equal_case = whole_layers_then_equal_tables(saved_replay_case, whole_layer_count);
    let experts_per_layer = usize::from(equal_case.routed_reply.experts_per_layer());
    let layer_count = equal_case.routed_reply.layer_count();
    let mut demand_breadth_by_layer = vec![0_usize; layer_count];
    for (layer_index, demand_breadth) in demand_breadth_by_layer.iter_mut().enumerate() {
        if equal_case.pinned_layer_indexes.contains(&layer_index) {
            continue;
        }
        *demand_breadth = saved_replay_case
            .prompt_demand_by_layer
            .get(layer_index)
            .map_or(experts_per_layer, |layer_demand| {
                experts_covering_most_demand(layer_demand)
            })
            .max(1);
    }
    let table_experts = capacity_in_experts(&equal_case)
        .saturating_sub(equal_case.pinned_layer_indexes.len() * experts_per_layer);
    let total_breadth: usize = demand_breadth_by_layer.iter().sum();
    let mut layer_slot_limits = vec![experts_per_layer; layer_count];
    for (layer_index, &demand_breadth) in demand_breadth_by_layer.iter().enumerate() {
        if demand_breadth > 0 {
            layer_slot_limits[layer_index] = (table_experts * demand_breadth)
                .checked_div(total_breadth)
                .unwrap_or(0)
                .min(experts_per_layer);
        }
    }
    ReplayCase {
        layer_slot_limits: Some(layer_slot_limits),
        ..equal_case
    }
}

/// How many of a layer's experts, most demanded first, cover nine tenths of
/// its demand.
fn experts_covering_most_demand(layer_demand: &[u64]) -> usize {
    let mut sorted_demand = layer_demand.to_vec();
    sorted_demand.sort_unstable_by(|left, right| right.cmp(left));
    let total_demand: u64 = sorted_demand.iter().sum();
    let mut covered_demand = 0_u64;
    for (covering_count, demand) in sorted_demand.into_iter().enumerate() {
        if covered_demand * 10 >= total_demand * 9 {
            return covering_count;
        }
        covered_demand += demand;
    }
    layer_demand.len()
}

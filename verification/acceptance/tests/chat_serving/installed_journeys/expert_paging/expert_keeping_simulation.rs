//! Replays a reply's expert routes against different ways of keeping experts
//! in memory, to say how many bytes each way would still read from storage.
//!
//! A model that does not fit keeps some experts and reads the rest as tokens
//! route to them. How fast it decodes is then mostly how many bytes a token
//! has to read, and that depends on which experts were kept. Serving one
//! reply answers that for one way of keeping only. A recorded reply (every
//! token's experts in every layer) can be replayed against any number of
//! ways in milliseconds, with no model and no graphics processor, so the
//! ways can be compared on exactly the same tokens under exactly the same
//! memory.
//!
//! Every way is given the same byte budget and is scored only on the tokens
//! after `first_scored_token_index`; the tokens before it fill its memory and
//! its counts, as a reply's earlier tokens do in serving.

use std::collections::{BTreeSet, HashMap};

/// One reply: for each token, for each layer, the experts it routed to.
pub(super) struct ReplyTrace {
    pub(super) token_routes: Vec<Vec<Vec<u16>>>,
    /// Bytes one expert of each layer occupies in memory.
    pub(super) expert_bytes_by_layer: Vec<u64>,
    pub(super) experts_per_layer: usize,
}

/// What a way of keeping experts did over the scored tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct KeepingOutcome {
    pub(super) scored_token_count: u64,
    pub(super) assignment_count: u64,
    pub(super) missed_assignment_count: u64,
    pub(super) missed_bytes: u64,
}

impl KeepingOutcome {
    /// Share of assignments found in memory, 0 to 100.
    pub(super) fn found_in_memory_percent(&self) -> f64 {
        if self.assignment_count == 0 {
            return 0.0;
        }
        let found_assignment_count = self.assignment_count - self.missed_assignment_count;
        found_assignment_count as f64 * 100.0 / self.assignment_count as f64
    }

    /// Bytes a scored token read from storage on average.
    pub(super) fn missed_bytes_per_token(&self) -> f64 {
        if self.scored_token_count == 0 {
            return 0.0;
        }
        self.missed_bytes as f64 / self.scored_token_count as f64
    }

    fn record(&mut self, expert_bytes: u64, was_in_memory: bool) {
        self.assignment_count += 1;
        if !was_in_memory {
            self.missed_assignment_count += 1;
            self.missed_bytes += expert_bytes;
        }
    }
}

/// An expert, named by its layer and its index in that layer.
type ExpertKey = (usize, u16);

/// Experts in memory under one byte budget, each with a rank; the lowest
/// rank is given back first when the budget is exceeded. Ranking by the time
/// of last use keeps the most recently used; ranking by use count first
/// keeps the most used.
struct RankedExpertMemory {
    byte_budget: u64,
    held_bytes: u64,
    rank_by_expert: HashMap<ExpertKey, (u64, u64)>,
    experts_by_rank: BTreeSet<((u64, u64), ExpertKey)>,
}

impl RankedExpertMemory {
    fn new(byte_budget: u64) -> Self {
        Self {
            byte_budget,
            held_bytes: 0,
            rank_by_expert: HashMap::new(),
            experts_by_rank: BTreeSet::new(),
        }
    }

    fn holds(&self, expert: ExpertKey) -> bool {
        self.rank_by_expert.contains_key(&expert)
    }

    /// Puts `expert` in memory at `rank`, or moves it there, then gives back
    /// the lowest-ranked experts until the budget holds. An expert larger
    /// than the whole budget is never kept.
    fn keep(&mut self, expert: ExpertKey, rank: (u64, u64), expert_bytes_by_layer: &[u64]) {
        if let Some(previous_rank) = self.rank_by_expert.insert(expert, rank) {
            self.experts_by_rank.remove(&(previous_rank, expert));
        } else {
            self.held_bytes += expert_bytes_by_layer[expert.0];
        }
        self.experts_by_rank.insert((rank, expert));
        while self.held_bytes > self.byte_budget {
            let Some((lowest_rank, lowest_expert)) = self.experts_by_rank.pop_first() else {
                return;
            };
            debug_assert_eq!(self.rank_by_expert.get(&lowest_expert), Some(&lowest_rank));
            self.rank_by_expert.remove(&lowest_expert);
            self.held_bytes -= expert_bytes_by_layer[lowest_expert.0];
        }
    }
}

/// Keeps the experts used most recently, across all layers, under one byte
/// budget: no layer is held whole and none has a fixed share.
pub(super) fn keep_most_recently_used(
    trace: &ReplyTrace,
    byte_budget: u64,
    first_scored_token_index: usize,
) -> KeepingOutcome {
    let mut memory = RankedExpertMemory::new(byte_budget);
    let mut outcome = KeepingOutcome::default();
    let mut use_sequence_number = 0_u64;
    for (token_index, token_route) in trace.token_routes.iter().enumerate() {
        let is_scored = token_index >= first_scored_token_index;
        outcome.scored_token_count += u64::from(is_scored);
        for (layer_index, layer_expert_ids) in token_route.iter().enumerate() {
            for &expert_id in layer_expert_ids {
                let expert = (layer_index, expert_id);
                if is_scored {
                    outcome.record(
                        trace.expert_bytes_by_layer[layer_index],
                        memory.holds(expert),
                    );
                }
                use_sequence_number += 1;
                memory.keep(
                    expert,
                    (use_sequence_number, 0),
                    &trace.expert_bytes_by_layer,
                );
            }
        }
    }
    outcome
}

/// Keeps the experts used most often so far, across all layers, under one
/// byte budget; among equally used experts the most recently used stays.
pub(super) fn keep_most_used_so_far(
    trace: &ReplyTrace,
    byte_budget: u64,
    first_scored_token_index: usize,
) -> KeepingOutcome {
    let mut memory = RankedExpertMemory::new(byte_budget);
    let mut use_count_by_expert: HashMap<ExpertKey, u64> = HashMap::new();
    let mut outcome = KeepingOutcome::default();
    let mut use_sequence_number = 0_u64;
    for (token_index, token_route) in trace.token_routes.iter().enumerate() {
        let is_scored = token_index >= first_scored_token_index;
        outcome.scored_token_count += u64::from(is_scored);
        for (layer_index, layer_expert_ids) in token_route.iter().enumerate() {
            for &expert_id in layer_expert_ids {
                let expert = (layer_index, expert_id);
                if is_scored {
                    outcome.record(
                        trace.expert_bytes_by_layer[layer_index],
                        memory.holds(expert),
                    );
                }
                use_sequence_number += 1;
                let use_count = use_count_by_expert.entry(expert).or_insert(0);
                *use_count += 1;
                memory.keep(
                    expert,
                    (*use_count, use_sequence_number),
                    &trace.expert_bytes_by_layer,
                );
            }
        }
    }
    outcome
}

/// How often each expert was used over a range of tokens.
fn use_counts(trace: &ReplyTrace, token_range: std::ops::Range<usize>) -> HashMap<ExpertKey, u64> {
    let mut use_count_by_expert: HashMap<ExpertKey, u64> = HashMap::new();
    for token_route in &trace.token_routes[token_range] {
        for (layer_index, layer_expert_ids) in token_route.iter().enumerate() {
            for &expert_id in layer_expert_ids {
                *use_count_by_expert
                    .entry((layer_index, expert_id))
                    .or_insert(0) += 1;
            }
        }
    }
    use_count_by_expert
}

/// Keeps one fixed set for the whole reply: the experts used most over
/// `learning_token_range`, filled until the byte budget is spent, and never
/// changed afterwards. With the range before the scored tokens this is a
/// usage list learned ahead of time; with the scored tokens themselves it is
/// the best any fixed set could have done, known only in hindsight.
pub(super) fn keep_a_fixed_set_of_the_most_used(
    trace: &ReplyTrace,
    byte_budget: u64,
    learning_token_range: std::ops::Range<usize>,
    first_scored_token_index: usize,
) -> KeepingOutcome {
    let mut experts_by_use: Vec<(u64, ExpertKey)> = Vec::new();
    for (expert, use_count) in use_counts(trace, learning_token_range) {
        experts_by_use.push((use_count, expert));
    }
    // Most used first; ties broken by layer and index so a replay repeats.
    experts_by_use.sort_by(|first, second| second.0.cmp(&first.0).then(first.1.cmp(&second.1)));
    let mut kept_experts: std::collections::HashSet<ExpertKey> = std::collections::HashSet::new();
    let mut held_bytes = 0_u64;
    for (_, expert) in experts_by_use {
        let expert_bytes = trace.expert_bytes_by_layer[expert.0];
        if held_bytes + expert_bytes > byte_budget {
            continue;
        }
        held_bytes += expert_bytes;
        kept_experts.insert(expert);
    }
    let mut outcome = KeepingOutcome::default();
    for token_route in &trace.token_routes[first_scored_token_index..] {
        outcome.scored_token_count += 1;
        for (layer_index, layer_expert_ids) in token_route.iter().enumerate() {
            for &expert_id in layer_expert_ids {
                outcome.record(
                    trace.expert_bytes_by_layer[layer_index],
                    kept_experts.contains(&(layer_index, expert_id)),
                );
            }
        }
    }
    outcome
}

/// Table slots each layer not held whole gets when `whole_layer_indexes` are
/// held whole and the rest of the budget is split evenly, in experts, over
/// the other layers. Zero when the whole layers alone exceed the budget.
pub(super) fn table_slots_beside_whole_layers(
    trace: &ReplyTrace,
    byte_budget: u64,
    whole_layer_indexes: &[usize],
) -> usize {
    let mut whole_layer_bytes = 0_u64;
    let mut one_slot_on_every_other_layer_bytes = 0_u64;
    for (layer_index, expert_bytes) in trace.expert_bytes_by_layer.iter().enumerate() {
        if whole_layer_indexes.contains(&layer_index) {
            whole_layer_bytes += expert_bytes * trace.experts_per_layer as u64;
        } else {
            one_slot_on_every_other_layer_bytes += expert_bytes;
        }
    }
    if whole_layer_bytes > byte_budget || one_slot_on_every_other_layer_bytes == 0 {
        return 0;
    }
    let affordable_slot_count =
        (byte_budget - whole_layer_bytes) / one_slot_on_every_other_layer_bytes;
    (affordable_slot_count as usize).min(trace.experts_per_layer)
}

/// The engine's present shape: some layers held whole, and on every other
/// layer a table of its most recently used experts, all tables the same size.
pub(super) fn keep_whole_layers_and_recent_tables(
    trace: &ReplyTrace,
    byte_budget: u64,
    whole_layer_indexes: &[usize],
    first_scored_token_index: usize,
) -> KeepingOutcome {
    let table_slot_count = table_slots_beside_whole_layers(trace, byte_budget, whole_layer_indexes);
    let layer_count = trace.expert_bytes_by_layer.len();
    // Each table lists its experts from least to most recently used.
    let mut recent_experts_by_layer: Vec<Vec<u16>> = vec![Vec::new(); layer_count];
    let mut outcome = KeepingOutcome::default();
    for (token_index, token_route) in trace.token_routes.iter().enumerate() {
        let is_scored = token_index >= first_scored_token_index;
        outcome.scored_token_count += u64::from(is_scored);
        for (layer_index, layer_expert_ids) in token_route.iter().enumerate() {
            let layer_is_whole = whole_layer_indexes.contains(&layer_index);
            let recent_experts = &mut recent_experts_by_layer[layer_index];
            for &expert_id in layer_expert_ids {
                let table_position = recent_experts
                    .iter()
                    .position(|recent_expert_id| *recent_expert_id == expert_id);
                if is_scored {
                    outcome.record(
                        trace.expert_bytes_by_layer[layer_index],
                        layer_is_whole || table_position.is_some(),
                    );
                }
                if layer_is_whole || table_slot_count == 0 {
                    continue;
                }
                if let Some(table_position) = table_position {
                    recent_experts.remove(table_position);
                } else if recent_experts.len() == table_slot_count {
                    recent_experts.remove(0);
                }
                recent_experts.push(expert_id);
            }
        }
    }
    outcome
}

/// The layers a token's routes repeat least on, worst first: holding these
/// whole saves the most reads for the bytes spent. Repetition is the share
/// of a layer's assignments that the same layer also made on one of the
/// `lookback_token_count` tokens before.
pub(super) fn layers_by_least_repeated_routes(
    trace: &ReplyTrace,
    lookback_token_count: usize,
) -> Vec<usize> {
    let repeated_percent_by_layer = repeated_route_percent_by_layer(trace, lookback_token_count);
    let mut layer_indexes: Vec<usize> = (0..repeated_percent_by_layer.len()).collect();
    layer_indexes.sort_by(|first, second| {
        repeated_percent_by_layer[*first]
            .total_cmp(&repeated_percent_by_layer[*second])
            .then(first.cmp(second))
    });
    layer_indexes
}

/// For each layer, the share (0 to 100) of its assignments naming an expert
/// the layer also routed to on one of the `lookback_token_count` tokens
/// before. A layer no token routed through reports zero.
pub(super) fn repeated_route_percent_by_layer(
    trace: &ReplyTrace,
    lookback_token_count: usize,
) -> Vec<f64> {
    let layer_count = trace.expert_bytes_by_layer.len();
    let mut assignment_count_by_layer = vec![0_u64; layer_count];
    let mut repeated_count_by_layer = vec![0_u64; layer_count];
    for (token_index, token_route) in trace.token_routes.iter().enumerate() {
        let lookback_start_index = token_index.saturating_sub(lookback_token_count);
        for (layer_index, layer_expert_ids) in token_route.iter().enumerate() {
            for expert_id in layer_expert_ids {
                assignment_count_by_layer[layer_index] += 1;
                let mut was_used_recently = false;
                for earlier_route in &trace.token_routes[lookback_start_index..token_index] {
                    was_used_recently |= earlier_route[layer_index].contains(expert_id);
                }
                repeated_count_by_layer[layer_index] += u64::from(was_used_recently);
            }
        }
    }
    let mut repeated_percent_by_layer = Vec::with_capacity(layer_count);
    for (assignment_count, repeated_count) in assignment_count_by_layer
        .iter()
        .zip(&repeated_count_by_layer)
    {
        repeated_percent_by_layer.push(if *assignment_count == 0 {
            0.0
        } else {
            *repeated_count as f64 * 100.0 / *assignment_count as f64
        });
    }
    repeated_percent_by_layer
}

/// How many different experts each layer used over the whole trace.
pub(super) fn distinct_expert_count_by_layer(trace: &ReplyTrace) -> Vec<usize> {
    let mut distinct_count_by_layer = vec![0_usize; trace.expert_bytes_by_layer.len()];
    for (layer_index, _) in use_counts(trace, 0..trace.token_routes.len()).into_keys() {
        distinct_count_by_layer[layer_index] += 1;
    }
    distinct_count_by_layer
}

#[cfg(test)]
#[path = "expert_keeping_simulation_tests.rs"]
mod tests;

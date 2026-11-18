//! A replay whose per-layer tables are re-sized every so many tokens.
//!
//! One capacity shared by every layer misses about half what equal tables
//! do, because a layer's need moves during a reply. A table the engine
//! multiplies from is one fixed-size array, though, so true sharing would
//! re-size arrays on almost every miss. Re-sizing every `interval_tokens`
//! tokens, from how many distinct experts each layer used in the window just
//! ended, asks how much of sharing's gain a cheap periodic re-size keeps.

use std::collections::BTreeSet;

use super::keeping_rules::KeepingRule;
use super::replay_lab::{HeldExperts, begin_replay, empty_outcome, expert_bytes_of, serve_token};
use super::{ReplayCase, ReplayOutcome};

/// Serves `replay_case`, whose `layer_slot_limits` set the tables' starting
/// sizes, re-dividing the table slots every `interval_tokens` tokens in
/// proportion to each table layer's distinct experts in the window, with at
/// least `minimum_table_slots` each. Pinned layers keep their size.
pub fn replay_with_rebalanced_tables(
    replay_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    interval_tokens: usize,
    minimum_table_slots: usize,
) -> ReplayOutcome {
    let mut current_case = replay_case.clone();
    let mut held_experts = begin_replay(&current_case, keeping_rule);
    let mut replay_outcome = empty_outcome(&current_case);
    let mut window_experts_by_layer =
        vec![BTreeSet::new(); current_case.routed_reply.layer_count()];
    for token_index in 0..current_case.routed_reply.token_count() {
        if interval_tokens > 0 && token_index > 0 && token_index % interval_tokens == 0 {
            rebalance_tables(
                &mut current_case,
                keeping_rule,
                &mut held_experts,
                &window_experts_by_layer,
                minimum_table_slots,
            );
            for window_experts in &mut window_experts_by_layer {
                window_experts.clear();
            }
        }
        for (layer_index, window_experts) in window_experts_by_layer.iter_mut().enumerate() {
            window_experts.extend(
                current_case
                    .routed_reply
                    .layer_route(token_index, layer_index)
                    .iter()
                    .copied(),
            );
        }
        serve_token(
            &current_case,
            keeping_rule,
            &mut held_experts,
            token_index,
            &mut replay_outcome,
        );
    }
    replay_outcome
}

/// New table sizes from the window's distinct experts, then the experts
/// past each shrunk table's new size given up, lowest priority first.
fn rebalance_tables(
    current_case: &mut ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
    window_experts_by_layer: &[BTreeSet<u16>],
    minimum_table_slots: usize,
) {
    let new_layer_slot_limits =
        rebalanced_layer_slot_limits(current_case, window_experts_by_layer, minimum_table_slots);
    for (layer_index, &new_slot_limit) in new_layer_slot_limits.iter().enumerate() {
        let held_count = held_experts.held_count(layer_index);
        if held_count <= new_slot_limit {
            continue;
        }
        let expert_bytes = expert_bytes_of(current_case, layer_index);
        for expert in held_experts
            .layer_experts_by_priority(layer_index)
            .into_iter()
            .take(held_count - new_slot_limit)
        {
            held_experts.remove(expert, expert_bytes);
            keeping_rule.forget(expert);
        }
    }
    current_case.layer_slot_limits = Some(new_layer_slot_limits);
}

/// The table slots re-divided over the table layers in proportion to their
/// window's distinct experts; pinned layers keep their limits.
pub(super) fn rebalanced_layer_slot_limits(
    current_case: &ReplayCase,
    window_experts_by_layer: &[BTreeSet<u16>],
    minimum_table_slots: usize,
) -> Vec<usize> {
    let current_limits = current_case
        .layer_slot_limits
        .clone()
        .unwrap_or_else(|| vec![usize::MAX; window_experts_by_layer.len()]);
    let experts_per_layer = usize::from(current_case.routed_reply.experts_per_layer());
    let mut table_slot_total = 0_usize;
    let mut window_breadth_total = 0_usize;
    for (layer_index, window_experts) in window_experts_by_layer.iter().enumerate() {
        if !current_case.pinned_layer_indexes.contains(&layer_index) {
            table_slot_total += current_limits[layer_index].min(experts_per_layer);
            window_breadth_total += window_experts.len().max(1);
        }
    }
    let mut new_limits = current_limits.clone();
    for (layer_index, window_experts) in window_experts_by_layer.iter().enumerate() {
        if current_case.pinned_layer_indexes.contains(&layer_index) {
            continue;
        }
        let share = (table_slot_total * window_experts.len().max(1))
            .checked_div(window_breadth_total)
            .unwrap_or(0);
        new_limits[layer_index] = share.max(minimum_table_slots).min(experts_per_layer);
    }
    new_limits
}

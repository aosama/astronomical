//! A recorded reply served again under the capacity it really had, token by
//! token (`capacity_schedule`), with a rule for what to give up when the
//! capacity falls below what is held.
//!
//! The engine's tables are one array each, so the only memory it can give
//! back at once is a whole table: today it drops the table holding the
//! fewest experts, again and again, until what is held fits. A table dropped
//! that way is read back expert by expert as the reply routes to it, so a
//! capacity that dips for one reading and comes back costs a whole table's
//! reads. Replaying the same routes under the same schedule with each rule
//! shows what a dip costs and what another rule would have saved.
//!
//! The lab differs from the engine in two ways worth knowing when its rows
//! are read: the engine may drop a table that holds no experts and frees a
//! table's whole padded array, while the lab drops only layers that hold
//! experts and frees exactly their experts' bytes. Pinned layers (held whole
//! so their routes never stop) are given up only when nothing else is left.

use super::keeping_rules::KeepingRule;
use super::replay_lab::{
    HeldExperts, ReplayCase, ReplayOutcome, begin_replay, empty_outcome, expert_bytes_of,
    serve_token,
};

/// What to give up when the capacity falls below what is held.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapacityTrim {
    /// Today's engine: every expert of the layer holding the fewest, layer
    /// after layer, until what is held fits.
    DropLeastOccupiedLayer,
    /// One expert at a time, the one the keeping rule values least: a bound
    /// for tables that could give back part of themselves.
    GiveUpLowestPriorityExpert,
}

/// What one scheduled replay did.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScheduledReplayOutcome {
    pub replay_outcome: ReplayOutcome,
    /// Layers given up whole because the capacity fell.
    pub dropped_layer_count: u64,
    /// Experts given up because the capacity fell, by either rule.
    pub trimmed_expert_count: u64,
}

/// Serves `replay_case` with token `t` under `capacity_by_token[t]`; tokens
/// past the end of the schedule keep its last capacity. The starting experts
/// are seated under the case's own capacity, the one the reply began with, so
/// a capacity that dips at the first token goes through `capacity_trim` like
/// any other.
#[must_use]
pub fn replay_under_capacity_schedule(
    replay_case: &ReplayCase,
    capacity_by_token: &[u64],
    keeping_rule: &mut dyn KeepingRule,
    capacity_trim: CapacityTrim,
) -> ScheduledReplayOutcome {
    let mut scheduled_case = replay_case.clone();
    let mut held_experts = begin_replay(&scheduled_case, keeping_rule);
    let mut scheduled_outcome = ScheduledReplayOutcome {
        replay_outcome: empty_outcome(&scheduled_case),
        ..ScheduledReplayOutcome::default()
    };
    for token_index in 0..scheduled_case.routed_reply.token_count() {
        if let Some(&capacity_bytes) = capacity_by_token.get(token_index) {
            scheduled_case.capacity_bytes = capacity_bytes;
        }
        trim_to_capacity(
            &scheduled_case,
            keeping_rule,
            &mut held_experts,
            capacity_trim,
            &mut scheduled_outcome,
        );
        serve_token(
            &scheduled_case,
            keeping_rule,
            &mut held_experts,
            token_index,
            &mut scheduled_outcome.replay_outcome,
        );
        scheduled_outcome
            .replay_outcome
            .note_token_served(token_index);
    }
    scheduled_outcome
}

fn trim_to_capacity(
    scheduled_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
    capacity_trim: CapacityTrim,
    scheduled_outcome: &mut ScheduledReplayOutcome,
) {
    while held_experts.held_bytes() > scheduled_case.capacity_bytes {
        let given_up_count = match capacity_trim {
            CapacityTrim::DropLeastOccupiedLayer => {
                let unpinned_layer =
                    held_experts.least_occupied_layer(&scheduled_case.pinned_layer_indexes);
                let Some(layer_index) = (match unpinned_layer {
                    Some(layer_index) => Some(layer_index),
                    None => held_experts.least_occupied_layer(&[]),
                }) else {
                    return;
                };
                scheduled_outcome.dropped_layer_count += 1;
                give_up_layer(scheduled_case, keeping_rule, held_experts, layer_index)
            }
            CapacityTrim::GiveUpLowestPriorityExpert => {
                let unpinned_expert =
                    held_experts.lowest_priority_expert(&scheduled_case.pinned_layer_indexes);
                let Some(expert) = (match unpinned_expert {
                    Some(expert) => Some(expert),
                    None => held_experts.lowest_priority_expert(&[]),
                }) else {
                    return;
                };
                held_experts.remove(expert, expert_bytes_of(scheduled_case, expert.layer_index));
                keeping_rule.forget(expert);
                1
            }
        };
        scheduled_outcome.trimmed_expert_count += given_up_count;
    }
}

/// Gives up every held expert of `layer_index`; returns how many.
fn give_up_layer(
    scheduled_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
    layer_index: usize,
) -> u64 {
    let expert_bytes = expert_bytes_of(scheduled_case, layer_index);
    let layer_experts = held_experts.layer_experts_by_priority(layer_index);
    for &expert in &layer_experts {
        held_experts.remove(expert, expert_bytes);
        keeping_rule.forget(expert);
    }
    layer_experts.len() as u64
}

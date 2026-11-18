//! The replay lab: one recorded reply served under one keeping rule, counting
//! how often a reply token needed an expert the `astro_expert_cache` did not
//! hold.
//!
//! Routing depends only on the model and the text, so a reply recorded from
//! the engine (its routes, what the `astro_expert_cache` held when it began,
//! and the capacity it had) can be served again here under any rule in
//! milliseconds, with no GPU. Every rule sees the same routes, the same
//! capacity in bytes and the same starting set, so the only difference between
//! two replays is the rule.
//!
//! A rule keeps a priority for each held expert; the lowest is given up first.
//! While one layer of one token is served, its routed experts are never given
//! up: the engine multiplies them together. A rule may also refuse to admit a
//! newcomer when keeping it would give up something worth more, in which case
//! the expert is read for this token and not kept. Belady's rule needs that
//! choice to be optimal.

use std::collections::{BTreeSet, HashMap};

use super::keeping_rules::{KeepingRule, ReplayPosition};
use super::{ExpertKey, RoutedReply};

/// One recorded reply and the `astro_expert_cache` it was served from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayCase {
    pub routed_reply: RoutedReply,
    /// One expert's payload bytes in each layer.
    pub expert_payload_bytes_by_layer: Vec<u64>,
    /// The `astro_expert_cache` capacity.
    pub capacity_bytes: u64,
    /// For each layer, the experts held when the reply began.
    pub held_expert_ids_by_layer_at_start: Vec<Vec<u16>>,
    /// When set, each layer may hold at most this many experts and a full
    /// layer gives up one of its own for a newcomer: memory divided by layer,
    /// the way the engine's slot tables divide it today. `None` shares the
    /// capacity across every layer.
    pub layer_slot_limits: Option<Vec<usize>>,
    /// Layers whose experts are never given up: layers kept whole so their
    /// routes never stop.
    pub pinned_layer_indexes: Vec<usize>,
}

/// What one replay did.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayOutcome {
    pub reply_token_count: usize,
    /// Routed experts found held.
    pub hit_count: u64,
    /// Routed experts not held, so each was read from storage.
    pub miss_count: u64,
    /// Payload bytes those misses read.
    pub read_bytes: u64,
    /// Layer-tokens served while their layer was not held whole. The engine
    /// keeps a route on the graphics processor only for a layer held whole;
    /// every other layer-token stops to copy its route to the host, so this
    /// is the stop count a rule costs.
    pub partly_held_layer_token_count: u64,
}

impl ReplayOutcome {
    /// Misses per reply token, the lab's score; zero for an empty reply.
    #[must_use]
    pub fn misses_per_token(&self) -> f64 {
        per_reply_token(self.miss_count, self.reply_token_count)
    }

    /// Layers served while partly held, per reply token.
    #[must_use]
    pub fn partly_held_layers_per_token(&self) -> f64 {
        per_reply_token(self.partly_held_layer_token_count, self.reply_token_count)
    }
}

fn per_reply_token(total: u64, reply_token_count: usize) -> f64 {
    if reply_token_count == 0 {
        return 0.0;
    }
    total as f64 / reply_token_count as f64
}

/// The held experts, ordered by the rule's priority.
struct HeldExperts {
    priority_by_expert: HashMap<ExpertKey, (u64, u64)>,
    experts_by_priority: BTreeSet<((u64, u64), ExpertKey)>,
    held_bytes: u64,
    held_count_by_layer: HashMap<usize, usize>,
}

impl HeldExperts {
    fn new() -> Self {
        Self {
            priority_by_expert: HashMap::new(),
            experts_by_priority: BTreeSet::new(),
            held_bytes: 0,
            held_count_by_layer: HashMap::new(),
        }
    }

    fn is_held(&self, expert: ExpertKey) -> bool {
        self.priority_by_expert.contains_key(&expert)
    }

    fn set_priority(&mut self, expert: ExpertKey, keep_priority: (u64, u64)) {
        if let Some(previous_priority) = self.priority_by_expert.insert(expert, keep_priority) {
            self.experts_by_priority
                .remove(&(previous_priority, expert));
        }
        self.experts_by_priority.insert((keep_priority, expert));
    }

    fn insert(&mut self, expert: ExpertKey, keep_priority: (u64, u64), expert_bytes: u64) {
        self.set_priority(expert, keep_priority);
        self.held_bytes = self.held_bytes.saturating_add(expert_bytes);
        *self
            .held_count_by_layer
            .entry(expert.layer_index)
            .or_insert(0) += 1;
    }

    fn held_count(&self, layer_index: usize) -> usize {
        self.held_count_by_layer
            .get(&layer_index)
            .copied()
            .unwrap_or(0)
    }

    fn remove(&mut self, expert: ExpertKey, expert_bytes: u64) {
        if let Some(keep_priority) = self.priority_by_expert.remove(&expert) {
            self.experts_by_priority.remove(&(keep_priority, expert));
            self.held_bytes = self.held_bytes.saturating_sub(expert_bytes);
            if let Some(held_count) = self.held_count_by_layer.get_mut(&expert.layer_index) {
                *held_count = held_count.saturating_sub(1);
            }
        }
    }
}

/// Serves `replay_case` under `keeping_rule`.
pub fn replay(replay_case: &ReplayCase, keeping_rule: &mut dyn KeepingRule) -> ReplayOutcome {
    keeping_rule.prepare(&replay_case.routed_reply);
    let mut held_experts = HeldExperts::new();
    seat_starting_experts(replay_case, keeping_rule, &mut held_experts);
    let routed_reply = &replay_case.routed_reply;
    let mut replay_outcome = ReplayOutcome {
        reply_token_count: routed_reply.token_count(),
        ..ReplayOutcome::default()
    };
    let mut protected_route = Vec::with_capacity(routed_reply.experts_per_token());
    for token_index in 0..routed_reply.token_count() {
        for layer_index in 0..routed_reply.layer_count() {
            protected_route.clear();
            for &expert_id in routed_reply.layer_route(token_index, layer_index) {
                protected_route.push(ExpertKey {
                    layer_index,
                    expert_id,
                });
            }
            let replay_position = ReplayPosition {
                token_index,
                layer_index,
            };
            serve_layer_route(
                replay_case,
                keeping_rule,
                &mut held_experts,
                &protected_route,
                replay_position,
                &mut replay_outcome,
            );
        }
    }
    replay_outcome
}

/// Seats the experts the reply began with, as far as the capacity allows.
fn seat_starting_experts(
    replay_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
) {
    for (layer_index, held_expert_ids) in replay_case
        .held_expert_ids_by_layer_at_start
        .iter()
        .enumerate()
    {
        let expert_bytes = expert_bytes_of(replay_case, layer_index);
        let layer_slot_limit = layer_slot_limit_of(replay_case, layer_index).unwrap_or(usize::MAX);
        for &expert_id in held_expert_ids.iter().take(layer_slot_limit) {
            if held_experts.held_bytes.saturating_add(expert_bytes) > replay_case.capacity_bytes {
                return;
            }
            let expert = ExpertKey {
                layer_index,
                expert_id,
            };
            let keep_priority = keeping_rule.starting_priority(expert);
            held_experts.insert(expert, keep_priority, expert_bytes);
        }
    }
}

fn serve_layer_route(
    replay_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
    protected_route: &[ExpertKey],
    replay_position: ReplayPosition,
    replay_outcome: &mut ReplayOutcome,
) {
    let expert_bytes = expert_bytes_of(replay_case, replay_position.layer_index);
    if held_experts.held_count(replay_position.layer_index)
        < usize::from(replay_case.routed_reply.experts_per_layer())
    {
        replay_outcome.partly_held_layer_token_count += 1;
    }
    for &expert in protected_route {
        let keep_priority = keeping_rule.priority_after_use(expert, replay_position);
        if held_experts.is_held(expert) {
            replay_outcome.hit_count += 1;
            held_experts.set_priority(expert, keep_priority);
            continue;
        }
        replay_outcome.miss_count += 1;
        replay_outcome.read_bytes = replay_outcome.read_bytes.saturating_add(expert_bytes);
        admit_if_worth_it(
            replay_case,
            keeping_rule,
            held_experts,
            protected_route,
            (expert, keep_priority, expert_bytes),
        );
    }
}

/// Keeps a newly read expert when room can be made for it without giving up
/// anything the rule values more. Every victim is chosen and checked before
/// any is given up, so a refusal part way never loses an expert for nothing.
fn admit_if_worth_it(
    replay_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
    protected_route: &[ExpertKey],
    (incoming_expert, incoming_priority, incoming_bytes): (ExpertKey, (u64, u64), u64),
) {
    if incoming_bytes > replay_case.capacity_bytes {
        return;
    }
    if let Some(layer_slot_limit) = layer_slot_limit_of(replay_case, incoming_expert.layer_index)
        && held_experts.held_count(incoming_expert.layer_index) >= layer_slot_limit
    {
        replace_within_layer(
            replay_case,
            keeping_rule,
            held_experts,
            protected_route,
            (incoming_expert, incoming_priority, incoming_bytes),
        );
        return;
    }
    let mut victims = Vec::new();
    let mut freed_bytes = 0_u64;
    for &(victim_priority, victim) in &held_experts.experts_by_priority {
        if held_experts.held_bytes - freed_bytes + incoming_bytes <= replay_case.capacity_bytes {
            break;
        }
        if protected_route.contains(&victim)
            || replay_case
                .pinned_layer_indexes
                .contains(&victim.layer_index)
        {
            continue;
        }
        if !keeping_rule.admits(incoming_expert, incoming_priority, victim, victim_priority) {
            return;
        }
        victims.push(victim);
        freed_bytes += expert_bytes_of(replay_case, victim.layer_index);
    }
    if held_experts.held_bytes - freed_bytes + incoming_bytes > replay_case.capacity_bytes {
        return;
    }
    for victim in victims {
        held_experts.remove(victim, expert_bytes_of(replay_case, victim.layer_index));
        keeping_rule.forget(victim);
    }
    held_experts.insert(incoming_expert, incoming_priority, incoming_bytes);
}

/// A full layer under a slot limit gives up its own lowest-priority expert,
/// if the rule finds the newcomer worth it.
fn replace_within_layer(
    replay_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
    protected_route: &[ExpertKey],
    (incoming_expert, incoming_priority, incoming_bytes): (ExpertKey, (u64, u64), u64),
) {
    let mut layer_victim = None;
    for &(victim_priority, victim) in &held_experts.experts_by_priority {
        if victim.layer_index == incoming_expert.layer_index
            && !protected_route.contains(&victim)
            && !replay_case
                .pinned_layer_indexes
                .contains(&victim.layer_index)
        {
            layer_victim = Some((victim_priority, victim));
            break;
        }
    }
    let Some((victim_priority, victim)) = layer_victim else {
        return;
    };
    if !keeping_rule.admits(incoming_expert, incoming_priority, victim, victim_priority) {
        return;
    }
    held_experts.remove(victim, expert_bytes_of(replay_case, victim.layer_index));
    keeping_rule.forget(victim);
    held_experts.insert(incoming_expert, incoming_priority, incoming_bytes);
}

fn layer_slot_limit_of(replay_case: &ReplayCase, layer_index: usize) -> Option<usize> {
    let layer_slot_limits = replay_case.layer_slot_limits.as_ref()?;
    Some(layer_slot_limits.get(layer_index).copied().unwrap_or(0))
}

fn expert_bytes_of(replay_case: &ReplayCase, layer_index: usize) -> u64 {
    replay_case
        .expert_payload_bytes_by_layer
        .get(layer_index)
        .copied()
        .unwrap_or(0)
}

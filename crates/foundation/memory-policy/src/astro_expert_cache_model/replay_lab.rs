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
    /// Whether a layer's table costs its whole slot allocation for as long
    /// as it holds any expert, as the engine's tables do: one array per
    /// layer, allocated before it fills and given back only whole. Needs
    /// `layer_slot_limits`, which give each table's size. Otherwise every
    /// expert costs its own bytes and nothing else.
    pub charges_whole_table_slots: bool,
}

/// What one replay did.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
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
    /// Misses while the first `EARLY_REPLY_TOKEN_COUNT` tokens were served.
    pub early_miss_count: u64,
    /// Misses in each layer, so a replay can be held against the engine's
    /// own misses layer by layer.
    pub miss_count_by_layer: Vec<u64>,
}

/// The opening of a reply, counted apart: what is held when a reply begins
/// was chosen by the prompt, and the engine's first reply tokens are slower
/// than the rest (about 33 ms against 18 on a 35B-A3B at a 16 GB limit).
pub const EARLY_REPLY_TOKEN_COUNT: usize = 48;

impl ReplayOutcome {
    /// One routed expert of `layer_index` read from storage.
    fn count_miss(&mut self, layer_index: usize, expert_bytes: u64) {
        self.miss_count += 1;
        self.read_bytes = self.read_bytes.saturating_add(expert_bytes);
        if self.miss_count_by_layer.len() <= layer_index {
            self.miss_count_by_layer.resize(layer_index + 1, 0);
        }
        self.miss_count_by_layer[layer_index] += 1;
    }

    /// Called once a token is served: keeps the miss count as it stood at
    /// the end of the reply's opening.
    pub(super) fn note_token_served(&mut self, token_index: usize) {
        if token_index < EARLY_REPLY_TOKEN_COUNT {
            self.early_miss_count = self.miss_count;
        }
    }

    /// Misses a token over the reply's opening; zero for an empty reply.
    #[must_use]
    pub fn early_misses_per_token(&self) -> f64 {
        per_reply_token(
            self.early_miss_count,
            self.reply_token_count.min(EARLY_REPLY_TOKEN_COUNT),
        )
    }

    /// Misses a token after the reply's opening; zero when the reply ends
    /// inside it.
    #[must_use]
    pub fn later_misses_per_token(&self) -> f64 {
        per_reply_token(
            self.miss_count.saturating_sub(self.early_miss_count),
            self.reply_token_count
                .saturating_sub(EARLY_REPLY_TOKEN_COUNT),
        )
    }

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
pub(super) struct HeldExperts {
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

    /// The held experts of `layer_index`, lowest priority first.
    pub(super) fn layer_experts_by_priority(&self, layer_index: usize) -> Vec<ExpertKey> {
        let mut layer_experts = Vec::new();
        for &(_, expert) in &self.experts_by_priority {
            if expert.layer_index == layer_index {
                layer_experts.push(expert);
            }
        }
        layer_experts
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

    /// What the held experts cost against the capacity: their own bytes, or
    /// with `charges_whole_table_slots` every table's whole allocation.
    pub(super) fn charged_bytes(&self, replay_case: &ReplayCase) -> u64 {
        if !replay_case.charges_whole_table_slots {
            return self.held_bytes;
        }
        let mut charged_bytes = 0_u64;
        for (&layer_index, &held_count) in &self.held_count_by_layer {
            if held_count > 0 {
                charged_bytes =
                    charged_bytes.saturating_add(table_bytes_of(replay_case, layer_index));
            }
        }
        charged_bytes
    }

    /// The layer holding the fewest experts, of those holding any and not
    /// in `spared_layer_indexes`; the lowest index among equals, as the
    /// engine scans its tables in order.
    pub(super) fn least_occupied_layer(&self, spared_layer_indexes: &[usize]) -> Option<usize> {
        let mut least: Option<(usize, usize)> = None;
        for (&layer_index, &held_count) in &self.held_count_by_layer {
            if held_count == 0 || spared_layer_indexes.contains(&layer_index) {
                continue;
            }
            let is_fewer = match least {
                None => true,
                Some((least_index, least_count)) => {
                    held_count < least_count
                        || (held_count == least_count && layer_index < least_index)
                }
            };
            if is_fewer {
                least = Some((layer_index, held_count));
            }
        }
        least.map(|(layer_index, _)| layer_index)
    }

    /// The held expert the rule values least, outside `spared_layer_indexes`.
    pub(super) fn lowest_priority_expert(
        &self,
        spared_layer_indexes: &[usize],
    ) -> Option<ExpertKey> {
        self.experts_by_priority
            .iter()
            .map(|&(_, expert)| expert)
            .find(|expert| !spared_layer_indexes.contains(&expert.layer_index))
    }

    pub(super) fn held_count(&self, layer_index: usize) -> usize {
        self.held_count_by_layer
            .get(&layer_index)
            .copied()
            .unwrap_or(0)
    }

    pub(super) fn remove(&mut self, expert: ExpertKey, expert_bytes: u64) {
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
    let mut held_experts = begin_replay(replay_case, keeping_rule);
    let mut replay_outcome = empty_outcome(replay_case);
    for token_index in 0..replay_case.routed_reply.token_count() {
        serve_token(
            replay_case,
            keeping_rule,
            &mut held_experts,
            token_index,
            &mut replay_outcome,
        );
        replay_outcome.note_token_served(token_index);
    }
    replay_outcome
}

/// Prepares the rule and seats the starting experts.
pub(super) fn begin_replay(
    replay_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
) -> HeldExperts {
    keeping_rule.prepare(&replay_case.routed_reply);
    let mut held_experts = HeldExperts::new();
    seat_starting_experts(replay_case, keeping_rule, &mut held_experts);
    held_experts
}

pub(super) fn empty_outcome(replay_case: &ReplayCase) -> ReplayOutcome {
    ReplayOutcome {
        reply_token_count: replay_case.routed_reply.token_count(),
        ..ReplayOutcome::default()
    }
}

/// Serves every layer of one token.
pub(super) fn serve_token(
    replay_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
    token_index: usize,
    replay_outcome: &mut ReplayOutcome,
) {
    let routed_reply = &replay_case.routed_reply;
    let mut protected_route = Vec::with_capacity(routed_reply.experts_per_token());
    for layer_index in 0..routed_reply.layer_count() {
        protected_route.clear();
        for &expert_id in routed_reply.layer_route(token_index, layer_index) {
            protected_route.push(ExpertKey {
                layer_index,
                expert_id,
            });
        }
        serve_layer_route(
            replay_case,
            keeping_rule,
            held_experts,
            &protected_route,
            ReplayPosition {
                token_index,
                layer_index,
            },
            replay_outcome,
        );
    }
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
        let (seated_count, capacity_is_full) = starting_experts_that_fit(
            replay_case,
            held_experts,
            layer_index,
            held_expert_ids.len(),
        );
        for &expert_id in held_expert_ids.iter().take(seated_count) {
            let expert = ExpertKey {
                layer_index,
                expert_id,
            };
            let keep_priority = keeping_rule.starting_priority(expert);
            held_experts.insert(expert, keep_priority, expert_bytes);
        }
        if capacity_is_full {
            return;
        }
    }
}

/// How many of a layer's `starting_count` experts are seated, and whether
/// the capacity is then full, so no later layer is seated. A layer takes no
/// more than its slot limit; a whole-slot table is seated whole or not at
/// all, and other experts one by one while their bytes fit.
fn starting_experts_that_fit(
    replay_case: &ReplayCase,
    held_experts: &HeldExperts,
    layer_index: usize,
    starting_count: usize,
) -> (usize, bool) {
    let allowed_count =
        starting_count.min(layer_slot_limit_of(replay_case, layer_index).unwrap_or(usize::MAX));
    if replay_case.charges_whole_table_slots {
        let table_fits = starting_count == 0
            || held_experts
                .charged_bytes(replay_case)
                .saturating_add(table_bytes_of(replay_case, layer_index))
                <= replay_case.capacity_bytes;
        return if table_fits {
            (allowed_count, false)
        } else {
            (0, true)
        };
    }
    let room_bytes = replay_case
        .capacity_bytes
        .saturating_sub(held_experts.held_bytes);
    let fitting_count = room_bytes
        .checked_div(expert_bytes_of(replay_case, layer_index))
        .map_or(usize::MAX, |fitting| {
            usize::try_from(fitting).unwrap_or(usize::MAX)
        });
    (
        allowed_count.min(fitting_count),
        fitting_count < allowed_count,
    )
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
        if held_experts.is_held(expert) {
            let keep_priority = keeping_rule.priority_after_use(expert, replay_position);
            replay_outcome.hit_count += 1;
            held_experts.set_priority(expert, keep_priority);
            continue;
        }
        let keep_priority = keeping_rule.priority_after_miss(expert, replay_position);
        replay_outcome.count_miss(replay_position.layer_index, expert_bytes);
        admit_read_expert(
            replay_case,
            keeping_rule,
            held_experts,
            protected_route,
            (expert, keep_priority, expert_bytes),
        );
    }
}

/// Offers a newly read expert to the tables, by the way the case charges
/// them.
fn admit_read_expert(
    replay_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
    protected_route: &[ExpertKey],
    incoming: (ExpertKey, (u64, u64), u64),
) {
    if replay_case.charges_whole_table_slots {
        admit_into_whole_slot_table(
            replay_case,
            keeping_rule,
            held_experts,
            protected_route,
            incoming,
        );
    } else {
        admit_if_worth_it(
            replay_case,
            keeping_rule,
            held_experts,
            protected_route,
            incoming,
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

/// Admission when tables are charged whole: an expert joins its layer's
/// table while the table has a free slot, a full table replaces within
/// itself, and a layer with no table gets one only if its whole allocation
/// fits under the capacity. The engine never gives up another layer's
/// experts to admit one; only a falling capacity does that.
fn admit_into_whole_slot_table(
    replay_case: &ReplayCase,
    keeping_rule: &mut dyn KeepingRule,
    held_experts: &mut HeldExperts,
    protected_route: &[ExpertKey],
    (incoming_expert, incoming_priority, incoming_bytes): (ExpertKey, (u64, u64), u64),
) {
    let layer_index = incoming_expert.layer_index;
    let table_slot_count = layer_slot_limit_of(replay_case, layer_index).unwrap_or(0);
    let held_count = held_experts.held_count(layer_index);
    if table_slot_count == 0 {
        return;
    }
    if held_count == 0
        && held_experts
            .charged_bytes(replay_case)
            .saturating_add(table_bytes_of(replay_case, layer_index))
            > replay_case.capacity_bytes
    {
        return;
    }
    if held_count < table_slot_count {
        held_experts.insert(incoming_expert, incoming_priority, incoming_bytes);
        return;
    }
    replace_within_layer(
        replay_case,
        keeping_rule,
        held_experts,
        protected_route,
        (incoming_expert, incoming_priority, incoming_bytes),
    );
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

/// One layer's whole table: its slot allocation times one expert's bytes.
pub(super) fn table_bytes_of(replay_case: &ReplayCase, layer_index: usize) -> u64 {
    let table_slot_count = layer_slot_limit_of(replay_case, layer_index).unwrap_or(0);
    expert_bytes_of(replay_case, layer_index)
        .saturating_mul(u64::try_from(table_slot_count).unwrap_or(u64::MAX))
}

pub(super) fn expert_bytes_of(replay_case: &ReplayCase, layer_index: usize) -> u64 {
    replay_case
        .expert_payload_bytes_by_layer
        .get(layer_index)
        .copied()
        .unwrap_or(0)
}

//! Keeping rules for the replay lab: which held expert the
//! `astro_expert_cache` gives up first, and whether a newly read expert is
//! worth keeping at all.
//!
//! A rule gives every held expert a priority; the lab gives up the lowest
//! first. Priorities are pairs so a rule can break ties (frequency, then
//! recency). `BeladyMinimum` reads the future and is the bound no rule that
//! only sees the past can beat.

use std::collections::HashMap;

use super::{ExpertKey, RoutedReply};

/// Where in the reply an expert is being used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayPosition {
    pub token_index: usize,
    pub layer_index: usize,
}

/// A rule the replay lab can serve a reply under.
pub trait KeepingRule {
    /// A short name for reports.
    fn name(&self) -> String;

    /// Called once before the reply is served.
    fn prepare(&mut self, _routed_reply: &RoutedReply) {}

    /// The priority of an expert held when the reply began.
    fn starting_priority(&mut self, expert: ExpertKey) -> (u64, u64);

    /// The priority of `expert` once it has been used at `replay_position`,
    /// whether it was held or just read.
    fn priority_after_use(
        &mut self,
        expert: ExpertKey,
        replay_position: ReplayPosition,
    ) -> (u64, u64);

    /// Whether `incoming` is worth giving up `victim` for. Every rule but a
    /// filtering one keeps every newcomer.
    fn admits(
        &mut self,
        _incoming: ExpertKey,
        _incoming_priority: (u64, u64),
        _victim: ExpertKey,
        _victim_priority: (u64, u64),
    ) -> bool {
        true
    }

    /// Called when `expert` is given up.
    fn forget(&mut self, _expert: ExpertKey) {}
}

/// One step of the reply: a token's layer, counted across the whole reply.
fn reply_step(replay_position: ReplayPosition, layer_count: usize) -> u64 {
    (replay_position.token_index * layer_count + replay_position.layer_index) as u64 + 1
}

/// Gives up the expert used longest ago.
#[derive(Debug, Default)]
pub struct LeastRecentlyUsed {
    layer_count: usize,
}

impl KeepingRule for LeastRecentlyUsed {
    fn name(&self) -> String {
        "least recently used".to_owned()
    }

    fn prepare(&mut self, routed_reply: &RoutedReply) {
        self.layer_count = routed_reply.layer_count();
    }

    fn starting_priority(&mut self, _expert: ExpertKey) -> (u64, u64) {
        (0, 0)
    }

    fn priority_after_use(
        &mut self,
        _expert: ExpertKey,
        replay_position: ReplayPosition,
    ) -> (u64, u64) {
        (reply_step(replay_position, self.layer_count), 0)
    }
}

/// Gives up the expert used least often in the reply so far, the one used
/// longest ago among equals.
#[derive(Debug, Default)]
pub struct LeastFrequentlyUsed {
    layer_count: usize,
    use_count_by_expert: HashMap<ExpertKey, u64>,
}

impl KeepingRule for LeastFrequentlyUsed {
    fn name(&self) -> String {
        "least frequently used".to_owned()
    }

    fn prepare(&mut self, routed_reply: &RoutedReply) {
        self.layer_count = routed_reply.layer_count();
        self.use_count_by_expert.clear();
    }

    fn starting_priority(&mut self, _expert: ExpertKey) -> (u64, u64) {
        (0, 0)
    }

    fn priority_after_use(
        &mut self,
        expert: ExpertKey,
        replay_position: ReplayPosition,
    ) -> (u64, u64) {
        let use_count = self.use_count_by_expert.entry(expert).or_insert(0);
        *use_count += 1;
        (*use_count, reply_step(replay_position, self.layer_count))
    }
}

/// Gives up the expert with the least recent use weight, where a use counts
/// half as much after `half_life_tokens` more tokens. Weights are kept on a
/// growing scale (a use at token `t` adds `2^(t / half_life)`) so held
/// experts never need re-weighting as time passes.
#[derive(Debug)]
pub struct DecayedFrequency {
    half_life_tokens: f64,
    layer_count: usize,
    use_weight_by_expert: HashMap<ExpertKey, f64>,
}

impl DecayedFrequency {
    #[must_use]
    pub fn new(half_life_tokens: f64) -> Self {
        Self {
            half_life_tokens,
            layer_count: 0,
            use_weight_by_expert: HashMap::new(),
        }
    }
}

impl KeepingRule for DecayedFrequency {
    fn name(&self) -> String {
        format!(
            "decayed frequency, half-life {} tokens",
            self.half_life_tokens
        )
    }

    fn prepare(&mut self, routed_reply: &RoutedReply) {
        self.layer_count = routed_reply.layer_count();
        self.use_weight_by_expert.clear();
    }

    fn starting_priority(&mut self, _expert: ExpertKey) -> (u64, u64) {
        (0, 0)
    }

    fn priority_after_use(
        &mut self,
        expert: ExpertKey,
        replay_position: ReplayPosition,
    ) -> (u64, u64) {
        let use_weight = self.use_weight_by_expert.entry(expert).or_insert(0.0);
        *use_weight += (replay_position.token_index as f64 / self.half_life_tokens).exp2();
        // A positive float's bits order the same way the float does.
        (
            use_weight.to_bits(),
            reply_step(replay_position, self.layer_count),
        )
    }
}

/// Keeps by recency, but admits a newcomer only when the reply has used it
/// more often than the expert it would replace (the admission test of
/// TinyLFU, with exact counts instead of a sketch).
#[derive(Debug, Default)]
pub struct FrequencyAdmittedRecency {
    layer_count: usize,
    use_count_by_expert: HashMap<ExpertKey, u64>,
}

impl KeepingRule for FrequencyAdmittedRecency {
    fn name(&self) -> String {
        "recency with frequency admission".to_owned()
    }

    fn prepare(&mut self, routed_reply: &RoutedReply) {
        self.layer_count = routed_reply.layer_count();
        self.use_count_by_expert.clear();
    }

    fn starting_priority(&mut self, _expert: ExpertKey) -> (u64, u64) {
        (0, 0)
    }

    fn priority_after_use(
        &mut self,
        expert: ExpertKey,
        replay_position: ReplayPosition,
    ) -> (u64, u64) {
        *self.use_count_by_expert.entry(expert).or_insert(0) += 1;
        (reply_step(replay_position, self.layer_count), 0)
    }

    fn admits(
        &mut self,
        incoming: ExpertKey,
        _incoming_priority: (u64, u64),
        victim: ExpertKey,
        _victim_priority: (u64, u64),
    ) -> bool {
        let incoming_use_count = self
            .use_count_by_expert
            .get(&incoming)
            .copied()
            .unwrap_or(0);
        let victim_use_count = self.use_count_by_expert.get(&victim).copied().unwrap_or(0);
        incoming_use_count > victim_use_count
    }
}

/// Belady's MIN: gives up the expert whose next use is furthest away, and
/// does not keep a newcomer needed later than everything it would replace.
/// It reads the future, so it is the bound for every rule that cannot.
#[derive(Debug, Default)]
pub struct BeladyMinimum {
    layer_count: usize,
    use_steps_by_expert: HashMap<ExpertKey, Vec<u64>>,
}

impl BeladyMinimum {
    /// Priority from the step of the next use after `after_step`: the sooner
    /// the next use, the higher; never used again is lowest.
    fn priority_for_next_use_after(&self, expert: ExpertKey, after_step: u64) -> (u64, u64) {
        let Some(use_steps) = self.use_steps_by_expert.get(&expert) else {
            return (0, 0);
        };
        // An expert is used at most once per step, so the steps are unique.
        let next_use_index = match use_steps.binary_search(&after_step) {
            Ok(this_use_index) => this_use_index + 1,
            Err(next_use_index) => next_use_index,
        };
        match use_steps.get(next_use_index) {
            Some(&next_use_step) => (u64::MAX - next_use_step, 0),
            None => (0, 0),
        }
    }
}

impl KeepingRule for BeladyMinimum {
    fn name(&self) -> String {
        "Belady's MIN (knows the future)".to_owned()
    }

    fn prepare(&mut self, routed_reply: &RoutedReply) {
        self.layer_count = routed_reply.layer_count();
        self.use_steps_by_expert.clear();
        for token_index in 0..routed_reply.token_count() {
            for layer_index in 0..routed_reply.layer_count() {
                let use_step = reply_step(
                    ReplayPosition {
                        token_index,
                        layer_index,
                    },
                    self.layer_count,
                );
                for &expert_id in routed_reply.layer_route(token_index, layer_index) {
                    self.use_steps_by_expert
                        .entry(ExpertKey {
                            layer_index,
                            expert_id,
                        })
                        .or_default()
                        .push(use_step);
                }
            }
        }
    }

    fn starting_priority(&mut self, expert: ExpertKey) -> (u64, u64) {
        self.priority_for_next_use_after(expert, 0)
    }

    fn priority_after_use(
        &mut self,
        expert: ExpertKey,
        replay_position: ReplayPosition,
    ) -> (u64, u64) {
        self.priority_for_next_use_after(expert, reply_step(replay_position, self.layer_count))
    }

    fn admits(
        &mut self,
        _incoming: ExpertKey,
        incoming_priority: (u64, u64),
        _victim: ExpertKey,
        victim_priority: (u64, u64),
    ) -> bool {
        incoming_priority > victim_priority
    }
}

/// Another rule with at most `replacements_per_layer_token` experts given up
/// per layer per token, as today's engine caps its tables
/// (`MAXIMUM_CANDIDATE_SLOT_REPLACEMENTS_PER_DECODE_STEP`); a newcomer past
/// the cap is read and not kept.
pub struct CappedReplacements {
    inner_rule: Box<dyn KeepingRule>,
    replacements_per_layer_token: usize,
    current_position: Option<ReplayPosition>,
    replacements_at_current_position: usize,
}

impl CappedReplacements {
    #[must_use]
    pub fn new(inner_rule: Box<dyn KeepingRule>, replacements_per_layer_token: usize) -> Self {
        Self {
            inner_rule,
            replacements_per_layer_token,
            current_position: None,
            replacements_at_current_position: 0,
        }
    }
}

impl KeepingRule for CappedReplacements {
    fn name(&self) -> String {
        format!(
            "{}, at most {} replacements per layer per token",
            self.inner_rule.name(),
            self.replacements_per_layer_token
        )
    }

    fn prepare(&mut self, routed_reply: &RoutedReply) {
        self.current_position = None;
        self.replacements_at_current_position = 0;
        self.inner_rule.prepare(routed_reply);
    }

    fn starting_priority(&mut self, expert: ExpertKey) -> (u64, u64) {
        self.inner_rule.starting_priority(expert)
    }

    fn priority_after_use(
        &mut self,
        expert: ExpertKey,
        replay_position: ReplayPosition,
    ) -> (u64, u64) {
        if self.current_position != Some(replay_position) {
            self.current_position = Some(replay_position);
            self.replacements_at_current_position = 0;
        }
        self.inner_rule.priority_after_use(expert, replay_position)
    }

    fn admits(
        &mut self,
        incoming: ExpertKey,
        incoming_priority: (u64, u64),
        victim: ExpertKey,
        victim_priority: (u64, u64),
    ) -> bool {
        if self.replacements_at_current_position >= self.replacements_per_layer_token {
            return false;
        }
        let admits = self
            .inner_rule
            .admits(incoming, incoming_priority, victim, victim_priority);
        if admits {
            self.replacements_at_current_position += 1;
        }
        admits
    }

    fn forget(&mut self, expert: ExpertKey) {
        self.inner_rule.forget(expert);
    }
}

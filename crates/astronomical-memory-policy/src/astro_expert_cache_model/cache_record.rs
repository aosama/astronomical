//! What the astro expert cache did: the evidence a slow reply is explained from.
//!
//! One blended hit rate hides the failure that matters. A model with 37
//! layers held whole and 3 layers starved reports 96% hits while those three
//! layers read from storage on every token. So the record keeps a row per
//! layer and a row per expert, separates a first read (unavoidable: the
//! expert was never in memory) from a repeat read (the cache gave up an
//! expert it needed again), and logs what each eviction gave up.

use std::collections::VecDeque;

use super::cache::ExpertKey;

/// Most recent evictions kept for inspection. A reply of a few thousand
/// tokens with a healthy cache evicts far fewer experts than this; a cache
/// that thrashes fills it, and the dropped count says by how much.
const RETAINED_EVICTION_EVENT_COUNT: usize = 4_096;

/// One expert's history in the cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertUse {
    /// Tokens that routed to this expert.
    pub use_count: u64,
    /// Times it was read from storage.
    pub read_count: u64,
    /// Times it was given up to make room.
    pub eviction_count: u64,
    pub is_held: bool,
}

/// One eviction: what was needed, what was given up for it, and how much the
/// evicted expert had been used, which is the question to ask of any cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvictionEvent {
    pub token_index: u64,
    pub needed: ExpertKey,
    pub evicted: ExpertKey,
    pub evicted_use_count: u64,
}

/// One layer's totals.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LayerCacheSummary {
    pub layer_index: usize,
    pub lookup_count: u64,
    pub hit_count: u64,
    /// Reads of experts that had never been in memory.
    pub first_read_count: u64,
    /// Reads of experts the cache had held before and gave up.
    pub repeat_read_count: u64,
    /// Reads served without keeping the expert, because nothing could be
    /// given up for it.
    pub unheld_read_count: u64,
    pub eviction_count: u64,
    pub held_expert_count: usize,
    /// Experts any token routed to.
    pub used_expert_count: usize,
    /// Experts in memory that no token has routed to: memory doing nothing.
    pub held_never_used_expert_count: usize,
}

/// Everything the cache did since it was created.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AstroExpertCacheRecord {
    token_count: u64,
    layer_totals: Vec<LayerCacheSummary>,
    expert_uses_by_layer: Vec<Vec<ExpertUse>>,
    eviction_events: VecDeque<EvictionEvent>,
    dropped_eviction_event_count: u64,
}

impl AstroExpertCacheRecord {
    pub(super) fn for_layers(expert_counts: impl Iterator<Item = u16>) -> Self {
        let expert_uses_by_layer = expert_counts
            .map(|expert_count| vec![ExpertUse::default(); usize::from(expert_count)])
            .collect::<Vec<_>>();
        let layer_totals = (0..expert_uses_by_layer.len())
            .map(|layer_index| LayerCacheSummary {
                layer_index,
                ..LayerCacheSummary::default()
            })
            .collect();
        Self {
            layer_totals,
            expert_uses_by_layer,
            ..Self::default()
        }
    }

    pub(super) fn begin_token(&mut self) {
        self.token_count += 1;
    }

    pub(super) fn expert_use_mut(&mut self, expert: ExpertKey) -> &mut ExpertUse {
        &mut self.expert_uses_by_layer[expert.layer_index][usize::from(expert.expert_id)]
    }

    pub(super) fn record_lookup(&mut self, expert: ExpertKey, is_hit: bool) {
        self.expert_use_mut(expert).use_count += 1;
        let layer_total = &mut self.layer_totals[expert.layer_index];
        layer_total.lookup_count += 1;
        if is_hit {
            layer_total.hit_count += 1;
        }
    }

    pub(super) fn record_read(&mut self, expert: ExpertKey, is_kept: bool) {
        let expert_use = self.expert_use_mut(expert);
        let was_in_memory_before = expert_use.read_count > 0 || expert_use.eviction_count > 0;
        expert_use.read_count += 1;
        expert_use.is_held = is_kept;
        let layer_total = &mut self.layer_totals[expert.layer_index];
        if was_in_memory_before {
            layer_total.repeat_read_count += 1;
        } else {
            layer_total.first_read_count += 1;
        }
        if !is_kept {
            layer_total.unheld_read_count += 1;
        }
    }

    pub(super) fn record_seated(&mut self, expert: ExpertKey) {
        self.expert_use_mut(expert).is_held = true;
    }

    pub(super) fn record_eviction(&mut self, needed: Option<ExpertKey>, evicted: ExpertKey) {
        let evicted_use = self.expert_use_mut(evicted);
        evicted_use.eviction_count += 1;
        evicted_use.is_held = false;
        let evicted_use_count = evicted_use.use_count;
        self.layer_totals[evicted.layer_index].eviction_count += 1;
        let Some(needed) = needed else {
            return;
        };
        if self.eviction_events.len() == RETAINED_EVICTION_EVENT_COUNT {
            self.eviction_events.pop_front();
            self.dropped_eviction_event_count += 1;
        }
        self.eviction_events.push_back(EvictionEvent {
            token_index: self.token_count,
            needed,
            evicted,
            evicted_use_count,
        });
    }

    #[must_use]
    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    /// One row per layer, with what it holds counted now.
    #[must_use]
    pub fn layer_summaries(&self) -> Vec<LayerCacheSummary> {
        self.layer_totals
            .iter()
            .zip(&self.expert_uses_by_layer)
            .map(|(layer_total, expert_uses)| summarize_layer(*layer_total, expert_uses))
            .collect()
    }

    /// One row per expert of one layer.
    #[must_use]
    pub fn expert_uses(&self, layer_index: usize) -> &[ExpertUse] {
        &self.expert_uses_by_layer[layer_index]
    }

    /// The most recent evictions, oldest first.
    pub fn eviction_events(&self) -> impl Iterator<Item = &EvictionEvent> {
        self.eviction_events.iter()
    }

    #[must_use]
    pub const fn dropped_eviction_event_count(&self) -> u64 {
        self.dropped_eviction_event_count
    }

    /// Reads from storage, of every kind, across all layers.
    #[must_use]
    pub fn read_count(&self) -> u64 {
        self.layer_totals
            .iter()
            .map(|layer_total| layer_total.first_read_count + layer_total.repeat_read_count)
            .sum()
    }

    /// Reads of experts the cache once held: the cache's own failures.
    #[must_use]
    pub fn repeat_read_count(&self) -> u64 {
        self.layer_totals
            .iter()
            .map(|layer_total| layer_total.repeat_read_count)
            .sum()
    }
}

fn summarize_layer(layer_total: LayerCacheSummary, expert_uses: &[ExpertUse]) -> LayerCacheSummary {
    let mut layer_summary = layer_total;
    for expert_use in expert_uses {
        if expert_use.is_held {
            layer_summary.held_expert_count += 1;
            if expert_use.use_count == 0 {
                layer_summary.held_never_used_expert_count += 1;
            }
        }
        if expert_use.use_count > 0 {
            layer_summary.used_expert_count += 1;
        }
    }
    layer_summary
}

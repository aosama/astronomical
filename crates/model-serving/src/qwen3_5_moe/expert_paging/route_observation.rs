//! Bounded history of true decode-time expert routes, captured for the
//! on-device route predictor program (#534, #536).
//!
//! Every generated token's native router already selects the ground-truth
//! experts for every sparse layer. This owner retains a bounded window of
//! those selections as labeled training examples: the input token identifier,
//! the preceding observed token's route, and this token's route. The ring
//! overwrites its oldest observation when full so the predictor always trains
//! on the most recent usage, and it reports stored and evicted totals so the
//! capture cost and turnover stay measurable.
//!
//! This module is intentionally free of Machine Learning framework types so
//! the record format and ring contract are provable hermetically.

use std::collections::VecDeque;

/// Sorted unique routed expert identifiers one sparse decoder layer selected
/// for one token. Quantized expert populations fit far below `u16::MAX`, so
/// the compact element keeps one token's complete route small enough for a
/// multi-thousand-observation resident history.
pub type LayerRoutedExpertIds = Vec<u16>;

/// One decode token's routed expert selection across all decoder layers in
/// layer order. `None` marks a layer that routed nothing (dense feed-forward
/// or an unobserved layer), which is itself part of the training label.
pub type ObservedExpertRoute = Vec<Option<LayerRoutedExpertIds>>;

/// One labeled training example for the expert-route predictor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteObservationRecord {
    /// The token identifier whose forward produced `token_route`.
    pub input_token_id: u32,
    /// The immediately preceding observed decode token's route, or `None`
    /// when this is the first observed token of its request.
    pub previous_token_route: Option<ObservedExpertRoute>,
    /// This token's true routed selection, the prediction label.
    pub token_route: ObservedExpertRoute,
}

/// Compacts one layer's raw routed expert identifiers into the sorted unique
/// form the history stores. Returns `None` when an identifier cannot fit the
/// compact element, which would indicate a router contract violation rather
/// than data to train on.
pub fn sorted_unique_layer_routed_expert_ids(
    raw_expert_ids: &[u32],
) -> Option<LayerRoutedExpertIds> {
    let mut compacted = Vec::with_capacity(raw_expert_ids.len());
    for raw_expert_id in raw_expert_ids {
        let expert_id = u16::try_from(*raw_expert_id).ok()?;
        compacted.push(expert_id);
    }
    compacted.sort_unstable();
    compacted.dedup();
    Some(compacted)
}

/// Bounded first-in-first-out history of route observations. Capacity is the
/// fixed token horizon; growth stops at capacity and the oldest observation
/// is evicted to admit the newest.
#[derive(Debug)]
pub struct RouteObservationRing {
    observations: VecDeque<RouteObservationRecord>,
    capacity: usize,
    stored_observation_count: u64,
    evicted_observation_count: u64,
}

impl RouteObservationRing {
    /// One request typically contributes hundreds of decode tokens; 2,048
    /// observations keep the most recent several conversations resident for
    /// training while staying a few megabytes even for large layer counts.
    pub const DEFAULT_OBSERVATION_CAPACITY: usize = 2_048;

    /// Creates a ring that retains the most recent `capacity` observations.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            observations: VecDeque::new(),
            capacity: capacity.max(1),
            stored_observation_count: 0,
            evicted_observation_count: 0,
        }
    }

    /// Stores one observation, evicting the oldest when the ring is full.
    /// Returns `true` when an older observation was evicted to make room.
    pub fn record_observation(&mut self, observation: RouteObservationRecord) -> bool {
        let evicted_oldest = self.observations.len() >= self.capacity;
        if evicted_oldest {
            self.observations.pop_front();
            self.evicted_observation_count = self.evicted_observation_count.saturating_add(1);
        }
        self.observations.push_back(observation);
        self.stored_observation_count = self.stored_observation_count.saturating_add(1);
        evicted_oldest
    }

    /// Observations currently retained, oldest first.
    pub fn observations(&self) -> impl Iterator<Item = &RouteObservationRecord> {
        self.observations.iter()
    }

    /// Observations currently retained.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Total observations ever stored, including evicted ones.
    pub fn stored_observation_count(&self) -> u64 {
        self.stored_observation_count
    }

    /// Total observations evicted by the capacity bound.
    pub fn evicted_observation_count(&self) -> u64 {
        self.evicted_observation_count
    }

    /// The token horizon this ring retains.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

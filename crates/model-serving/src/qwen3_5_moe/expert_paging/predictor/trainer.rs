//! Budget-bounded training slices over the route-observation history.
//!
//! The trainer is a function, not a thread owner: the serving side decides
//! when a slice runs, and this module drains as many observations as fit the
//! wall-time budget, in order, then yields. Every outcome is bounded — a slice
//! never runs unbounded, and a caller that passes a zero budget gets zero
//! steps. This shape keeps the trainer provable hermetically and keeps the
//! serving-side threading decision in the serving side, where the GPU and SSD
//! contention facts live.

use std::time::{Duration, Instant};

use crate::qwen3_5_moe::expert_paging::predictor::network::ExpertRoutePredictor;
use crate::qwen3_5_moe::expert_paging::route_observation::RouteObservationRing;

/// What one bounded training slice accomplished.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainerSliceOutcome {
    /// Observations consumed from the ring, oldest first.
    pub consumed_record_count: usize,
    /// Summed training loss over the consumed observations.
    pub summed_loss_millis: f32,
    /// Whether the budget ran out before the ring was drained.
    pub stopped_for_budget: bool,
}

/// Trains on observations until the ring is drained or the budget expires.
///
/// Observations are consumed oldest-first, matching the ring's eviction
/// order, so the trainer always spends its budget on the oldest examples and
/// leaves the newest for a later slice. A zero or elapsed budget consumes
/// nothing. The budget is checked between records, never inside one, so a
/// single record's work is atomic and the loss it returns is complete.
#[must_use]
pub fn train_predictor_slice(
    predictor: &mut ExpertRoutePredictor,
    ring: &mut RouteObservationRing,
    budget: Duration,
    started_at: Instant,
) -> TrainerSliceOutcome {
    if budget.is_zero() {
        return TrainerSliceOutcome {
            consumed_record_count: 0,
            summed_loss_millis: 0.0,
            stopped_for_budget: false,
        };
    }
    let mut consumed_record_count = 0_usize;
    let mut summed_loss = 0.0_f32;
    let mut stopped_for_budget = false;
    while ring.observation_count() > 0 {
        if started_at.elapsed() >= budget {
            stopped_for_budget = true;
            break;
        }
        let Some(observation) = ring.pop_oldest_observation() else {
            break;
        };
        summed_loss += predictor.train_on_record(&observation);
        consumed_record_count += 1;
    }
    TrainerSliceOutcome {
        consumed_record_count,
        summed_loss_millis: summed_loss,
        stopped_for_budget,
    }
}

/// Scores one record against the current weights, then applies one SGD step.
///
/// Scoring first is the honest online protocol: the hit rate is what a
/// prefetch would have seen if it had used this prediction before the native
/// router revealed the label.
#[must_use]
pub fn evaluate_then_train(
    predictor: &mut ExpertRoutePredictor,
    record: &crate::qwen3_5_moe::expert_paging::route_observation::RouteObservationRecord,
    top_k: usize,
) -> (u64, u64) {
    let accuracy =
        super::evaluate_predictor_accuracy(predictor, std::slice::from_ref(record), top_k);
    let hit_count: u64 = accuracy.layer_hit_counts.iter().sum();
    let evaluated_count: u64 = accuracy.layer_total_counts.iter().sum();
    predictor.train_on_record(record);
    (hit_count, evaluated_count)
}

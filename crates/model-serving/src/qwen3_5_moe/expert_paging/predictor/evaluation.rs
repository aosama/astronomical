//! Predictor accuracy measurement against held-out route observations.
//!
//! The number that matters is not raw accuracy but how much a prediction
//! would have helped: top-k hit rate says how often the true experts appear
//! among the k the predictor ranks highest, and coverage-over-baseline
//! compares that against simply reusing the previous token's route, which is
//! the zero-learned behavior any predictor must beat.

use crate::qwen3_5_moe::expert_paging::predictor::network::ExpertRoutePredictor;
use crate::qwen3_5_moe::expert_paging::route_observation::RouteObservationRecord;

/// Per-layer and aggregate top-k accuracy over a set of records.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PredictorLayerAccuracy {
    /// One entry per sparse layer: routed-expert hits and routed-expert
    /// totals at top-k. Layers with no labeled records report zero totals.
    pub layer_hit_counts: Vec<u64>,
    pub layer_total_counts: Vec<u64>,
}

impl PredictorLayerAccuracy {
    /// Top-k hit rate for one layer, or `None` when nothing was measured.
    #[must_use]
    pub fn layer_hit_rate(&self, layer_index: usize) -> Option<f64> {
        let total_count = *self.layer_total_counts.get(layer_index)?;
        if total_count == 0 {
            return None;
        }
        Some(
            f64::from(u32::try_from(self.layer_hit_counts[layer_index]).unwrap_or(0))
                / f64::from(u32::try_from(total_count).unwrap_or(0)),
        )
    }

    /// Aggregate top-k hit rate across every measured layer.
    #[must_use]
    pub fn overall_hit_rate(&self) -> Option<f64> {
        let total_count: u64 = self.layer_total_counts.iter().sum();
        if total_count == 0 {
            return None;
        }
        let hit_count: u64 = self.layer_hit_counts.iter().sum();
        Some(
            f64::from(u32::try_from(hit_count).unwrap_or(0))
                / f64::from(u32::try_from(total_count).unwrap_or(0)),
        )
    }
}

/// Measures top-k hit rate of the predictor against the held-out records.
///
/// A record's layer with no route is skipped: there is nothing to hit.
/// The previous token's route is supplied exactly as the observation stored
/// it, so evaluation sees the same inputs training saw.
#[must_use]
pub fn evaluate_predictor_accuracy(
    predictor: &ExpertRoutePredictor,
    records: &[RouteObservationRecord],
    top_k: usize,
) -> PredictorLayerAccuracy {
    let config = predictor.config();
    let mut accuracy = PredictorLayerAccuracy {
        layer_hit_counts: vec![0; config.layer_count],
        layer_total_counts: vec![0; config.layer_count],
    };
    for record in records {
        let logits_per_layer = predictor.forward_logits(
            record.input_token_id,
            record.previous_token_route.as_deref(),
        );
        for (layer_index, layer_label) in record.token_route.iter().enumerate() {
            let Some(routed_expert_ids) = layer_label else {
                continue;
            };
            let Some(layer_logits) = logits_per_layer.get(layer_index) else {
                continue;
            };
            let top_k_expert_ids = top_expert_ids(layer_logits, top_k);
            for routed_expert_id in routed_expert_ids.iter().copied() {
                accuracy.layer_total_counts[layer_index] += 1;
                if top_k_expert_ids.contains(&usize::from(routed_expert_id)) {
                    accuracy.layer_hit_counts[layer_index] += 1;
                }
            }
        }
    }
    accuracy
}

/// Indices of the `top_k` largest logits, highest first. Ties keep the lower
/// expert id first so measurement is deterministic.
fn top_expert_ids(logits: &[f32], top_k: usize) -> Vec<usize> {
    let mut expert_id_order: Vec<usize> = (0..logits.len()).collect();
    expert_id_order.sort_unstable_by(|left, right| {
        logits[*right]
            .partial_cmp(&logits[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    expert_id_order.truncate(top_k);
    expert_id_order
}

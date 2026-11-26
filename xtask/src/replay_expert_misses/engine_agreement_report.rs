//! Whether the replay lab stands for the engine: today's engine replayed
//! under the capacity it really had, held against the misses the engine
//! itself counted, in total and layer by layer.
//!
//! The lab's other rows are worth reading only while this one agrees within
//! `LAB_AGREEMENT_TOLERANCE`. When it does not, the layers it names are where
//! the lab's model of the engine is wrong.

use memory_policy::astro_expert_cache_model::{
    CapacityTrim, ReplayOutcome, SavedReplayCase, capacity_by_token, replay_under_capacity_schedule,
};

use super::capacity_divisions::divided_like_todays_engine;
use super::todays_engine_rule;

/// How far the lab's misses may stray from the engine's and still stand for
/// it, as a share of the engine's.
const LAB_AGREEMENT_TOLERANCE: f64 = 0.10;
/// Layers named when the lab and the engine disagree, the furthest first.
const DISAGREEING_LAYERS_REPORTED: usize = 6;

/// The engine's misses in each layer, over every recorded token.
pub(super) fn engine_misses_by_layer(engine_miss_counts_by_token: &[Vec<u32>]) -> Vec<u64> {
    let mut engine_misses_by_layer: Vec<u64> = Vec::new();
    for token_miss_counts in engine_miss_counts_by_token {
        if engine_misses_by_layer.len() < token_miss_counts.len() {
            engine_misses_by_layer.resize(token_miss_counts.len(), 0);
        }
        for (layer_index, &layer_miss_count) in token_miss_counts.iter().enumerate() {
            engine_misses_by_layer[layer_index] += u64::from(layer_miss_count);
        }
    }
    engine_misses_by_layer
}

/// The layers where the lab and the engine differ most, the furthest first:
/// `(layer_index, engine_misses, lab_misses)`.
pub(super) fn furthest_layers(
    engine_misses_by_layer: &[u64],
    lab_misses_by_layer: &[u64],
    layer_limit: usize,
) -> Vec<(usize, u64, u64)> {
    let layer_count = engine_misses_by_layer.len().max(lab_misses_by_layer.len());
    let mut layer_misses = Vec::with_capacity(layer_count);
    for layer_index in 0..layer_count {
        layer_misses.push((
            layer_index,
            engine_misses_by_layer
                .get(layer_index)
                .copied()
                .unwrap_or(0),
            lab_misses_by_layer.get(layer_index).copied().unwrap_or(0),
        ));
    }
    layer_misses.sort_by_key(|&(layer_index, engine_misses, lab_misses)| {
        (
            std::cmp::Reverse(engine_misses.abs_diff(lab_misses)),
            layer_index,
        )
    });
    layer_misses.truncate(layer_limit);
    layer_misses
}

/// The agreement rows for one saved case; none for a recording that did not
/// keep the engine's misses.
pub(super) fn report_engine_agreement(saved_replay_case: &SavedReplayCase) -> Vec<String> {
    let engine_miss_counts_by_token = &saved_replay_case.engine_miss_counts_by_token;
    if engine_miss_counts_by_token.is_empty() {
        return Vec::new();
    }
    let lab_outcome = replay_todays_engine_as_scheduled(saved_replay_case);
    let engine_misses_by_layer = engine_misses_by_layer(engine_miss_counts_by_token);
    let engine_miss_count: u64 = engine_misses_by_layer.iter().sum();
    let engine_misses_per_token =
        engine_miss_count as f64 / engine_miss_counts_by_token.len() as f64;
    let lab_misses_per_token = lab_outcome.misses_per_token();
    let lab_agrees = (lab_misses_per_token - engine_misses_per_token).abs()
        <= LAB_AGREEMENT_TOLERANCE * engine_misses_per_token;
    let mut report_lines = vec![format!(
        "[replay-expert-misses] workload={} engine_agreement engine_misses_per_token={engine_misses_per_token:.2} lab_misses_per_token={lab_misses_per_token:.2} lab_stands_for_engine={lab_agrees} recorded_tokens={}",
        saved_replay_case.workload_name,
        engine_miss_counts_by_token.len(),
    )];
    let engine_division = divided_like_todays_engine(saved_replay_case);
    for (layer_index, engine_misses, lab_misses) in furthest_layers(
        &engine_misses_by_layer,
        &lab_outcome.miss_count_by_layer,
        DISAGREEING_LAYERS_REPORTED,
    ) {
        report_lines.push(format!(
            "[replay-expert-misses] workload={} engine_agreement layer={layer_index} engine_misses={engine_misses} lab_misses={lab_misses} held_at_start={} table_slots={}",
            saved_replay_case.workload_name,
            engine_division
                .held_expert_ids_by_layer_at_start
                .get(layer_index)
                .map_or(0, Vec::len),
            engine_division
                .layer_slot_limits
                .as_ref()
                .and_then(|layer_slot_limits| layer_slot_limits.get(layer_index))
                .copied()
                .unwrap_or(0),
        ));
    }
    report_lines
}

/// Today's engine, in its layout and under the capacity it really had.
fn replay_todays_engine_as_scheduled(saved_replay_case: &SavedReplayCase) -> ReplayOutcome {
    let engine_division = divided_like_todays_engine(saved_replay_case);
    let recorded_schedule = capacity_by_token(
        &saved_replay_case.capacity_moments,
        engine_division.routed_reply.token_count(),
        engine_division.capacity_bytes,
    );
    let mut keeping_rule = todays_engine_rule(saved_replay_case);
    replay_under_capacity_schedule(
        &engine_division,
        &recorded_schedule,
        keeping_rule.as_mut(),
        CapacityTrim::DropLeastOccupiedLayer,
    )
    .replay_outcome
}

#[cfg(test)]
#[path = "engine_agreement_report_tests.rs"]
mod tests;

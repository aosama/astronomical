//! Iterative cumulative-latency planning across the remaining prompt episode.
//!
//! Reachable states are evaluated from smaller to larger remaining-token counts,
//! so each transition reads an already-computed suffix cost. Avoiding recursion
//! keeps one-token terminal-tail evidence safe for long context windows.

use std::collections::{BTreeMap, BTreeSet};

use super::PromptProcessingMeasurementContext;
use super::optimizer::CandidateChunkMeasurement;

/// Returns the first candidate in the lowest predicted-latency remaining episode.
pub(crate) fn lowest_cumulative_latency_candidate_index(
    candidate_chunk_size_tokens: &[usize],
    remaining_prompt_tokens: usize,
    measurement_context: PromptProcessingMeasurementContext,
    unknown_elapsed_millis_per_token: u128,
    measurements_for_candidate: &impl Fn(
        PromptProcessingMeasurementContext,
        usize,
    ) -> Vec<CandidateChunkMeasurement>,
) -> usize {
    let reachable_state_identifiers = collect_reachable_state_identifiers(
        candidate_chunk_size_tokens,
        remaining_prompt_tokens,
        measurement_context,
        measurements_for_candidate,
    );
    let mut predicted_cost_by_state = BTreeMap::new();
    let mut initial_planned_state = PlannedState {
        candidate_index: 0,
        predicted_elapsed_millis: 0,
        predicted_token_advancement: 0,
    };
    for (state_remaining_prompt_tokens, state_measurement_context) in reachable_state_identifiers {
        let planned_state = choose_candidate_for_state(
            candidate_chunk_size_tokens,
            state_remaining_prompt_tokens,
            state_measurement_context,
            unknown_elapsed_millis_per_token,
            measurements_for_candidate,
            &predicted_cost_by_state,
        );
        predicted_cost_by_state.insert(
            (state_remaining_prompt_tokens, state_measurement_context),
            planned_state.predicted_elapsed_millis,
        );
        if state_remaining_prompt_tokens == remaining_prompt_tokens
            && state_measurement_context == measurement_context
        {
            initial_planned_state = planned_state;
        }
    }
    initial_planned_state.candidate_index
}

#[derive(Clone, Copy)]
struct PlannedState {
    candidate_index: usize,
    predicted_elapsed_millis: u128,
    predicted_token_advancement: u128,
}

fn choose_candidate_for_state(
    candidate_chunk_size_tokens: &[usize],
    remaining_prompt_tokens: usize,
    measurement_context: PromptProcessingMeasurementContext,
    unknown_elapsed_millis_per_token: u128,
    measurements_for_candidate: &impl Fn(
        PromptProcessingMeasurementContext,
        usize,
    ) -> Vec<CandidateChunkMeasurement>,
    predicted_cost_by_state: &BTreeMap<(usize, PromptProcessingMeasurementContext), u128>,
) -> PlannedState {
    if remaining_prompt_tokens == 0 {
        return PlannedState {
            candidate_index: 0,
            predicted_elapsed_millis: 0,
            predicted_token_advancement: 0,
        };
    }

    let eligible_candidate_count =
        candidate_chunk_size_tokens.partition_point(|candidate_chunk_size_tokens| {
            *candidate_chunk_size_tokens <= remaining_prompt_tokens
        });
    let candidate_indices: Vec<usize> = if eligible_candidate_count == 0 {
        vec![0]
    } else {
        (0..eligible_candidate_count).collect()
    };
    let mut best_planned_state: Option<PlannedState> = None;

    for candidate_index in candidate_indices {
        let candidate_measurements =
            measurements_for_candidate(measurement_context, candidate_index);
        let selected_candidate_chunk_size_tokens = candidate_chunk_size_tokens[candidate_index];
        let planned_state = if candidate_measurements.is_empty() {
            PlannedState {
                candidate_index,
                predicted_elapsed_millis: unknown_elapsed_millis_per_token
                    .saturating_mul(remaining_prompt_tokens as u128),
                predicted_token_advancement: selected_candidate_chunk_size_tokens
                    .min(remaining_prompt_tokens)
                    as u128,
            }
        } else {
            let mut cumulative_elapsed_millis = 0_u128;
            let mut cumulative_token_advancement = 0_u128;
            for candidate_measurement in &candidate_measurements {
                let processed_prompt_token_count = candidate_measurement
                    .processed_prompt_token_count
                    .min(remaining_prompt_tokens);
                let next_remaining_prompt_tokens =
                    remaining_prompt_tokens.saturating_sub(processed_prompt_token_count);
                let next_state_cost = if next_remaining_prompt_tokens == 0 {
                    0
                } else {
                    predicted_cost_by_state
                        .get(&(
                            next_remaining_prompt_tokens,
                            candidate_measurement.next_measurement_context,
                        ))
                        .copied()
                        .unwrap_or(u128::MAX)
                };
                cumulative_elapsed_millis = cumulative_elapsed_millis.saturating_add(
                    u128::from(candidate_measurement.forward_elapsed_millis)
                        .saturating_add(next_state_cost),
                );
                cumulative_token_advancement = cumulative_token_advancement
                    .saturating_add(processed_prompt_token_count as u128);
            }
            let measurement_count = candidate_measurements.len() as u128;
            PlannedState {
                candidate_index,
                predicted_elapsed_millis: cumulative_elapsed_millis / measurement_count,
                predicted_token_advancement: cumulative_token_advancement / measurement_count,
            }
        };

        if best_planned_state.is_none_or(|current_best| {
            planned_state.predicted_elapsed_millis < current_best.predicted_elapsed_millis
                || (planned_state.predicted_elapsed_millis == current_best.predicted_elapsed_millis
                    && (planned_state.predicted_token_advancement
                        > current_best.predicted_token_advancement
                        || (planned_state.predicted_token_advancement
                            == current_best.predicted_token_advancement
                            && selected_candidate_chunk_size_tokens
                                > candidate_chunk_size_tokens[current_best.candidate_index])))
        }) {
            best_planned_state = Some(planned_state);
        }
    }

    best_planned_state.unwrap_or(PlannedState {
        candidate_index: 0,
        predicted_elapsed_millis: u128::MAX,
        predicted_token_advancement: 0,
    })
}

fn collect_reachable_state_identifiers(
    candidate_chunk_size_tokens: &[usize],
    remaining_prompt_tokens: usize,
    measurement_context: PromptProcessingMeasurementContext,
    measurements_for_candidate: &impl Fn(
        PromptProcessingMeasurementContext,
        usize,
    ) -> Vec<CandidateChunkMeasurement>,
) -> BTreeSet<(usize, PromptProcessingMeasurementContext)> {
    let initial_state_identifier = (remaining_prompt_tokens, measurement_context);
    let mut pending_state_identifiers = vec![initial_state_identifier];
    let mut reachable_state_identifiers = BTreeSet::new();

    while let Some((state_remaining_prompt_tokens, state_measurement_context)) =
        pending_state_identifiers.pop()
    {
        if state_remaining_prompt_tokens == 0
            || !reachable_state_identifiers
                .insert((state_remaining_prompt_tokens, state_measurement_context))
        {
            continue;
        }
        let eligible_candidate_count =
            candidate_chunk_size_tokens.partition_point(|candidate_chunk_size_tokens| {
                *candidate_chunk_size_tokens <= state_remaining_prompt_tokens
            });
        let candidate_indices: Vec<usize> = if eligible_candidate_count == 0 {
            vec![0]
        } else {
            (0..eligible_candidate_count).collect()
        };
        for candidate_index in candidate_indices {
            for candidate_measurement in
                measurements_for_candidate(state_measurement_context, candidate_index)
            {
                let processed_prompt_token_count = candidate_measurement
                    .processed_prompt_token_count
                    .min(state_remaining_prompt_tokens);
                let next_remaining_prompt_tokens =
                    state_remaining_prompt_tokens.saturating_sub(processed_prompt_token_count);
                if next_remaining_prompt_tokens < state_remaining_prompt_tokens {
                    pending_state_identifiers.push((
                        next_remaining_prompt_tokens,
                        candidate_measurement.next_measurement_context,
                    ));
                }
            }
        }
    }

    reachable_state_identifiers
}

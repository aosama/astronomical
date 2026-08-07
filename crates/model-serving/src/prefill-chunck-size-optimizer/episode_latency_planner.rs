use std::collections::{BTreeMap, BTreeSet};

use super::PrefillChunckSizeOptimizerContext;
use super::optimizer::CandidatePrefillChunckObservation;

pub(crate) fn lowest_cumulative_latency_candidate_index(
    candidate_prefill_chunck_tokens: &[usize],
    remaining_prompt_tokens: usize,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    unknown_elapsed_millis_per_token: u128,
    observations_for_action: &impl Fn(
        PrefillChunckSizeOptimizerContext,
        usize,
    ) -> Vec<CandidatePrefillChunckObservation>,
) -> usize {
    let reachable_state_identifiers = collect_reachable_state_identifiers(
        candidate_prefill_chunck_tokens,
        remaining_prompt_tokens,
        prompt_processing_context,
        observations_for_action,
    );
    let mut predicted_cost_by_state = BTreeMap::new();
    let mut initial_planned_state = PlannedState {
        candidate_index: 0,
        predicted_elapsed_millis: 0,
        predicted_token_advancement: 0,
    };
    for (state_remaining_prompt_tokens, state_prompt_processing_context) in
        reachable_state_identifiers
    {
        let planned_state = choose_candidate_for_state(
            candidate_prefill_chunck_tokens,
            state_remaining_prompt_tokens,
            state_prompt_processing_context,
            unknown_elapsed_millis_per_token,
            observations_for_action,
            &predicted_cost_by_state,
        );
        predicted_cost_by_state.insert(
            (
                state_remaining_prompt_tokens,
                state_prompt_processing_context,
            ),
            planned_state.predicted_elapsed_millis,
        );
        if state_remaining_prompt_tokens == remaining_prompt_tokens
            && state_prompt_processing_context == prompt_processing_context
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
    candidate_prefill_chunck_tokens: &[usize],
    remaining_prompt_tokens: usize,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    unknown_elapsed_millis_per_token: u128,
    observations_for_action: &impl Fn(
        PrefillChunckSizeOptimizerContext,
        usize,
    ) -> Vec<CandidatePrefillChunckObservation>,
    predicted_cost_by_state: &BTreeMap<(usize, PrefillChunckSizeOptimizerContext), u128>,
) -> PlannedState {
    if remaining_prompt_tokens == 0 {
        return PlannedState {
            candidate_index: 0,
            predicted_elapsed_millis: 0,
            predicted_token_advancement: 0,
        };
    }

    let eligible_candidate_count =
        candidate_prefill_chunck_tokens.partition_point(|candidate_prefill_chunck_tokens| {
            *candidate_prefill_chunck_tokens <= remaining_prompt_tokens
        });
    let candidate_indices: Vec<usize> = if eligible_candidate_count == 0 {
        vec![0]
    } else {
        (0..eligible_candidate_count).collect()
    };
    let mut best_planned_state: Option<PlannedState> = None;

    for candidate_index in candidate_indices {
        let candidate_observations =
            observations_for_action(prompt_processing_context, candidate_index);
        let requested_prefill_chunck_tokens = candidate_prefill_chunck_tokens[candidate_index];
        let planned_state = if candidate_observations.is_empty() {
            PlannedState {
                candidate_index,
                predicted_elapsed_millis: unknown_elapsed_millis_per_token
                    .saturating_mul(remaining_prompt_tokens as u128),
                predicted_token_advancement: requested_prefill_chunck_tokens
                    .min(remaining_prompt_tokens)
                    as u128,
            }
        } else {
            let mut cumulative_elapsed_millis = 0_u128;
            let mut cumulative_token_advancement = 0_u128;
            for candidate_observation in &candidate_observations {
                let actual_prefill_chunck_tokens = candidate_observation
                    .actual_prefill_chunck_tokens
                    .min(remaining_prompt_tokens);
                let next_remaining_prompt_tokens =
                    remaining_prompt_tokens.saturating_sub(actual_prefill_chunck_tokens);
                let next_state_cost = if next_remaining_prompt_tokens == 0 {
                    0
                } else {
                    predicted_cost_by_state
                        .get(&(
                            next_remaining_prompt_tokens,
                            candidate_observation.next_prompt_processing_context,
                        ))
                        .copied()
                        .unwrap_or(u128::MAX)
                };
                cumulative_elapsed_millis = cumulative_elapsed_millis.saturating_add(
                    u128::from(candidate_observation.elapsed_millis)
                        .saturating_add(next_state_cost),
                );
                cumulative_token_advancement = cumulative_token_advancement
                    .saturating_add(actual_prefill_chunck_tokens as u128);
            }
            let observation_count = candidate_observations.len() as u128;
            PlannedState {
                candidate_index,
                predicted_elapsed_millis: cumulative_elapsed_millis / observation_count,
                predicted_token_advancement: cumulative_token_advancement / observation_count,
            }
        };

        if best_planned_state.is_none_or(|current_best| {
            planned_state.predicted_elapsed_millis < current_best.predicted_elapsed_millis
                || (planned_state.predicted_elapsed_millis == current_best.predicted_elapsed_millis
                    && (planned_state.predicted_token_advancement
                        > current_best.predicted_token_advancement
                        || (planned_state.predicted_token_advancement
                            == current_best.predicted_token_advancement
                            && requested_prefill_chunck_tokens
                                > candidate_prefill_chunck_tokens[current_best.candidate_index])))
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
    candidate_prefill_chunck_tokens: &[usize],
    remaining_prompt_tokens: usize,
    prompt_processing_context: PrefillChunckSizeOptimizerContext,
    observations_for_action: &impl Fn(
        PrefillChunckSizeOptimizerContext,
        usize,
    ) -> Vec<CandidatePrefillChunckObservation>,
) -> BTreeSet<(usize, PrefillChunckSizeOptimizerContext)> {
    let initial_state_identifier = (remaining_prompt_tokens, prompt_processing_context);
    let mut pending_state_identifiers = vec![initial_state_identifier];
    let mut reachable_state_identifiers = BTreeSet::new();

    while let Some((state_remaining_prompt_tokens, state_prompt_processing_context)) =
        pending_state_identifiers.pop()
    {
        if state_remaining_prompt_tokens == 0
            || !reachable_state_identifiers.insert((
                state_remaining_prompt_tokens,
                state_prompt_processing_context,
            ))
        {
            continue;
        }
        let eligible_candidate_count =
            candidate_prefill_chunck_tokens.partition_point(|candidate_prefill_chunck_tokens| {
                *candidate_prefill_chunck_tokens <= state_remaining_prompt_tokens
            });
        let candidate_indices: Vec<usize> = if eligible_candidate_count == 0 {
            vec![0]
        } else {
            (0..eligible_candidate_count).collect()
        };
        for candidate_index in candidate_indices {
            for candidate_observation in
                observations_for_action(state_prompt_processing_context, candidate_index)
            {
                let actual_prefill_chunck_tokens = candidate_observation
                    .actual_prefill_chunck_tokens
                    .min(state_remaining_prompt_tokens);
                let next_remaining_prompt_tokens =
                    state_remaining_prompt_tokens.saturating_sub(actual_prefill_chunck_tokens);
                if next_remaining_prompt_tokens < state_remaining_prompt_tokens {
                    pending_state_identifiers.push((
                        next_remaining_prompt_tokens,
                        candidate_observation.next_prompt_processing_context,
                    ));
                }
            }
        }
    }

    reachable_state_identifiers
}

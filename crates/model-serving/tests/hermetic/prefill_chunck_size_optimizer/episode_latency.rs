use super::*;

#[test]
fn should_prefer_lower_episode_latency_over_higher_immediate_throughput() {
    let mut prefill_chunck_size_optimizer =
        PrefillChunckSizeOptimizer::new(vec![4, 6], 5).expect("candidate set should be valid");
    let starting_context = PrefillChunckSizeOptimizerContext::new(70);
    let four_token_destination_context = PrefillChunckSizeOptimizerContext::new(71);
    let six_token_destination_context = PrefillChunckSizeOptimizerContext::new(72);

    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        starting_context,
        4,
        4,
        3,
        four_token_destination_context,
    );
    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        starting_context,
        6,
        6,
        2,
        six_token_destination_context,
    );
    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        four_token_destination_context,
        4,
        4,
        1,
        four_token_destination_context,
    );
    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        six_token_destination_context,
        4,
        2,
        10,
        six_token_destination_context,
    );

    let decision =
        prefill_chunck_size_optimizer.ask_with_maximum_prefill_chunck_tokens(starting_context, 8);
    assert_eq!(
        decision.candidate_prefill_chunck_tokens, 4,
        "the 6-token action has better immediate throughput but a slower complete path"
    );
}

#[test]
fn should_break_equal_episode_cost_ties_toward_greater_advancement() {
    let mut prefill_chunck_size_optimizer =
        PrefillChunckSizeOptimizer::new(vec![4, 8], 5).expect("candidate set should be valid");
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(73);
    record_self_transition_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        4,
        &[5],
    );
    record_self_transition_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        8,
        &[10],
    );

    let decision = prefill_chunck_size_optimizer
        .ask_with_maximum_prefill_chunck_tokens(prompt_processing_context, 8);
    assert_eq!(decision.candidate_prefill_chunck_tokens, 8);
}

#[test]
fn should_use_the_smallest_candidate_for_a_short_terminal_tail() {
    let mut prefill_chunck_size_optimizer =
        PrefillChunckSizeOptimizer::new(vec![128, 256], 5).expect("candidate set should be valid");
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(74);
    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        128,
        64,
        20,
        prompt_processing_context,
    );

    let decision = prefill_chunck_size_optimizer
        .ask_with_maximum_prefill_chunck_tokens(prompt_processing_context, 64);
    assert_eq!(decision.candidate_prefill_chunck_tokens, 128);
}

#[test]
fn should_plan_a_long_episode_after_a_one_token_terminal_tail() {
    let mut prefill_chunck_size_optimizer =
        PrefillChunckSizeOptimizer::new(vec![1_024], 5).expect("candidate set should be valid");
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(75);
    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        1_024,
        1,
        1,
        prompt_processing_context,
    );

    let decision = prefill_chunck_size_optimizer
        .ask_with_maximum_prefill_chunck_tokens(prompt_processing_context, 68_576);

    assert_eq!(decision.candidate_prefill_chunck_tokens, 1_024);
    assert_eq!(
        decision.reason,
        PrefillChunckSizeOptimizerDecisionReason::CumulativeLatencyPlanning
    );
}

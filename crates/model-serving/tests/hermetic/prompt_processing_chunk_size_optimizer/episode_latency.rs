use super::*;

#[test]
fn should_prefer_lower_episode_latency_over_higher_immediate_throughput() {
    let mut chunk_size_optimizer = PromptProcessingChunkSizeOptimizer::new(vec![4, 6], 5)
        .expect("candidate set should be valid");
    let starting_context = PromptProcessingMeasurementContext::isolated(70);
    let four_token_destination_context = PromptProcessingMeasurementContext::isolated(71);
    let six_token_destination_context = PromptProcessingMeasurementContext::isolated(72);

    record_chunk_measurement(
        &mut chunk_size_optimizer,
        starting_context,
        4,
        4,
        3,
        four_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        starting_context,
        4,
        4,
        3,
        four_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        starting_context,
        6,
        6,
        2,
        six_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        starting_context,
        6,
        6,
        2,
        six_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        four_token_destination_context,
        4,
        4,
        1,
        four_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        four_token_destination_context,
        4,
        4,
        1,
        four_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        six_token_destination_context,
        4,
        4,
        10,
        six_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        six_token_destination_context,
        4,
        4,
        10,
        six_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        four_token_destination_context,
        6,
        6,
        100,
        four_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        four_token_destination_context,
        6,
        6,
        100,
        four_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        six_token_destination_context,
        6,
        6,
        10,
        six_token_destination_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        six_token_destination_context,
        6,
        6,
        10,
        six_token_destination_context,
    );

    let selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(starting_context, 8);
    assert_eq!(
        selection.selected_candidate_chunk_size_tokens, 4,
        "the 6-token action has better immediate throughput but a slower complete path"
    );
}

#[test]
fn should_break_equal_episode_cost_ties_toward_greater_advancement() {
    let mut chunk_size_optimizer = PromptProcessingChunkSizeOptimizer::new(vec![4, 8], 5)
        .expect("candidate set should be valid");
    let measurement_context = PromptProcessingMeasurementContext::isolated(73);
    record_same_context_measurements(&mut chunk_size_optimizer, measurement_context, 4, &[5, 5]);
    record_same_context_measurements(&mut chunk_size_optimizer, measurement_context, 8, &[10, 10]);

    let selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(measurement_context, 8);
    assert_eq!(selection.selected_candidate_chunk_size_tokens, 8);
}

#[test]
fn should_use_the_smallest_candidate_for_a_short_terminal_tail() {
    let mut chunk_size_optimizer = PromptProcessingChunkSizeOptimizer::new(vec![128, 256], 5)
        .expect("candidate set should be valid");
    let measurement_context = PromptProcessingMeasurementContext::isolated(74);

    let selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(measurement_context, 64);
    assert_eq!(selection.selected_candidate_chunk_size_tokens, 128);
}

#[test]
fn should_reject_one_token_terminal_tail_evidence() {
    let mut chunk_size_optimizer = PromptProcessingChunkSizeOptimizer::new(vec![1_024], 5)
        .expect("candidate set should be valid");
    let measurement_context = PromptProcessingMeasurementContext::isolated(75);
    let measurement_error = chunk_size_optimizer
        .record_measurement(
            measurement_context,
            1_024,
            PromptProcessingChunkMeasurement::transition(1, 1, measurement_context),
        )
        .expect_err("a one-token tail must not become full-candidate evidence");
    assert!(
        measurement_error
            .to_string()
            .contains("selected 1024, processed 1")
    );

    let selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(measurement_context, 68_576);

    assert_eq!(selection.selected_candidate_chunk_size_tokens, 1_024);
    assert_eq!(
        selection.reason,
        PromptProcessingChunkSizeSelectionReason::ExploreUnmeasuredCandidate
    );
}

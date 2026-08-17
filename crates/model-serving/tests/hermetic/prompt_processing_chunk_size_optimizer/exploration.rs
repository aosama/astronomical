use super::*;

#[test]
fn should_explore_largest_eligible_unmeasured_candidate_first() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let measurement_context = PromptProcessingMeasurementContext::isolated(7);

    let first_selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(measurement_context, 700);
    assert_eq!(first_selection.selected_candidate_chunk_size_tokens, 512);
    assert_eq!(
        first_selection.reason,
        PromptProcessingChunkSizeSelectionReason::ExploreUnmeasuredCandidate
    );

    record_chunk_measurement(
        &mut chunk_size_optimizer,
        measurement_context,
        512,
        512,
        400,
        measurement_context,
    );

    let second_selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(measurement_context, 700);
    assert_eq!(second_selection.selected_candidate_chunk_size_tokens, 256);
    assert_eq!(
        second_selection.reason,
        PromptProcessingChunkSizeSelectionReason::ExploreUnmeasuredCandidate
    );
}

#[test]
fn should_reverse_candidate_order_for_the_second_exploration_pass() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let measurement_context = PromptProcessingMeasurementContext::isolated(8);

    for expected_candidate_chunk_size_tokens in [1_024, 512, 256, 256, 512, 1_024] {
        let selection = chunk_size_optimizer
            .select_candidate_chunk_size_with_maximum(measurement_context, 1_024);
        assert_eq!(
            selection.selected_candidate_chunk_size_tokens,
            expected_candidate_chunk_size_tokens
        );
        assert_eq!(
            selection.reason,
            PromptProcessingChunkSizeSelectionReason::ExploreUnmeasuredCandidate
        );
        record_chunk_measurement(
            &mut chunk_size_optimizer,
            measurement_context,
            expected_candidate_chunk_size_tokens,
            expected_candidate_chunk_size_tokens,
            100,
            measurement_context,
        );
    }

    let converged_selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(measurement_context, 1_024);
    assert_eq!(
        converged_selection.reason,
        PromptProcessingChunkSizeSelectionReason::MinimizeProjectedRemainingPromptLatency
    );
}

#[test]
fn should_not_treat_an_ineligible_candidate_as_missing_measurements() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let measurement_context = PromptProcessingMeasurementContext::isolated(11);

    record_same_context_measurements(
        &mut chunk_size_optimizer,
        measurement_context,
        256,
        &[300, 300],
    );
    record_same_context_measurements(
        &mut chunk_size_optimizer,
        measurement_context,
        512,
        &[400, 400],
    );

    let selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(measurement_context, 700);
    assert_eq!(
        selection.reason,
        PromptProcessingChunkSizeSelectionReason::MinimizeProjectedRemainingPromptLatency
    );
}

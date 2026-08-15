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
fn should_not_treat_an_ineligible_candidate_as_missing_measurements() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let measurement_context = PromptProcessingMeasurementContext::isolated(11);

    record_same_context_measurements(&mut chunk_size_optimizer, measurement_context, 256, &[300]);
    record_same_context_measurements(&mut chunk_size_optimizer, measurement_context, 512, &[400]);

    let selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(measurement_context, 700);
    assert_eq!(
        selection.reason,
        PromptProcessingChunkSizeSelectionReason::MinimizeProjectedRemainingPromptLatency
    );
}

#[test]
fn should_refresh_an_eligible_candidate_after_its_measurement_becomes_stale() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let measurement_context = PromptProcessingMeasurementContext::isolated(13);
    for candidate_chunk_size_tokens in [256, 512, 1_024] {
        record_same_context_measurements(
            &mut chunk_size_optimizer,
            measurement_context,
            candidate_chunk_size_tokens,
            &[candidate_chunk_size_tokens as u64],
        );
    }

    let mut selected_stale_measurement_refresh = false;
    for _selection_index in 0..20 {
        let selection = chunk_size_optimizer.select_candidate_chunk_size(measurement_context);
        if selection.reason
            == PromptProcessingChunkSizeSelectionReason::RefreshStaleCandidateMeasurement
        {
            selected_stale_measurement_refresh = true;
            break;
        }
    }

    assert!(
        selected_stale_measurement_refresh,
        "an eligible candidate should refresh after five times the candidate count selections"
    );
}

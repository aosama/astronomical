use super::*;

#[test]
fn should_refresh_stale_measurement_without_resetting_other_candidate_measurements() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let measurement_context = PromptProcessingMeasurementContext::isolated(61);
    for candidate_chunk_size_tokens in [256, 512, 1_024] {
        record_same_context_measurements(
            &mut chunk_size_optimizer,
            measurement_context,
            candidate_chunk_size_tokens,
            &[candidate_chunk_size_tokens as u64],
        );
    }

    let stale_measurement_selection = (0..20)
        .find_map(|_selection_index| {
            let selection = chunk_size_optimizer.select_candidate_chunk_size(measurement_context);
            (selection.reason
                == PromptProcessingChunkSizeSelectionReason::RefreshStaleCandidateMeasurement)
                .then_some(selection)
        })
        .expect("one candidate should become stale");

    record_chunk_measurement(
        &mut chunk_size_optimizer,
        measurement_context,
        stale_measurement_selection.selected_candidate_chunk_size_tokens,
        stale_measurement_selection.selected_candidate_chunk_size_tokens,
        stale_measurement_selection.selected_candidate_chunk_size_tokens as u64,
        measurement_context,
    );

    let next_selection = chunk_size_optimizer.select_candidate_chunk_size(measurement_context);
    assert_ne!(
        next_selection.reason,
        PromptProcessingChunkSizeSelectionReason::ExploreUnmeasuredCandidate,
        "measurement refresh must not discard measurements for every candidate"
    );
}

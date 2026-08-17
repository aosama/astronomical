use super::*;

#[test]
fn should_reject_a_memory_reduced_measurement() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let unconstrained_context = PromptProcessingMeasurementContext::isolated(47);
    let capacity_reduced_context = PromptProcessingMeasurementContext::isolated(48);

    record_same_context_measurements(
        &mut chunk_size_optimizer,
        unconstrained_context,
        256,
        &[300],
    );
    record_same_context_measurements(
        &mut chunk_size_optimizer,
        unconstrained_context,
        512,
        &[450],
    );
    let measurement_error = chunk_size_optimizer
        .record_measurement(
            unconstrained_context,
            1_024,
            PromptProcessingChunkMeasurement::transition(512, 2_000, capacity_reduced_context),
        )
        .expect_err("memory-reduced work must not become candidate evidence");
    assert!(
        measurement_error
            .to_string()
            .contains("selected 1024, processed 512")
    );

    let selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(unconstrained_context, 1_024);
    assert_eq!(selection.selected_candidate_chunk_size_tokens, 1_024);
    assert_eq!(
        selection.reason,
        PromptProcessingChunkSizeSelectionReason::ExploreUnmeasuredCandidate
    );
}

#[test]
fn should_reject_a_final_prompt_segment_measurement() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let measurement_context = PromptProcessingMeasurementContext::isolated(49);

    let measurement_error = chunk_size_optimizer
        .record_measurement(
            measurement_context,
            256,
            PromptProcessingChunkMeasurement::transition(64, 80, measurement_context),
        )
        .expect_err("a terminal tail must not become full-candidate evidence");
    assert!(
        measurement_error
            .to_string()
            .contains("selected 256, processed 64")
    );

    let selection =
        chunk_size_optimizer.select_candidate_chunk_size_with_maximum(measurement_context, 64);
    assert_eq!(selection.selected_candidate_chunk_size_tokens, 256);
    assert_eq!(
        selection.reason,
        PromptProcessingChunkSizeSelectionReason::RemainingTokensBelowSmallestCandidate
    );
}

#[test]
fn should_reject_a_measurement_without_processed_prompt_tokens() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let measurement_context = PromptProcessingMeasurementContext::isolated(51);

    let measurement_error = chunk_size_optimizer
        .record_measurement(
            measurement_context,
            256,
            PromptProcessingChunkMeasurement::transition(0, 100, measurement_context),
        )
        .expect_err("zero processed prompt tokens must be rejected");
    assert!(measurement_error.to_string().contains("must be positive"));
}

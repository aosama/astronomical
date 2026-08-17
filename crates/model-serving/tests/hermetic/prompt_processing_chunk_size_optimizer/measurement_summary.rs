use super::*;

#[test]
fn should_pool_measurements_across_positions_in_the_same_execution_profile() {
    let mut chunk_size_optimizer = PromptProcessingChunkSizeOptimizer::new(vec![256, 512], 5)
        .expect("candidate set should be valid");
    let first_position_context =
        PromptProcessingMeasurementContext::with_position_independent_execution_profile(1, 99);
    let second_position_context =
        PromptProcessingMeasurementContext::with_position_independent_execution_profile(2, 99);
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        first_position_context,
        256,
        256,
        100,
        second_position_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        first_position_context,
        512,
        512,
        200,
        second_position_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        second_position_context,
        256,
        256,
        150,
        second_position_context,
    );
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        second_position_context,
        512,
        512,
        200,
        second_position_context,
    );

    let measurement_summaries =
        chunk_size_optimizer.candidate_measurement_summaries(first_position_context);

    assert!(measurement_summaries.all_candidates_have_measurements);
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries.len(),
        2
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[0].candidate_chunk_size_tokens,
        256
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[0].measurement_source,
        CandidateMeasurementSource::ExecutionProfile
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[0].measurement_count,
        2
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[0]
            .average_processed_prompt_token_count,
        256
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[0].average_forward_elapsed_millis,
        125
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[1].measurement_source,
        CandidateMeasurementSource::ExecutionProfile
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[1].measurement_count,
        2
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[1]
            .average_processed_prompt_token_count,
        512
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[1].average_forward_elapsed_millis,
        200
    );
}

#[test]
fn should_report_missing_measurements_without_inventing_values() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    let measurement_context = PromptProcessingMeasurementContext::isolated(7);
    record_same_context_measurements(
        &mut chunk_size_optimizer,
        measurement_context,
        1_024,
        &[300],
    );

    let measurement_summaries =
        chunk_size_optimizer.candidate_measurement_summaries(measurement_context);

    assert!(!measurement_summaries.all_candidates_have_measurements);
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[0].measurement_source,
        CandidateMeasurementSource::NoMeasurementsAvailable
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[0].measurement_count,
        0
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[0].average_forward_elapsed_millis,
        0
    );
    assert_eq!(
        measurement_summaries.candidate_measurement_summaries[2].measurement_count,
        1
    );
}

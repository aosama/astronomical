use super::*;

#[test]
fn should_prefer_exact_candidate_evidence_and_fall_back_to_the_context_family() {
    let mut prefill_chunck_size_optimizer =
        PrefillChunckSizeOptimizer::new(vec![256, 512], 5).expect("candidate set should be valid");
    let first_position_context = PrefillChunckSizeOptimizerContext::new_with_fallback(1, 99);
    let second_position_context = PrefillChunckSizeOptimizerContext::new_with_fallback(2, 99);
    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        first_position_context,
        256,
        256,
        100,
        second_position_context,
    );
    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        second_position_context,
        256,
        128,
        150,
        second_position_context,
    );
    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        second_position_context,
        512,
        512,
        200,
        second_position_context,
    );

    let context_evidence = prefill_chunck_size_optimizer.context_evidence(first_position_context);

    assert!(context_evidence.has_observations_for_every_candidate);
    assert_eq!(context_evidence.candidate_evidence.len(), 2);
    assert_eq!(
        context_evidence.candidate_evidence[0].candidate_prefill_chunck_tokens,
        256
    );
    assert_eq!(context_evidence.candidate_evidence[0].observation_count, 1);
    assert_eq!(
        context_evidence.candidate_evidence[0].average_actual_prefill_chunck_tokens,
        256
    );
    assert_eq!(
        context_evidence.candidate_evidence[0].average_elapsed_millis,
        100
    );
    assert_eq!(context_evidence.candidate_evidence[1].observation_count, 1);
    assert_eq!(
        context_evidence.candidate_evidence[1].average_actual_prefill_chunck_tokens,
        512
    );
    assert_eq!(
        context_evidence.candidate_evidence[1].average_elapsed_millis,
        200
    );
}

#[test]
fn should_report_incomplete_evidence_without_inventing_candidate_measurements() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(7);
    record_self_transition_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        1_024,
        &[300],
    );

    let context_evidence =
        prefill_chunck_size_optimizer.context_evidence(prompt_processing_context);

    assert!(!context_evidence.has_observations_for_every_candidate);
    assert_eq!(context_evidence.candidate_evidence[0].observation_count, 0);
    assert_eq!(
        context_evidence.candidate_evidence[0].average_elapsed_millis,
        0
    );
    assert_eq!(context_evidence.candidate_evidence[2].observation_count, 1);
}

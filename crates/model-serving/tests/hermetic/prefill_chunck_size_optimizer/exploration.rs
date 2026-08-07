use super::*;

#[test]
fn should_explore_largest_eligible_unobserved_candidate_first() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(7);

    let first_decision = prefill_chunck_size_optimizer
        .ask_with_maximum_prefill_chunck_tokens(prompt_processing_context, 700);
    assert_eq!(first_decision.candidate_prefill_chunck_tokens, 512);
    assert_eq!(
        first_decision.reason,
        PrefillChunckSizeOptimizerDecisionReason::InitialExploration
    );

    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        512,
        512,
        400,
        prompt_processing_context,
    );

    let second_decision = prefill_chunck_size_optimizer
        .ask_with_maximum_prefill_chunck_tokens(prompt_processing_context, 700);
    assert_eq!(second_decision.candidate_prefill_chunck_tokens, 256);
    assert_eq!(
        second_decision.reason,
        PrefillChunckSizeOptimizerDecisionReason::InitialExploration
    );
}

#[test]
fn should_not_treat_an_ineligible_candidate_as_missing_evidence() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(11);

    record_self_transition_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        256,
        &[300],
    );
    record_self_transition_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        512,
        &[400],
    );

    let decision = prefill_chunck_size_optimizer
        .ask_with_maximum_prefill_chunck_tokens(prompt_processing_context, 700);
    assert_eq!(
        decision.reason,
        PrefillChunckSizeOptimizerDecisionReason::CumulativeLatencyPlanning
    );
}

#[test]
fn should_probe_an_eligible_candidate_after_its_observation_becomes_stale() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(13);
    for candidate_prefill_chunck_tokens in [256, 512, 1_024] {
        record_self_transition_observations(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context,
            candidate_prefill_chunck_tokens,
            &[candidate_prefill_chunck_tokens as u64],
        );
    }

    let mut observed_stale_probe = false;
    for _decision_index in 0..20 {
        let decision = prefill_chunck_size_optimizer.ask(prompt_processing_context);
        if decision.reason == PrefillChunckSizeOptimizerDecisionReason::StaleObservationProbe {
            observed_stale_probe = true;
            break;
        }
    }

    assert!(
        observed_stale_probe,
        "an eligible candidate should be probed after five times the candidate count decisions"
    );
}

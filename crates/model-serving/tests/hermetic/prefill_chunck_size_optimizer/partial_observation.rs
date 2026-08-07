use super::*;

#[test]
fn should_retain_a_memory_reduced_transition_under_the_requested_action() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let unconstrained_context = PrefillChunckSizeOptimizerContext::new(47);
    let capacity_reduced_context = PrefillChunckSizeOptimizerContext::new(48);

    record_self_transition_observations(
        &mut prefill_chunck_size_optimizer,
        unconstrained_context,
        256,
        &[300],
    );
    record_self_transition_observations(
        &mut prefill_chunck_size_optimizer,
        unconstrained_context,
        512,
        &[450],
    );
    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        unconstrained_context,
        1_024,
        512,
        2_000,
        capacity_reduced_context,
    );

    let decision = prefill_chunck_size_optimizer
        .ask_with_maximum_prefill_chunck_tokens(unconstrained_context, 1_024);
    assert_eq!(decision.candidate_prefill_chunck_tokens, 512);
}

#[test]
fn should_retain_a_prompt_tail_transition() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(49);

    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        256,
        64,
        80,
        prompt_processing_context,
    );

    let decision = prefill_chunck_size_optimizer
        .ask_with_maximum_prefill_chunck_tokens(prompt_processing_context, 64);
    assert_eq!(decision.candidate_prefill_chunck_tokens, 256);
    assert_eq!(
        decision.reason,
        PrefillChunckSizeOptimizerDecisionReason::Fallback
    );
}

#[test]
fn should_reject_a_transition_without_token_advancement() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(51);

    let transition_error = prefill_chunck_size_optimizer
        .tell(
            prompt_processing_context,
            256,
            PrefillChunckSizeOptimizerObservation::transition(0, 100, prompt_processing_context),
        )
        .expect_err("zero token advancement must be rejected");
    assert!(transition_error.to_string().contains("must be positive"));
}

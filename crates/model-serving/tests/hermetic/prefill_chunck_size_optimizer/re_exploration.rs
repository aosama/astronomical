use super::*;

#[test]
fn should_report_stale_probe_without_resetting_all_candidate_evidence() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(61);
    for candidate_prefill_chunck_tokens in [256, 512, 1_024] {
        record_self_transition_observations(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context,
            candidate_prefill_chunck_tokens,
            &[candidate_prefill_chunck_tokens as u64],
        );
    }

    let stale_probe_decision = (0..20)
        .find_map(|_decision_index| {
            let decision = prefill_chunck_size_optimizer.ask(prompt_processing_context);
            (decision.reason == PrefillChunckSizeOptimizerDecisionReason::StaleObservationProbe)
                .then_some(decision)
        })
        .expect("one candidate should become stale");

    record_transition_observation(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        stale_probe_decision.candidate_prefill_chunck_tokens,
        stale_probe_decision.candidate_prefill_chunck_tokens,
        stale_probe_decision.candidate_prefill_chunck_tokens as u64,
        prompt_processing_context,
    );

    let next_decision = prefill_chunck_size_optimizer.ask(prompt_processing_context);
    assert_ne!(
        next_decision.reason,
        PrefillChunckSizeOptimizerDecisionReason::InitialExploration,
        "stale probing must not discard evidence for every candidate"
    );
}

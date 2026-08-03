use super::*;

#[test]
fn should_explore_each_candidate_until_it_has_three_observations() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(7);

    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        256,
        &[1_000, 1_000],
    );
    assert_eq!(
        ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context
        ),
        256,
        "256 should still be explored until it reaches three trusted observations"
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        256,
        &[1_000],
    );

    assert_eq!(
        ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context
        ),
        512,
        "after 256 reaches three observations the next untested candidate should be explored"
    );
}

#[test]
fn should_interleave_candidate_exploration_until_all_candidates_are_trusted() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(11);
    let expected_candidate_sequence = [256, 512, 1_024, 256, 512, 1_024, 256, 512, 1_024];

    for expected_candidate_prefill_chunck_tokens in expected_candidate_sequence {
        let selected_candidate_prefill_chunck_tokens = ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context,
        );
        assert_eq!(
            selected_candidate_prefill_chunck_tokens,
            expected_candidate_prefill_chunck_tokens
        );
        record_full_observations(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context,
            selected_candidate_prefill_chunck_tokens,
            &[1_000],
        );
    }
}

#[test]
fn should_not_trust_a_candidate_with_fewer_than_three_observations() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(13);

    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        256,
        &[100],
    );
    assert_eq!(
        ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context
        ),
        256,
        "a candidate with fewer than three observations should still be explored, not skipped"
    );
}

#[test]
fn should_report_exploration_reason_before_trust() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(53);

    let (candidate_prefill_chunck_tokens, reason) = ask_decision(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
    );
    assert_eq!(candidate_prefill_chunck_tokens, 256);
    assert_eq!(
        reason,
        PrefillChunckSizeOptimizerDecisionReason::Exploration,
        "untested candidates should report exploration"
    );
}

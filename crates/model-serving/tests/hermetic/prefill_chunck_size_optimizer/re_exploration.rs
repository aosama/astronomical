use super::*;

#[test]
fn should_re_explore_when_the_chosen_candidate_drifts_above_twice_its_median() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(31);

    // Trust all candidates so exploitation picks the best.
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        256,
        &[1_000, 1_000, 1_000],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        512,
        &[500, 500, 500],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        1_024,
        &[600, 600, 600],
    );
    assert_eq!(
        ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context
        ),
        1_024
    );

    // Drift: 1024 suddenly takes 5_000ms — well over 2x its median of 600ms.
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        1_024,
        &[5_000],
    );
    assert_ne!(
        ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context
        ),
        1_024,
        "a drift spike above twice the median should force re-exploration instead of exploitation"
    );
}

#[test]
fn should_cycle_through_all_candidates_once_during_re_exploration_before_settling() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(37);

    // Trust all candidates so exploitation picks the best.
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        256,
        &[1_000, 1_000, 1_000],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        512,
        &[500, 500, 500],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        1_024,
        &[600, 600, 600],
    );

    // Trigger drift on 1024.
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        1_024,
        &[5_000],
    );

    assert_eq!(
        ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context
        ),
        256
    );
    assert_eq!(
        ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context
        ),
        512
    );
    assert_eq!(
        ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context
        ),
        1_024
    );
}

#[test]
fn should_isolate_drift_detection_per_context() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let drifting_context = PrefillChunckSizeOptimizerContext::new(41);
    let stable_context = PrefillChunckSizeOptimizerContext::new(43);

    // Trust all candidates in both contexts.
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        drifting_context,
        256,
        &[1_000, 1_000, 1_000],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        drifting_context,
        512,
        &[500, 500, 500],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        drifting_context,
        1_024,
        &[600, 600, 600],
    );

    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        stable_context,
        256,
        &[1_000, 1_000, 1_000],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        stable_context,
        512,
        &[500, 500, 500],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        stable_context,
        1_024,
        &[600, 600, 600],
    );

    // Trigger drift only in the drifting context.
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        drifting_context,
        1_024,
        &[5_000],
    );

    assert_ne!(
        ask_candidate_prefill_chunck_tokens(&mut prefill_chunck_size_optimizer, drifting_context),
        1_024
    );
    assert_eq!(
        ask_candidate_prefill_chunck_tokens(&mut prefill_chunck_size_optimizer, stable_context),
        1_024,
        "drift in one context should not disturb a stable context"
    );
}

#[test]
fn should_report_re_exploration_reason_after_drift() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(61);

    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        256,
        &[1_000, 1_000, 1_000],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        512,
        &[500, 500, 500],
    );
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        1_024,
        &[600, 600, 600],
    );

    // Trigger drift on 1024.
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        1_024,
        &[5_000],
    );

    let (candidate_prefill_chunck_tokens, reason) = ask_decision(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
    );
    assert_eq!(candidate_prefill_chunck_tokens, 256);
    assert_eq!(
        reason,
        PrefillChunckSizeOptimizerDecisionReason::ReExplorationAfterDrift,
        "drift-triggered re-exploration should report the reason"
    );
}

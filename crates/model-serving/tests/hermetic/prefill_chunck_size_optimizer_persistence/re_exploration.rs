use super::PrefillChunckSizeOptimizerDecisionReason;
use super::support::{
    DEFAULT_CANDIDATES, DRIFT_TRIGGER_FACTOR, OPTIMIZER_STATE_FILE_NAME,
    SLIDING_WINDOW_OBSERVATION_COUNT, TRUSTED_OBSERVATION_COUNT, context_at_bucket,
    create_optimizer_with_default_candidates, explore_all_candidates, load_expect_some,
    record_full_observation, temporary_directory,
};

#[test]
fn should_round_trip_re_exploration_state_after_drift_detection() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();

    let context = context_at_bucket(0);

    // Fully explore all candidates with consistent low elapsed times
    explore_all_candidates(&mut original_optimizer, context, 1000);

    // Now record a drift-triggering observation for candidate 2048:
    // elapsed time that is >2× the median (drift_trigger_factor=2).
    // With 3 observations at 1000ms each, median is 1000ms.
    // An observation of 5000ms is 5× the median, which triggers drift.
    record_full_observation(&mut original_optimizer, context, 2048, 5000);

    // After drift, the optimizer enters re-exploration mode.
    // Save BEFORE calling ask(), so the loaded optimizer starts at
    // the same cursor position.
    original_optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    // Now call ask() on the original to see what it decides
    let decision_after_drift = original_optimizer.ask(context);
    assert_eq!(
        decision_after_drift.reason,
        PrefillChunckSizeOptimizerDecisionReason::ReExplorationAfterDrift,
        "optimizer should be in re-exploration mode after drift"
    );

    let mut loaded_optimizer = load_expect_some(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );

    // The loaded optimizer should make the same first decision in
    // re-exploration as the original (both start at cursor 0)
    let loaded_decision_after_drift = loaded_optimizer.ask(context);
    assert_eq!(
        decision_after_drift.candidate_prefill_chunck_tokens,
        loaded_decision_after_drift.candidate_prefill_chunck_tokens,
        "loaded optimizer should make the same re-exploration decision"
    );
    assert_eq!(
        decision_after_drift.reason, loaded_decision_after_drift.reason,
        "loaded optimizer should have the same re-exploration reason"
    );
}

#[test]
fn should_round_trip_re_exploration_with_partial_progress() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();

    let context = context_at_bucket(0);

    // Fully explore to establish trust
    explore_all_candidates(&mut original_optimizer, context, 1000);

    // Trigger drift for candidate 2048
    record_full_observation(&mut original_optimizer, context, 2048, 5000);

    // Re-exploration has started — step through 2 of 5 candidates
    for _re_exploration_step in 0..2 {
        let decision = original_optimizer.ask(context);
        assert_eq!(
            decision.reason,
            PrefillChunckSizeOptimizerDecisionReason::ReExplorationAfterDrift,
            "should still be re-exploring"
        );
        record_full_observation(
            &mut original_optimizer,
            context,
            decision.candidate_prefill_chunck_tokens,
            800,
        );
    }

    // Save mid-re-exploration
    original_optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    let mut loaded_optimizer = load_expect_some(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );

    // Continue re-exploration from the loaded state — both optimizers should
    // be at the same cursor position (cursor=2, remaining=3).
    let original_next_decision = original_optimizer.ask(context);
    let loaded_next_decision = loaded_optimizer.ask(context);

    assert_eq!(
        original_next_decision.candidate_prefill_chunck_tokens,
        loaded_next_decision.candidate_prefill_chunck_tokens,
        "loaded optimizer should continue re-exploration from the same cursor"
    );
    assert_eq!(
        original_next_decision.reason, loaded_next_decision.reason,
        "loaded optimizer should have the same decision reason"
    );
}

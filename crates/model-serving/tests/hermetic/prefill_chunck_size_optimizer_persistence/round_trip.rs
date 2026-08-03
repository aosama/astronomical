use super::support::{
    DEFAULT_CANDIDATES, DRIFT_TRIGGER_FACTOR, OPTIMIZER_STATE_FILE_NAME,
    SLIDING_WINDOW_OBSERVATION_COUNT, TRUSTED_OBSERVATION_COUNT, context_at_bucket,
    create_optimizer_with_default_candidates, explore_all_candidates, load_expect_some,
    record_full_observation, temporary_directory,
};

#[test]
fn should_round_trip_a_fresh_optimizer_with_no_observations() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();

    original_optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed for fresh optimizer");

    let mut loaded_optimizer = load_expect_some(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );

    // A fresh optimizer and the loaded optimizer should make the same first
    // decision since both have no observations.
    let context = context_at_bucket(0);
    let original_decision = original_optimizer.ask(context);
    let loaded_decision = loaded_optimizer.ask(context);
    assert_eq!(
        original_decision.candidate_prefill_chunck_tokens,
        loaded_decision.candidate_prefill_chunck_tokens,
        "fresh optimizer and loaded optimizer should make the same first decision"
    );
    assert_eq!(
        original_decision.reason, loaded_decision.reason,
        "fresh optimizer and loaded optimizer should give the same decision reason"
    );
}

#[test]
fn should_round_trip_an_optimizer_with_observations_across_multiple_context_buckets() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();

    // Explore context bucket 0 with fast elapsed times (smaller chunks win)
    let low_context = context_at_bucket(0);
    explore_all_candidates(&mut original_optimizer, low_context, 500);

    // Explore context bucket 2 with slower elapsed times
    let high_context = context_at_bucket(2);
    explore_all_candidates(&mut original_optimizer, high_context, 5000);

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

    // Both optimizers should now be in exploitation mode and pick the same
    // best candidate for each context bucket.
    let original_low_context_decision = original_optimizer.ask(low_context);
    let loaded_low_context_decision = loaded_optimizer.ask(low_context);
    assert_eq!(
        original_low_context_decision.candidate_prefill_chunck_tokens,
        loaded_low_context_decision.candidate_prefill_chunck_tokens,
        "original and loaded should pick the same candidate in context bucket 0"
    );
    assert_eq!(
        original_low_context_decision.reason, loaded_low_context_decision.reason,
        "original and loaded should have the same decision reason in context bucket 0"
    );

    let original_high_context_decision = original_optimizer.ask(high_context);
    let loaded_high_context_decision = loaded_optimizer.ask(high_context);
    assert_eq!(
        original_high_context_decision.candidate_prefill_chunck_tokens,
        loaded_high_context_decision.candidate_prefill_chunck_tokens,
        "original and loaded should pick the same candidate in context bucket 2"
    );
    assert_eq!(
        original_high_context_decision.reason, loaded_high_context_decision.reason,
        "original and loaded should have the same decision reason in context bucket 2"
    );
}

#[test]
fn should_preserve_sliding_window_behavior_after_round_trip() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();

    let context = context_at_bucket(0);

    // Fully explore all candidates first
    explore_all_candidates(&mut original_optimizer, context, 1000);

    // Now record 6 more observations for the first candidate (2048) to overflow
    // the sliding window of 5. The 6th observation (elapsed=20000) should push
    // out the first observation (elapsed=1000).
    for elapsed_millis in [1000, 1200, 1400, 1600, 1800, 20000] {
        record_full_observation(&mut original_optimizer, context, 2048, elapsed_millis);
    }

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

    // Both optimizers should make the same exploitation decision because
    // the sliding window state (which observations are retained) is preserved.
    let original_decision = original_optimizer.ask(context);
    let loaded_decision = loaded_optimizer.ask(context);
    assert_eq!(
        original_decision.candidate_prefill_chunck_tokens,
        loaded_decision.candidate_prefill_chunck_tokens,
        "sliding window overflow should round-trip correctly"
    );
}

#[test]
fn should_round_trip_a_partially_explored_optimizer_with_single_observations() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();

    let context = context_at_bucket(0);

    // Record only 1 observation per candidate — still in exploration phase
    for &candidate_prefill_chunck_tokens in &DEFAULT_CANDIDATES {
        let decision = original_optimizer.ask(context);
        assert_eq!(
            decision.reason,
            super::PrefillChunckSizeOptimizerDecisionReason::Exploration,
            "should be in exploration phase before persistence test"
        );
        record_full_observation(
            &mut original_optimizer,
            context,
            candidate_prefill_chunck_tokens,
            1000 + candidate_prefill_chunck_tokens as u64,
        );
    }

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

    // Both should still be in exploration — only 1 observation per candidate,
    // need 3 to become trusted.
    let original_decision = original_optimizer.ask(context);
    let loaded_decision = loaded_optimizer.ask(context);
    assert_eq!(
        original_decision.candidate_prefill_chunck_tokens,
        loaded_decision.candidate_prefill_chunck_tokens,
        "partially explored optimizer should round-trip exploration state"
    );
    assert_eq!(
        original_decision.reason, loaded_decision.reason,
        "partially explored optimizer should preserve decision reason"
    );
}

#[test]
fn should_round_trip_context_bucket_with_zero_observations_after_ask_only() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();

    // Calling ask() creates a context entry but records no observations.
    // The context_statistics map gets an entry with default (empty) candidate stats.
    let context = context_at_bucket(0);
    let _ = original_optimizer.ask(context);

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

    // Both should start exploration for the same context bucket.
    let original_decision = original_optimizer.ask(context);
    let loaded_decision = loaded_optimizer.ask(context);
    assert_eq!(
        original_decision.candidate_prefill_chunck_tokens,
        loaded_decision.candidate_prefill_chunck_tokens,
        "context bucket with no observations should round-trip correctly"
    );
}

#[test]
fn should_round_trip_large_elapsed_millis_values() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();

    let context = context_at_bucket(0);

    // Use a very large elapsed_millis (near u32::MAX) to verify no overflow
    // in serialization or throughput calculation
    let large_elapsed_millis = 4_000_000_000u64; // ~46 days of milliseconds
    for _observation_round in 0..TRUSTED_OBSERVATION_COUNT {
        for &candidate_prefill_chunck_tokens in &DEFAULT_CANDIDATES {
            record_full_observation(
                &mut original_optimizer,
                context,
                candidate_prefill_chunck_tokens,
                large_elapsed_millis,
            );
        }
    }

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

    let original_decision = original_optimizer.ask(context);
    let loaded_decision = loaded_optimizer.ask(context);
    assert_eq!(
        original_decision.candidate_prefill_chunck_tokens,
        loaded_decision.candidate_prefill_chunck_tokens,
        "large elapsed_millis values should round-trip correctly"
    );
}

use super::PrefillChunckSizeOptimizerContext;
use super::support::*;

#[test]
fn should_round_trip_a_fresh_optimizer_deterministically() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();
    original_optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    let mut loaded_optimizer = load_expect_some(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        SLIDING_WINDOW_OBSERVATION_COUNT,
    );
    let prompt_processing_context = context_at_bucket(0);
    assert_eq!(
        original_optimizer.ask(prompt_processing_context),
        loaded_optimizer.ask(prompt_processing_context)
    );
}

#[test]
fn should_round_trip_reduced_transitions_and_next_context() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();
    let unconstrained_context = PrefillChunckSizeOptimizerContext::new_with_fallback(10, 1);
    let capacity_reduced_context = PrefillChunckSizeOptimizerContext::new_with_fallback(11, 2);
    for candidate_prefill_chunck_tokens in DEFAULT_CANDIDATES {
        record_transition_observation(
            &mut original_optimizer,
            unconstrained_context,
            candidate_prefill_chunck_tokens,
            candidate_prefill_chunck_tokens.min(512),
            candidate_prefill_chunck_tokens as u64,
            capacity_reduced_context,
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
        SLIDING_WINDOW_OBSERVATION_COUNT,
    );
    assert_eq!(
        original_optimizer.ask(unconstrained_context),
        loaded_optimizer.ask(unconstrained_context)
    );
}

#[test]
fn should_preserve_recent_window_and_decision_sequence() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();
    let prompt_processing_context = context_at_bucket(0);
    observe_all_candidates(&mut original_optimizer, prompt_processing_context, 1_000);
    for elapsed_millis in [1_000, 1_200, 1_400, 1_600, 20_000] {
        record_transition_observation(
            &mut original_optimizer,
            prompt_processing_context,
            2_048,
            2_048,
            elapsed_millis,
            prompt_processing_context,
        );
    }
    for _decision_index in 0..7 {
        let _decision = original_optimizer.ask(prompt_processing_context);
    }
    original_optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    let mut loaded_optimizer = load_expect_some(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        SLIDING_WINDOW_OBSERVATION_COUNT,
    );
    assert_eq!(
        original_optimizer.ask(prompt_processing_context),
        loaded_optimizer.ask(prompt_processing_context)
    );
}

use super::support::*;

#[test]
fn should_round_trip_stale_observation_probe_state() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();
    let prompt_processing_context = context_at_bucket(0);
    observe_all_candidates(&mut original_optimizer, prompt_processing_context, 1_000);
    for _decision_index in 0..20 {
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

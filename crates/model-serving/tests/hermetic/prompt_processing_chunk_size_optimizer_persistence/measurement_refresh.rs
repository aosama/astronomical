use super::support::*;

#[test]
fn should_round_trip_stale_measurement_refresh_state() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();
    let measurement_context = context_at_position_range(0);
    measure_all_candidates(&mut original_optimizer, measurement_context, 1_000);
    for _selection_index in 0..20 {
        let _selection = original_optimizer.select_candidate_chunk_size(measurement_context);
    }
    original_optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");
    let mut loaded_optimizer = load_expect_some(
        &state_file_path_for_model(&optimizer_directory, "test-model", "rev-1"),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        MAXIMUM_RETAINED_MEASUREMENTS,
    );
    assert_eq!(
        original_optimizer.select_candidate_chunk_size(measurement_context),
        loaded_optimizer.select_candidate_chunk_size(measurement_context)
    );
}

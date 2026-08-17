use super::PromptProcessingMeasurementContext;
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
        &state_file_path_for_model(&optimizer_directory, "test-model", "rev-1"),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        MAXIMUM_RETAINED_MEASUREMENTS,
    );
    let measurement_context = context_at_position_range(0);
    assert_eq!(
        original_optimizer.select_candidate_chunk_size(measurement_context),
        loaded_optimizer.select_candidate_chunk_size(measurement_context)
    );
}

#[test]
fn should_round_trip_measurements_and_next_execution_profile() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();
    let unconstrained_context =
        PromptProcessingMeasurementContext::with_position_independent_execution_profile(10, 1);
    let capacity_reduced_context =
        PromptProcessingMeasurementContext::with_position_independent_execution_profile(11, 2);
    for candidate_chunk_size_tokens in DEFAULT_CANDIDATES {
        record_chunk_measurement(
            &mut original_optimizer,
            unconstrained_context,
            candidate_chunk_size_tokens,
            candidate_chunk_size_tokens,
            candidate_chunk_size_tokens as u64,
            capacity_reduced_context,
        );
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
        original_optimizer.select_candidate_chunk_size(unconstrained_context),
        loaded_optimizer.select_candidate_chunk_size(unconstrained_context)
    );
}

#[test]
fn should_preserve_the_recent_measurement_window() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();
    let measurement_context = context_at_position_range(0);
    measure_all_candidates(&mut original_optimizer, measurement_context, 1_000);
    for forward_elapsed_millis in [1_000, 1_200, 1_400, 1_600, 20_000] {
        record_chunk_measurement(
            &mut original_optimizer,
            measurement_context,
            2_048,
            2_048,
            forward_elapsed_millis,
            measurement_context,
        );
    }
    for _selection_index in 0..7 {
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

#[test]
fn should_resume_the_converged_winner_across_positions_after_restart() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut original_optimizer = create_optimizer_with_default_candidates();
    let execution_profile_identifier = 42;
    for (position_range_identifier, candidate_chunk_size_tokens) in
        DEFAULT_CANDIDATES.into_iter().enumerate()
    {
        let measurement_context =
            PromptProcessingMeasurementContext::with_position_independent_execution_profile(
                position_range_identifier as u64,
                execution_profile_identifier,
            );
        for _sample_index in 0..2 {
            record_chunk_measurement(
                &mut original_optimizer,
                measurement_context,
                candidate_chunk_size_tokens,
                candidate_chunk_size_tokens,
                if candidate_chunk_size_tokens == 2_048 {
                    100
                } else {
                    5_000
                },
                measurement_context,
            );
        }
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
    for position_range_identifier in 100..300 {
        let measurement_context =
            PromptProcessingMeasurementContext::with_position_independent_execution_profile(
                position_range_identifier,
                execution_profile_identifier,
            );
        let selection = loaded_optimizer.select_candidate_chunk_size(measurement_context);
        assert_eq!(selection.selected_candidate_chunk_size_tokens, 2_048);
    }
}

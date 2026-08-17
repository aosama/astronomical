//! Acceptance coverage for a user whose optimizer learns once and remains stable.

use super::*;

const SHARED_EXECUTION_PROFILE_IDENTIFIER: u64 = 91;

#[test]
fn should_keep_the_learned_winner_across_positions_without_periodic_reexploration() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    train_known_winner(&mut chunk_size_optimizer);

    for position_range_identifier in 100..400 {
        let selection = chunk_size_optimizer.select_candidate_chunk_size_with_maximum(
            context_at_position(position_range_identifier),
            1_024,
        );

        assert_eq!(selection.selected_candidate_chunk_size_tokens, 1_024);
        assert_eq!(
            selection.reason,
            PromptProcessingChunkSizeSelectionReason::MinimizeProjectedRemainingPromptLatency
        );
    }
}

#[test]
fn should_pool_a_new_position_observation_with_mature_profile_evidence() {
    let mut chunk_size_optimizer = three_candidate_optimizer();
    for sample_position_identifier in 1..=5 {
        let measurement_context = context_at_position(sample_position_identifier);
        for (candidate_chunk_size_tokens, forward_elapsed_millis) in
            [(256, 400), (512, 700), (1_024, 100)]
        {
            record_chunk_measurement(
                &mut chunk_size_optimizer,
                measurement_context,
                candidate_chunk_size_tokens,
                candidate_chunk_size_tokens,
                forward_elapsed_millis,
                context_at_position(sample_position_identifier + 1),
            );
        }
    }
    let noisy_position_context = context_at_position(6);
    record_chunk_measurement(
        &mut chunk_size_optimizer,
        noisy_position_context,
        1_024,
        1_024,
        5_000,
        context_at_position(7),
    );

    let selection = chunk_size_optimizer
        .select_candidate_chunk_size_with_maximum(noisy_position_context, 1_024);

    assert_eq!(selection.selected_candidate_chunk_size_tokens, 1_024);
    assert_eq!(
        selection.reason,
        PromptProcessingChunkSizeSelectionReason::MinimizeProjectedRemainingPromptLatency
    );
}

fn train_known_winner(chunk_size_optimizer: &mut PromptProcessingChunkSizeOptimizer) {
    for (position_range_identifier, candidate_chunk_size_tokens, forward_elapsed_millis) in
        [(1, 1_024, 600), (2, 512, 500), (3, 256, 400)]
    {
        for sample_offset in 0..2 {
            record_chunk_measurement(
                chunk_size_optimizer,
                context_at_position(position_range_identifier + sample_offset),
                candidate_chunk_size_tokens,
                candidate_chunk_size_tokens,
                forward_elapsed_millis,
                context_at_position(position_range_identifier + sample_offset + 1),
            );
        }
    }
}

fn context_at_position(position_range_identifier: u64) -> PromptProcessingMeasurementContext {
    PromptProcessingMeasurementContext::with_position_independent_execution_profile(
        position_range_identifier,
        SHARED_EXECUTION_PROFILE_IDENTIFIER,
    )
}

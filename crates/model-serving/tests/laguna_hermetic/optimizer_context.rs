//! Laguna optimizer contexts stay descriptor-driven and write no files when disabled.

use astronomical_ipc_protocol::ExpertMemoryMode;
use astronomical_model_serving::{
    LagunaPromptProcessingChunkSizer, LagunaPromptProcessingExecutionProfile,
};

use super::support::{
    LagunaQualificationSize, config_value, normalize, qualification_config_value,
};

fn execution_profile(
    layer_count: usize,
    storage_fingerprint: [u8; 32],
    expert_memory_mode: ExpertMemoryMode,
) -> LagunaPromptProcessingExecutionProfile {
    LagunaPromptProcessingExecutionProfile::from_canonical_descriptors(
        &normalize(config_value(layer_count)),
        &storage_fingerprint,
        expert_memory_mode,
        false,
    )
}

#[test]
fn should_keep_synthetic_xs_and_s_optimizer_contexts_distinct() {
    let synthetic_profile = execution_profile(3, [1; 32], ExpertMemoryMode::Resident);
    let extra_small_profile = LagunaPromptProcessingExecutionProfile::from_canonical_descriptors(
        &normalize(qualification_config_value(
            LagunaQualificationSize::ExtraSmall,
        )),
        &[2; 32],
        ExpertMemoryMode::Resident,
        false,
    );
    let small_profile = LagunaPromptProcessingExecutionProfile::from_canonical_descriptors(
        &normalize(qualification_config_value(LagunaQualificationSize::Small)),
        &[3; 32],
        ExpertMemoryMode::Paged,
        false,
    );
    let mut chunk_sizer = LagunaPromptProcessingChunkSizer::for_optimized_with_behavior(
        8_192,
        vec![1_024, 2_048, 4_096, 8_192],
        7,
        16_384,
    )
    .expect("an optimized Laguna sizer should construct");
    chunk_sizer.start_prompt_processing_request(0);
    let synthetic_context = chunk_sizer.exact_measurement_context_identifier(0, synthetic_profile);
    let extra_small_context =
        chunk_sizer.exact_measurement_context_identifier(0, extra_small_profile);
    let small_context = chunk_sizer.exact_measurement_context_identifier(0, small_profile);
    assert_ne!(synthetic_context, extra_small_context);
    assert_ne!(synthetic_context, small_context);
    assert_ne!(extra_small_context, small_context);
}

#[test]
fn should_reuse_position_independent_profiles_across_ranges() {
    let execution_profile = execution_profile(4, [9; 32], ExpertMemoryMode::Resident);
    let mut chunk_sizer = LagunaPromptProcessingChunkSizer::for_optimized_with_behavior(
        8_192,
        vec![1_024, 2_048],
        7,
        1_024,
    )
    .expect("an optimized Laguna sizer should construct");
    chunk_sizer.start_prompt_processing_request(0);
    let first_range = chunk_sizer.exact_measurement_context_identifier(0, execution_profile);
    let second_range = chunk_sizer.exact_measurement_context_identifier(1_024, execution_profile);
    assert_ne!(first_range, second_range);
    assert_eq!(first_range >> 32, second_range >> 32);
}

#[test]
fn should_isolate_restored_prefix_and_capacity_reduction_contexts() {
    let execution_profile = execution_profile(2, [4; 32], ExpertMemoryMode::Paged);
    let mut chunk_sizer = LagunaPromptProcessingChunkSizer::for_optimized_with_behavior(
        4_096,
        vec![1_024, 2_048, 4_096],
        3,
        16_384,
    )
    .expect("an optimized Laguna sizer should construct");
    chunk_sizer.start_prompt_processing_request(0);
    let cold_context = chunk_sizer.exact_measurement_context_identifier(0, execution_profile);
    chunk_sizer.start_prompt_processing_request(2_048);
    let restored_context =
        chunk_sizer.exact_measurement_context_identifier(2_048, execution_profile);
    assert_ne!(cold_context, restored_context);

    let chunk_end = chunk_sizer.next_prompt_processing_chunk_end(2_048, 3_072, execution_profile);
    assert_eq!(chunk_end, 3_072);
    chunk_sizer.record_prompt_processing_chunk_transition(1_024, 12, true, execution_profile);
    let reduced_context =
        chunk_sizer.exact_measurement_context_identifier(3_072, execution_profile);
    assert_ne!(restored_context, reduced_context);
}

#[test]
fn should_record_a_measurement_only_after_the_selected_chunk_completes() {
    let execution_profile = execution_profile(2, [5; 32], ExpertMemoryMode::Resident);
    let mut chunk_sizer = LagunaPromptProcessingChunkSizer::for_optimized_with_behavior(
        4_096,
        vec![1_024, 2_048],
        3,
        16_384,
    )
    .expect("an optimized Laguna sizer should construct");
    chunk_sizer.start_prompt_processing_request(0);
    assert!(!chunk_sizer.has_pending_selection());
    let chunk_end = chunk_sizer.next_prompt_processing_chunk_end(0, 3_000, execution_profile);
    assert!(chunk_sizer.has_pending_selection());
    assert!(chunk_end > 0);
    assert!(
        chunk_sizer
            .latest_prompt_processing_chunk_optimization_outcome()
            .is_none()
    );
    chunk_sizer.record_prompt_processing_chunk_transition(chunk_end, 15, false, execution_profile);
    assert!(!chunk_sizer.has_pending_selection());
    let outcome = chunk_sizer
        .take_latest_prompt_processing_chunk_optimization_outcome()
        .expect("a completed chunk should publish an optimizer outcome");
    assert_eq!(outcome.processed_prompt_token_count, chunk_end);
    assert_eq!(outcome.forward_elapsed_millis, 15);
    assert!(
        chunk_sizer
            .latest_prompt_processing_chunk_optimization_outcome()
            .is_none(),
        "one optimizer measurement should be emitted only once"
    );
}

#[test]
fn should_use_the_remaining_prompt_when_it_is_shorter_than_every_candidate() {
    let execution_profile = execution_profile(2, [6; 32], ExpertMemoryMode::Resident);
    let mut chunk_sizer =
        LagunaPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(2_048)
            .expect("a fixed Laguna sizer should construct");
    chunk_sizer.start_prompt_processing_request(0);
    assert!(!chunk_sizer.is_optimized());
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(0, 700, execution_profile),
        700
    );
}

#[test]
fn should_write_no_optimizer_files_in_fixed_mode() {
    let execution_profile = execution_profile(2, [7; 32], ExpertMemoryMode::Resident);
    let isolated_optimizer_directory =
        tempfile::tempdir().expect("an isolated optimizer directory should be created");
    let mut chunk_sizer =
        LagunaPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(1_024)
            .expect("a fixed Laguna sizer should construct");
    chunk_sizer.start_prompt_processing_request(0);
    let _chunk_end = chunk_sizer.next_prompt_processing_chunk_end(0, 500, execution_profile);
    chunk_sizer.record_prompt_processing_chunk_transition(500, 8, false, execution_profile);
    let directory_entries = std::fs::read_dir(isolated_optimizer_directory.path())
        .expect("the isolated optimizer directory should remain readable")
        .count();
    assert_eq!(directory_entries, 0);
}

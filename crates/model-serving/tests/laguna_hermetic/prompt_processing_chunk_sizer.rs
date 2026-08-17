//! Behavior coverage for deterministic Laguna prompt chunking.

use astronomical_model_serving::LagunaPromptProcessingChunkSizer;

#[test]
fn should_process_fixed_chunks_and_an_exact_terminal_remainder() {
    let chunk_sizer =
        LagunaPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(2_048)
            .expect("the qualified fixed size should construct");

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(0, 5_000, false),
        2_048
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(4_096, 5_000, false),
        5_000
    );
}

#[test]
fn should_use_the_ssd_streaming_fixed_size_only_while_experts_are_paged() {
    let chunk_sizer = LagunaPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, Some(256))
        .expect("the resident and SSD-streaming fixed sizes should construct");

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(0, 5_000, true),
        256
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(0, 5_000, false),
        2_048
    );
}

#[test]
fn should_reject_invalid_resident_and_ssd_streaming_chunk_sizes() {
    assert!(
        LagunaPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(0).is_err()
    );
    assert!(
        LagunaPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, Some(0))
            .is_err()
    );
    assert!(
        LagunaPromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, Some(4_096))
            .is_err()
    );
}

#[test]
fn should_halve_a_capacity_constrained_chunk_without_falling_below_one_token() {
    assert_eq!(
        LagunaPromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(2_048),
        Some(1_024)
    );
    assert_eq!(
        LagunaPromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(3),
        Some(1)
    );
    assert_eq!(
        LagunaPromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(1),
        None
    );
}

//! Behavior coverage for deterministic Qwen prompt chunking.

use astronomical_model_serving::Qwen3_5PromptProcessingChunkSizer;

#[test]
fn should_process_fixed_chunks_and_an_exact_terminal_remainder() {
    let chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(2_048)
            .expect("the qualified fixed size should construct");

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(0, 5_000),
        2_048
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(4_096, 5_000),
        5_000
    );
}

#[test]
fn should_use_the_ssd_streaming_fixed_size_only_while_experts_are_paged() {
    let chunk_sizer = Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, Some(256))
        .expect("the resident and SSD-streaming fixed sizes should construct");

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(0, 5_000, true),
        256
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(0, 5_000, false),
        2_048
    );
}

#[test]
fn should_bound_the_next_chunk_by_proven_executable_capacity() {
    let chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(2_048)
            .expect("the qualified fixed size should construct");

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_with_maximum_executable_capacity(
            0, 5_000, false, 512,
        ),
        512
    );
    assert_eq!(
        Qwen3_5PromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(512),
        Some(256)
    );
}

#[test]
fn should_reject_invalid_resident_and_ssd_streaming_chunk_sizes() {
    assert!(
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(0)
            .is_err()
    );
    assert!(
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, Some(0))
            .is_err()
    );
    assert!(
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, Some(4_096))
            .is_err()
    );
}

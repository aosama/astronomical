//! Behavior coverage for deterministic Qwen prompt chunking.

use astronomical_model_serving::Qwen3_5PromptProcessingChunkSizer;

#[test]
fn should_process_fixed_chunks_and_an_exact_terminal_remainder() {
    let chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(2_048)
            .expect("the chosen fixed size should construct");

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
    let chunk_sizer = Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, 256)
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
            .expect("the chosen fixed size should construct");

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
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, 0)
            .is_err()
    );
    let larger_paged_chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, 4_096)
            .expect("a larger paged chunk should construct");
    assert_eq!(
        larger_paged_chunk_sizer
            .next_prompt_processing_chunk_end_for_expert_residency(0, 9_000, true,),
        4_096
    );
    assert_eq!(
        larger_paged_chunk_sizer
            .next_prompt_processing_chunk_end_for_expert_residency(0, 9_000, false,),
        2_048
    );
}

#[test]
fn should_default_paged_chunks_larger_than_resident_chunks() {
    let chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, 8_192)
            .expect("an explicit larger paged chunk should construct");
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(0, 20_000, true),
        8_192
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(0, 20_000, false),
        2_048
    );
}

#[test]
fn should_fold_a_short_paged_remainder_instead_of_paying_a_second_leftover_stream() {
    let chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, 2_048)
            .expect("the resident and SSD-streaming fixed sizes should construct");

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(0, 4_401, true),
        2_048
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(2_048, 4_401, true),
        4_401
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(2_048, 4_401, false),
        4_096
    );
}

#[test]
fn should_keep_full_paged_chunks_when_the_remainder_fills_another_configured_chunk() {
    let chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(2_048, 2_048)
            .expect("the resident and SSD-streaming fixed sizes should construct");

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(0, 10_000, true),
        2_048
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(2_048, 10_000, true),
        4_096
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(4_096, 10_000, true),
        6_144
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end_for_expert_residency(6_144, 10_000, true),
        10_000
    );
}

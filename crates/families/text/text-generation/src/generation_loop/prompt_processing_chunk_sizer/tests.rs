use super::PromptProcessingChunkSizer;

fn chunk_sizer_of(prompt_processing_chunk_size_tokens: u32) -> PromptProcessingChunkSizer {
    PromptProcessingChunkSizer::for_chunk_size_tokens(prompt_processing_chunk_size_tokens)
        .expect("a positive chunk size should construct")
}

// Positive: chunks end at the configured size, then at the prompt end.
#[test]
fn should_cut_chunks_at_the_configured_size_and_end_at_the_prompt() {
    let chunk_sizer = chunk_sizer_of(2_048);

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(0, 5_000),
        2_048
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(4_096, 5_000),
        5_000
    );
}

// Positive: memory planning and execution use the same configured size.
#[test]
fn should_use_the_configured_size_for_admission_and_execution() {
    let chunk_sizer = chunk_sizer_of(4_096);

    assert_eq!(
        chunk_sizer.prompt_processing_operation_bound_tokens(),
        4_096
    );
    assert_eq!(
        chunk_sizer.maximum_prompt_processing_chunk_size_tokens(),
        4_096
    );
}

// Edge: a short remainder gets its own chunk instead of stretching the one
// before past the configured size.
#[test]
fn should_never_exceed_the_configured_chunk_when_a_short_remainder_follows() {
    let chunk_sizer = chunk_sizer_of(2_048);

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(0, 4_401),
        2_048
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(2_048, 4_401),
        4_096
    );
    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(4_096, 4_401),
        4_401
    );
}

// Edge: a prompt of exactly one chunk ends at the chunk boundary.
#[test]
fn should_end_a_one_chunk_prompt_at_the_chunk_boundary() {
    let chunk_sizer = chunk_sizer_of(2_048);

    assert_eq!(
        chunk_sizer.next_prompt_processing_chunk_end(0, 2_048),
        2_048
    );
}

// Negative: a zero chunk size is refused.
#[test]
fn should_reject_a_zero_chunk_size() {
    assert!(PromptProcessingChunkSizer::for_chunk_size_tokens(0).is_err());
}

/// One chunk sizer serves every engine, so its source names none of them.
#[test]
fn should_name_no_engine_in_the_chunk_sizer_source() {
    let chunk_sizer_source = include_str!("../prompt_processing_chunk_sizer.rs");
    for forbidden_fragment in [
        "AdaptiveRamGrowthExecutionProfile",
        "ssd_streaming",
        "Resident",
        "Streaming",
    ] {
        for (line_index, source_line) in chunk_sizer_source.lines().enumerate() {
            if source_line.trim_start().starts_with("//") {
                continue;
            }
            assert!(
                !source_line.contains(forbidden_fragment),
                "prompt_processing_chunk_sizer.rs:{} names {forbidden_fragment}",
                line_index + 1,
            );
        }
    }
}

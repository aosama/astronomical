//! Chunk-sizer reproduction of the 2026-10-15 paged-prefill stall: the
//! halving ladder a capacity failure walks, and the request-wide collapse to
//! single-token chunks after the first constrained success.

use super::Qwen3_5StreamingPromptProcessingChunkSizer;

fn sizer_for_test_chunk_span() -> Qwen3_5StreamingPromptProcessingChunkSizer {
    Qwen3_5StreamingPromptProcessingChunkSizer::for_ssd_streaming_chunk_size_tokens(2_048)
        .expect("the test chunk span should be valid")
}

/// The incident's 39-token prompt entered the halving ladder at its whole
/// length (39 tokens fold under the 2,048-token span) and each typed
/// active-memory failure halved the executable attempt:
/// 39 -> 19 -> 9 -> 4 -> 2 -> 1 -> none. Six failed forwards, each re-paying
/// at least one complete-layer SSD read, before a single-token success.
#[test]
fn should_walk_the_incident_halving_ladder_from_a_folded_whole_prompt() {
    let mut executable_chunk_size_tokens = Some(39_usize);
    let mut ladder = Vec::new();
    while let Some(attempt_size_tokens) = executable_chunk_size_tokens {
        ladder.push(attempt_size_tokens);
        executable_chunk_size_tokens =
            Qwen3_5StreamingPromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(
                attempt_size_tokens,
            );
    }
    assert_eq!(ladder, vec![39, 19, 9, 4, 2, 1]);
}

/// A prompt shorter than the configured span is one folded chunk: the
/// incident's first attempt was the entire 39-token prompt, so its failure
/// retried nothing smaller until recovery halved it.
#[test]
fn should_fold_a_short_prompt_into_one_whole_chunk_attempt() {
    let sizer = sizer_for_test_chunk_span();
    assert_eq!(
        sizer.next_prompt_processing_chunk_end_with_maximum_executable_capacity(0, 39, usize::MAX),
        39
    );
}

/// The request-wide collapse: after the halving ladder's one-token success,
/// the recorded maximum executable capacity clamps every remaining chunk of
/// the request to one token. The incident's log shows nineteen consecutive
/// single-token chunks after the first success, each republishing the full
/// residency plan — the sizer never widens again within the request.
#[test]
fn should_collapse_every_later_chunk_of_the_request_to_the_last_successful_attempt_size() {
    let sizer = sizer_for_test_chunk_span();
    let remaining_chunk_ranges = sizer.remaining_prompt_processing_chunk_ranges(1, 39, 1);
    assert_eq!(remaining_chunk_ranges.len(), 38);
    assert!(
        remaining_chunk_ranges
            .iter()
            .all(|(chunk_start, chunk_end)| chunk_end - chunk_start == 1),
        "every later chunk must stay one token wide"
    );
}

//! Layer-major span plans: the chunk path's spans with the prompt cache on
//! and off, a prompt one span long, a block-aligned prompt, and a cursor
//! that does not start at zero.

use super::layer_major_span_plan;
use crate::qwen3_5_streaming::model::layer_major_prefill::LayerMajorSpan;

const CHUNK_TOKENS: usize = 4096;
const BLOCK_TOKENS: usize = 1792;

fn span(
    start_token: usize,
    end_token: usize,
    interior_boundary_tokens: Vec<usize>,
    ends_at_inner_boundary: bool,
) -> LayerMajorSpan {
    LayerMajorSpan {
        start_token,
        end_token,
        interior_boundary_tokens,
        ends_at_inner_boundary,
    }
}

#[test]
fn should_run_the_cached_chunks_and_leave_the_tail_token_alone() {
    // The engine ran this 4,350-token prompt as 1,792 + 2,557 + 1 tokens.
    let span_plan = layer_major_span_plan(0, 4350, CHUNK_TOKENS, Some(BLOCK_TOKENS))
        .expect("two spans should make a layer-major plan");

    assert_eq!(span_plan.group_end_token, 4349);
    assert_eq!(
        span_plan.spans,
        vec![
            span(0, 1792, Vec::new(), true),
            span(1792, 4349, vec![1792], false),
        ]
    );
}

#[test]
fn should_run_configured_chunks_to_the_end_without_the_cache() {
    let span_plan = layer_major_span_plan(0, 10_000, CHUNK_TOKENS, None)
        .expect("three spans should make a layer-major plan");

    assert_eq!(span_plan.group_end_token, 10_000);
    assert_eq!(
        span_plan.spans,
        vec![
            span(0, 4096, Vec::new(), false),
            span(4096, 8192, Vec::new(), false),
            span(8192, 10_000, Vec::new(), false),
        ]
    );
}

#[test]
fn should_leave_a_one_span_prompt_to_the_chunk_path() {
    assert_eq!(layer_major_span_plan(0, 1000, CHUNK_TOKENS, None), None);
    assert_eq!(layer_major_span_plan(0, 4096, CHUNK_TOKENS, None), None);
}

#[test]
fn should_publish_the_last_boundary_from_the_final_state_of_a_block_aligned_prompt() {
    let span_plan = layer_major_span_plan(0, 3584, CHUNK_TOKENS, Some(BLOCK_TOKENS))
        .expect("two spans should make a layer-major plan");

    assert_eq!(span_plan.group_end_token, 3584);
    assert_eq!(
        span_plan.spans,
        vec![
            span(0, 1792, Vec::new(), true),
            span(1792, 3584, Vec::new(), false),
        ]
    );
}

#[test]
fn should_plan_from_a_restored_cursor_relative_to_it() {
    let span_plan = layer_major_span_plan(1000, 9200, CHUNK_TOKENS, None)
        .expect("two spans should make a layer-major plan");

    assert_eq!(span_plan.group_end_token, 9200);
    assert_eq!(
        span_plan.spans.first().map(|first| first.start_token),
        Some(0)
    );
    assert_eq!(
        span_plan.spans.last().map(|last| last.end_token),
        Some(8200)
    );
}

#[test]
fn should_plan_nothing_at_the_end_of_the_prompt() {
    assert_eq!(layer_major_span_plan(500, 500, CHUNK_TOKENS, None), None);
}

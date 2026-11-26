//! Layer-major span plans: the chunk path's spans with the astro SSD prompt cache on
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
fn should_run_the_astro_ssd_prompt_cache_block_chunks_and_leave_the_tail_token_alone() {
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
fn should_run_configured_chunks_to_the_end_without_the_astro_ssd_prompt_cache() {
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
fn should_plan_one_span_for_a_prompt_that_fits_one() {
    let short_prompt_plan =
        layer_major_span_plan(0, 1000, CHUNK_TOKENS, None).expect("a short prompt is one span");
    assert_eq!(
        short_prompt_plan.spans,
        vec![span(0, 1000, Vec::new(), false)]
    );
    assert_eq!(short_prompt_plan.group_end_token, 1000);
    let full_span_plan = layer_major_span_plan(0, 4096, CHUNK_TOKENS, None)
        .expect("a prompt of exactly one span is one span");
    assert_eq!(full_span_plan.spans, vec![span(0, 4096, Vec::new(), false)]);
}

#[test]
fn should_plan_nothing_when_no_token_remains() {
    assert_eq!(layer_major_span_plan(1000, 1000, CHUNK_TOKENS, None), None);
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
        span_plan.spans,
        vec![
            span(0, 4096, Vec::new(), false),
            span(4096, 8192, Vec::new(), false),
            span(8192, 8200, Vec::new(), false),
        ]
    );
}

#[test]
fn should_keep_a_boundary_inside_a_span_when_chunks_exceed_one_block() {
    let span_plan = layer_major_span_plan(0, 5000, 2048, Some(BLOCK_TOKENS))
        .expect("several spans should make a layer-major plan");

    let first_span = &span_plan.spans[0];
    assert_eq!((first_span.start_token, first_span.end_token), (0, 2048));
    assert_eq!(first_span.interior_boundary_tokens, vec![1792]);
    assert!(!first_span.ends_at_inner_boundary);
}

#[test]
fn should_start_with_a_one_token_span_from_a_cursor_one_short_of_a_block() {
    let span_plan = layer_major_span_plan(1791, 4350, CHUNK_TOKENS, Some(BLOCK_TOKENS))
        .expect("two spans should make a layer-major plan");

    assert_eq!(span_plan.spans[0], span(0, 1, Vec::new(), true));
    assert_eq!(span_plan.group_end_token, 4349);
}

#[test]
fn should_plan_the_one_span_left_from_a_block_cursor_up_to_the_tail_capture() {
    let span_plan = layer_major_span_plan(1792, 4350, CHUNK_TOKENS, Some(BLOCK_TOKENS))
        .expect("the span that remains is a plan of one span");

    assert_eq!(span_plan.spans, vec![span(0, 2557, vec![1792], false)]);
    assert_eq!(span_plan.group_end_token, 4349);
}

#[test]
fn should_treat_a_zero_block_as_no_astro_ssd_prompt_cache() {
    assert_eq!(
        layer_major_span_plan(0, 10_000, CHUNK_TOKENS, Some(0)),
        layer_major_span_plan(0, 10_000, CHUNK_TOKENS, None)
    );
}

#[test]
fn should_make_progress_with_a_zero_chunk_size() {
    let span_plan = layer_major_span_plan(0, 3, 0, None)
        .expect("one-token spans should make a layer-major plan");

    assert_eq!(span_plan.spans.len(), 3);
}

#[test]
fn should_report_the_largest_span() {
    let span_plan = layer_major_span_plan(0, 4350, CHUNK_TOKENS, Some(BLOCK_TOKENS))
        .expect("two spans should make a layer-major plan");

    assert_eq!(span_plan.largest_span_token_count(), 2557);
}

#[test]
fn should_plan_nothing_at_the_end_of_the_prompt() {
    assert_eq!(layer_major_span_plan(500, 500, CHUNK_TOKENS, None), None);
}

#[test]
fn should_cut_a_plan_down_to_its_first_span_as_the_groups_end() {
    let first_span_plan = layer_major_span_plan(0, 4350, CHUNK_TOKENS, Some(BLOCK_TOKENS))
        .expect("two spans should make a layer-major plan")
        .limited_to_first_span(0);

    assert_eq!(first_span_plan.group_end_token, 1792);
    assert_eq!(
        first_span_plan.spans,
        vec![span(0, 1792, Vec::new(), false)],
        "the span ends the group, so its end is no inner boundary"
    );
}

#[test]
fn should_leave_a_one_span_plan_as_it_is() {
    let one_span_plan = layer_major_span_plan(0, 1000, CHUNK_TOKENS, None)
        .expect("one span should make a layer-major plan");

    assert_eq!(
        one_span_plan.clone().limited_to_first_span(0),
        one_span_plan
    );
}

#[test]
fn should_end_the_first_span_from_a_cursor_past_zero() {
    let first_span_plan = layer_major_span_plan(1000, 10_000, CHUNK_TOKENS, None)
        .expect("three spans should make a layer-major plan")
        .limited_to_first_span(1000);

    assert_eq!(first_span_plan.group_end_token, 1000 + CHUNK_TOKENS);
    assert_eq!(first_span_plan.spans.len(), 1);
    assert_eq!(first_span_plan.largest_span_token_count(), CHUNK_TOKENS);
}

#[test]
fn should_read_the_rest_of_a_prompt_in_later_groups_of_one_span() {
    let mut prefill_cursor = 0;
    let mut group_ends = Vec::new();
    while let Some(span_plan) = layer_major_span_plan(prefill_cursor, 10_000, CHUNK_TOKENS, None) {
        let first_span_plan = span_plan.limited_to_first_span(prefill_cursor);
        prefill_cursor = first_span_plan.group_end_token;
        group_ends.push(prefill_cursor);
    }

    assert_eq!(group_ends, vec![4096, 8192, 10_000]);
}

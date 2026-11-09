//! Span tiling for layer-major prefill: spans must start at zero, follow one
//! another without gaps or overlaps, never be empty, and cover every token.

use super::{LayerMajorSpan, even_layer_major_spans, validate_layer_major_spans};

fn span(start_token: usize, end_token: usize) -> LayerMajorSpan {
    LayerMajorSpan {
        start_token,
        end_token,
        interior_boundary_tokens: Vec::new(),
        ends_at_inner_boundary: false,
    }
}

#[test]
fn should_accept_spans_that_tile_every_token() {
    assert!(
        validate_layer_major_spans(4350, &[span(0, 1792), span(1792, 4349), span(4349, 4350)])
            .is_ok()
    );
}

#[test]
fn should_accept_a_single_span() {
    assert!(validate_layer_major_spans(10, &[span(0, 10)]).is_ok());
}

#[test]
fn should_refuse_no_spans() {
    assert!(validate_layer_major_spans(0, &[]).is_err());
}

#[test]
fn should_refuse_a_gap_between_spans() {
    assert!(validate_layer_major_spans(20, &[span(0, 10), span(11, 20)]).is_err());
}

#[test]
fn should_refuse_overlapping_spans() {
    assert!(validate_layer_major_spans(20, &[span(0, 10), span(9, 20)]).is_err());
}

#[test]
fn should_refuse_an_empty_span() {
    assert!(validate_layer_major_spans(10, &[span(0, 10), span(10, 10)]).is_err());
}

#[test]
fn should_refuse_spans_that_stop_short_of_the_last_token() {
    assert!(validate_layer_major_spans(20, &[span(0, 10), span(10, 19)]).is_err());
}

#[test]
fn should_refuse_spans_that_do_not_start_at_zero() {
    assert!(validate_layer_major_spans(10, &[span(1, 10)]).is_err());
}

#[test]
fn should_split_a_prompt_into_even_spans_with_a_shorter_last_one() {
    let spans = even_layer_major_spans(1000, 256);
    assert_eq!(spans.len(), 4);
    assert_eq!(
        spans.last().map(|last| (last.start_token, last.end_token)),
        Some((768, 1000))
    );
    assert!(validate_layer_major_spans(1000, &spans).is_ok());
}

#[test]
fn should_make_one_span_when_the_prompt_fits_in_one() {
    assert_eq!(even_layer_major_spans(100, 256).len(), 1);
}

#[test]
fn should_make_no_span_for_an_empty_prompt() {
    assert!(even_layer_major_spans(0, 256).is_empty());
}

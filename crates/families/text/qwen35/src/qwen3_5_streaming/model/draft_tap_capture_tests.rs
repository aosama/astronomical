//! Taps are kept only when asked for, in the drafter's order, newest rows
//! first to stay, and a tap that fell short yields nothing.

use mlx_bridge::MlxArray;
use serial_test::file_serial;

use super::DraftTapCapture;
use crate::qwen3_5_streaming::model::in_graph_route_lookup::in_graph_route_lookup_tests::test_runtime;

/// `[1, rows, 1]` rows whose values are `marker`, so a test can tell which
/// span and which layer an array came from.
fn span_rows(marker: u32, row_count: usize) -> MlxArray {
    MlxArray::from_u32(&vec![marker; row_count], &[1, row_count as i32, 1])
        .expect("a span's rows upload")
}

fn markers(span_rows: &[MlxArray]) -> Vec<u32> {
    let mut span_markers = Vec::with_capacity(span_rows.len());
    for rows in span_rows {
        span_markers.push(rows.to_vec_u32().expect("readable")[0]);
    }
    span_markers
}

#[test]
#[file_serial]
fn should_keep_requested_layers_in_the_order_they_were_asked_for() {
    let _runtime = test_runtime();
    let capture = DraftTapCapture::default();
    // The drafter lists layer 5 before layer 2; one forward of four rows.
    capture.begin(&[5, 2], 1_000);
    for layer_index in 0..8 {
        capture.keep_layer_rows(layer_index, 30, &span_rows(layer_index as u32, 4));
    }

    let taps = capture.take().expect("both requested layers ran");

    assert_eq!(taps.first_position, 30);
    assert_eq!(markers(&taps.span_rows_by_tap[0]), vec![5]);
    assert_eq!(markers(&taps.span_rows_by_tap[1]), vec![2]);
    assert!(capture.take().is_none(), "taking ends the request");
}

#[test]
#[file_serial]
fn should_keep_a_prompts_spans_in_position_order_layer_by_layer() {
    let _runtime = test_runtime();
    let capture = DraftTapCapture::default();
    capture.begin(&[1, 3], 1_000);
    // Every span of a layer arrives before any span of the next.
    for layer_index in 0..4 {
        for (span_index, first_position) in [0, 8, 16].into_iter().enumerate() {
            let marker = (layer_index * 10 + span_index) as u32;
            capture.keep_layer_rows(layer_index, first_position, &span_rows(marker, 8));
        }
    }

    let taps = capture.take().expect("every span of both taps ran");

    assert_eq!(taps.first_position, 0);
    assert_eq!(markers(&taps.span_rows_by_tap[0]), vec![10, 11, 12]);
    assert_eq!(markers(&taps.span_rows_by_tap[1]), vec![30, 31, 32]);
}

#[test]
#[file_serial]
fn should_let_the_oldest_spans_go_once_newer_ones_reach_the_row_limit() {
    let _runtime = test_runtime();
    let capture = DraftTapCapture::default();
    // Twelve rows wanted, spans of eight: the last two spans hold sixteen,
    // and dropping the second-last would leave eight, short of twelve.
    capture.begin(&[0], 12);
    for (span_index, first_position) in [0, 8, 16, 24].into_iter().enumerate() {
        capture.keep_layer_rows(0, first_position, &span_rows(span_index as u32, 8));
    }

    let taps = capture.take().expect("the tap ran");

    assert_eq!(taps.first_position, 16);
    assert_eq!(markers(&taps.span_rows_by_tap[0]), vec![2, 3]);
}

#[test]
#[file_serial]
fn should_replace_a_span_that_is_run_again_and_every_span_after_it() {
    let _runtime = test_runtime();
    let capture = DraftTapCapture::default();
    capture.begin(&[0], 1_000);
    capture.keep_layer_rows(0, 0, &span_rows(1, 8));
    capture.keep_layer_rows(0, 8, &span_rows(2, 8));
    capture.keep_layer_rows(0, 16, &span_rows(3, 8));
    // The span at position 8 failed and is read again, then the one after.
    capture.keep_layer_rows(0, 8, &span_rows(7, 8));
    capture.keep_layer_rows(0, 16, &span_rows(9, 8));

    let taps = capture.take().expect("the tap ran");

    assert_eq!(markers(&taps.span_rows_by_tap[0]), vec![1, 7, 9]);
}

#[test]
#[file_serial]
fn should_keep_nothing_from_a_forward_nobody_asked_taps_of() {
    let _runtime = test_runtime();
    let capture = DraftTapCapture::default();

    capture.keep_layer_rows(3, 0, &span_rows(3, 4));

    assert!(capture.take().is_none());
}

#[test]
#[file_serial]
fn should_yield_nothing_when_a_requested_layer_never_ran_or_fell_short() {
    let _runtime = test_runtime();
    let never_ran = DraftTapCapture::default();
    never_ran.begin(&[1, 40], 1_000);
    never_ran.keep_layer_rows(1, 0, &span_rows(1, 4));
    assert!(
        never_ran.take().is_none(),
        "a short list would pair rows with the wrong taps"
    );

    let fell_short = DraftTapCapture::default();
    fell_short.begin(&[1, 2], 1_000);
    fell_short.keep_layer_rows(1, 0, &span_rows(1, 4));
    fell_short.keep_layer_rows(1, 4, &span_rows(1, 4));
    fell_short.keep_layer_rows(2, 0, &span_rows(2, 4));
    assert!(
        fell_short.take().is_none(),
        "taps that cover different rows cannot be joined"
    );
}

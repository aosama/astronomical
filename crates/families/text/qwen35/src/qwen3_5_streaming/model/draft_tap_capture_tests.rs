//! Taps are kept only when asked for, in the drafter's order, and a missing
//! layer yields nothing rather than a short list.

use mlx_bridge::MlxArray;
use serial_test::file_serial;

use super::DraftTapCapture;
use crate::qwen3_5_streaming::model::in_graph_route_lookup::in_graph_route_lookup_tests::test_runtime;

fn layer_output(marker: u32) -> MlxArray {
    MlxArray::from_u32(&[marker], &[1]).expect("a layer output uploads")
}

#[test]
#[file_serial]
fn should_keep_requested_layers_in_the_order_they_were_asked_for() {
    // Arrays need MLX up, though nothing here is computed.
    let _runtime = test_runtime();
    let capture = DraftTapCapture::default();
    // The drafter lists layer 5 before layer 2.
    capture.begin(&[5, 2]);
    for layer_index in 0..8 {
        capture.keep_layer_output(layer_index, &layer_output(layer_index as u32));
    }

    let taps = capture.take().expect("both requested layers ran");

    assert_eq!(taps.len(), 2);
    assert_eq!(taps[0].to_vec_u32().expect("readable"), vec![5]);
    assert_eq!(taps[1].to_vec_u32().expect("readable"), vec![2]);
    assert!(capture.take().is_none(), "taking ends the request");
}

#[test]
#[file_serial]
fn should_keep_nothing_from_a_forward_nobody_asked_taps_of() {
    // Arrays need MLX up, though nothing here is computed.
    let _runtime = test_runtime();
    let capture = DraftTapCapture::default();

    capture.keep_layer_output(3, &layer_output(3));

    assert!(capture.take().is_none());
}

#[test]
#[file_serial]
fn should_yield_nothing_when_a_requested_layer_never_ran() {
    // Arrays need MLX up, though nothing here is computed.
    let _runtime = test_runtime();
    let capture = DraftTapCapture::default();
    capture.begin(&[1, 40]);
    capture.keep_layer_output(1, &layer_output(1));

    assert!(
        capture.take().is_none(),
        "a short list would pair rows with the wrong taps"
    );
}

#[test]
#[file_serial]
fn should_keep_the_second_run_of_a_forward_served_twice() {
    // Arrays need MLX up, though nothing here is computed.
    let _runtime = test_runtime();
    let capture = DraftTapCapture::default();
    capture.begin(&[0]);
    capture.keep_layer_output(0, &layer_output(7));
    capture.keep_layer_output(0, &layer_output(9));

    let taps = capture.take().expect("the layer ran");

    assert_eq!(taps[0].to_vec_u32().expect("readable"), vec![9]);
}

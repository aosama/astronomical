use serial_test::serial;

use super::{reflect_one_sample_at_start, source_convolution_geometry, upsampler_geometry};
use crate::speech_models::speech_test_support::{
    assert_close, initialize_test_runtime, read_back, upload,
};

#[test]
#[serial]
fn should_copy_the_second_sample_in_front() {
    let runtime = initialize_test_runtime();
    let features = upload(&runtime, &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], &[3, 2]);
    let padded = read_back(
        &runtime,
        &reflect_one_sample_at_start(&runtime, &features).expect("padding"),
    );
    assert_close(&padded, &[2.0, 20.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 0.0);
}

#[test]
fn should_upsample_ten_then_six_times_with_centred_padding() {
    let first = upsampler_geometry(0);
    let second = upsampler_geometry(1);
    assert_eq!((first.stride, first.padding), (10, 5));
    assert_eq!((second.stride, second.padding), (6, 3));
}

// Boundary: the finest stage needs no striding of the excitation.
#[test]
fn should_stride_the_excitation_down_only_before_the_last_stage() {
    let coarse = source_convolution_geometry(0);
    let finest = source_convolution_geometry(1);
    assert_eq!((coarse.stride, coarse.padding), (6, 3));
    assert_eq!((finest.stride, finest.padding), (1, 0));
}

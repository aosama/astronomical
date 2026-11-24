use serial_test::serial;

use super::{elu, leaky_relu, modulate};
use crate::speech_models::speech_test_support::{
    assert_close, initialize_test_runtime, read_back, upload,
};

#[test]
#[serial]
fn should_scale_only_negative_values_in_leaky_relu() {
    let runtime = initialize_test_runtime();
    let input = upload(&runtime, &[-2.0, 0.0, 3.0], &[3]);
    let activated = read_back(
        &runtime,
        &leaky_relu(&runtime, &input, 0.2).expect("activation"),
    );
    assert_close(&activated, &[-0.4, 0.0, 3.0], 1e-6);
}

#[test]
#[serial]
fn should_follow_exp_minus_one_below_zero_in_elu() {
    let runtime = initialize_test_runtime();
    let input = upload(&runtime, &[-1.0, 0.0, 2.5], &[3]);
    let activated = read_back(&runtime, &elu(&runtime, &input).expect("activation"));
    assert_close(&activated, &[(-1.0f32).exp() - 1.0, 0.0, 2.5], 1e-6);
}

// Edge: a very negative input approaches -1 without overflow.
#[test]
#[serial]
fn should_approach_minus_one_for_very_negative_elu_inputs() {
    let runtime = initialize_test_runtime();
    let input = upload(&runtime, &[-80.0], &[1]);
    let activated = read_back(&runtime, &elu(&runtime, &input).expect("activation"));
    assert_close(&activated, &[-1.0], 1e-6);
}

#[test]
#[serial]
fn should_scale_by_one_plus_scale_then_shift() {
    let runtime = initialize_test_runtime();
    let normalized = upload(&runtime, &[1.0, -2.0], &[1, 2]);
    let shift = upload(&runtime, &[0.5, 0.0], &[1, 2]);
    let scale = upload(&runtime, &[1.0, -1.0], &[1, 2]);
    let modulated = read_back(
        &runtime,
        &modulate(&runtime, &normalized, &shift, &scale).expect("modulation"),
    );
    assert_close(&modulated, &[2.5, 0.0], 1e-6);
}

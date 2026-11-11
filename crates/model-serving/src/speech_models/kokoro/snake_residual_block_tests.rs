use serial_test::serial;

use super::snake;
use crate::speech_models::speech_test_support::{
    assert_close, initialize_test_runtime, read_back, upload,
};

#[test]
#[serial]
fn should_add_the_squared_sine_over_the_frequency() {
    let runtime = initialize_test_runtime();
    let input = upload(&runtime, &[0.5, -1.0], &[1, 2]);
    let frequency = upload(&runtime, &[2.0, 1.0], &[1, 2]);
    let activated = read_back(
        &runtime,
        &snake(&runtime, &input, &frequency).expect("snake"),
    );
    let expected = [
        0.5 + 1.0f32.sin().powi(2) / 2.0,
        -1.0 + (-1.0f32).sin().powi(2),
    ];
    assert_close(&activated, &expected, 1e-5);
}

// Edge: Snake is the identity at the zeros of the sine.
#[test]
#[serial]
fn should_leave_zero_unchanged() {
    let runtime = initialize_test_runtime();
    let input = upload(&runtime, &[0.0], &[1, 1]);
    let frequency = upload(&runtime, &[3.0], &[1, 1]);
    assert_close(
        &read_back(
            &runtime,
            &snake(&runtime, &input, &frequency).expect("snake"),
        ),
        &[0.0],
        0.0,
    );
}

use serial_test::serial;

use super::uncentred_rms_norm;
use crate::speech_models::speech_test_support::{
    assert_close, initialize_test_runtime, read_back, upload,
};

#[test]
#[serial]
fn should_divide_by_the_root_mean_square_without_centring() {
    let runtime = initialize_test_runtime();
    let input_values = [1.0f32, 2.0, 3.0, 6.0];
    let mean_square = input_values.iter().map(|value| value * value).sum::<f32>() / 4.0;
    let input = upload(&runtime, &input_values, &[1, 4]);
    let norm_scale = upload(&runtime, &[2.0; 4], &[4]);
    let normalized = read_back(
        &runtime,
        &uncentred_rms_norm(&runtime, &input, &norm_scale).expect("norm"),
    );
    let expected: Vec<f32> = input_values
        .iter()
        .map(|value| 2.0 * value / (mean_square + 1e-5).sqrt())
        .collect();
    assert_close(&normalized, &expected, 1e-5);
}

// Edge: zeros stay zeros; the epsilon keeps the division finite.
#[test]
#[serial]
fn should_keep_a_zero_row_at_zero() {
    let runtime = initialize_test_runtime();
    let input = upload(&runtime, &[0.0; 4], &[1, 4]);
    let norm_scale = upload(&runtime, &[1.0; 4], &[4]);
    assert_close(
        &read_back(
            &runtime,
            &uncentred_rms_norm(&runtime, &input, &norm_scale).expect("norm"),
        ),
        &[0.0; 4],
        0.0,
    );
}

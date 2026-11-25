use mlx_bridge::MlxDtype;
use model_primitives::synthetic_weights_test_support::initialize_test_runtime as test_runtime;
use serial_test::file_serial;

use super::grouped_rms_norm;

#[test]
#[file_serial]
fn should_normalize_each_lane_over_its_own_features() {
    let runtime = test_runtime();
    // Two lanes of two features: lane one [3, 4], lane two [1, 1].
    let lanes = runtime
        .array_from_f32(&[3.0, 4.0, 1.0, 1.0], &[1, 1, 4])
        .expect("input");
    let scale = runtime
        .array_from_f32(&[1.0, 2.0, 1.0, 0.5], &[4])
        .expect("scale");
    let normalized = grouped_rms_norm(&runtime, &lanes, &scale, 2, 0.0).expect("norm");
    let values = normalized.to_vec_f32().expect("values");
    // Lane one's RMS is sqrt((9 + 16) / 2); lane two's is 1.
    let lane_one_root = (12.5_f32).sqrt();
    let expected = [3.0 / lane_one_root, 8.0 / lane_one_root, 1.0, 0.5];
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
}

#[test]
#[file_serial]
fn should_return_the_input_dtype_after_computing_in_float32() {
    let runtime = test_runtime();
    let lanes = runtime
        .astype(
            &runtime
                .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4])
                .expect("input"),
            MlxDtype::BFloat16,
        )
        .expect("bfloat16 input");
    let scale = runtime.array_from_f32(&[1.0; 4], &[4]).expect("scale");
    let normalized = grouped_rms_norm(&runtime, &lanes, &scale, 4, 1e-6).expect("norm");
    assert_eq!(normalized.dtype(), MlxDtype::BFloat16);
}

#[test]
#[file_serial]
fn should_refuse_a_width_that_does_not_divide_into_groups() {
    let runtime = test_runtime();
    let lanes = runtime.array_from_f32(&[1.0; 6], &[1, 6]).expect("input");
    let scale = runtime.array_from_f32(&[1.0; 6], &[6]).expect("scale");
    assert!(grouped_rms_norm(&runtime, &lanes, &scale, 4, 1e-6).is_err());
}

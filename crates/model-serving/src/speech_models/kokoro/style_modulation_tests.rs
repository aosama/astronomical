use serial_test::serial;

use super::StyleModulation;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_test_support::{
    assert_close, checkpoint_from_arrays, initialize_test_runtime, read_back, upload,
};

/// A projection whose weights are zero, so its bias alone sets the scale
/// (first channel) and shift (second channel) for a one-channel input.
fn constant_style_modulation(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    scale: f32,
    shift: f32,
) -> StyleModulation {
    let tensors = checkpoint_from_arrays(
        runtime,
        &[
            ("norm.fc.weight", upload(runtime, &[0.0, 0.0], &[2, 1])),
            ("norm.fc.bias", upload(runtime, &[scale, shift], &[2])),
        ],
    );
    let checkpoint = CheckpointTensors {
        runtime,
        tensors: &tensors,
    };
    StyleModulation::load(&checkpoint, "norm", 1).expect("the modulation loads")
}

#[test]
#[serial]
fn should_normalise_over_the_length_then_apply_the_style() {
    let runtime = initialize_test_runtime();
    let modulation = constant_style_modulation(&runtime, 1.0, 0.5);
    let input = upload(&runtime, &[1.0, 3.0], &[2, 1]);
    let style = upload(&runtime, &[0.0], &[1, 1]);
    let modulated = read_back(
        &runtime,
        &modulation
            .instance_modulate(&runtime, &input, &style)
            .expect("AdaIN"),
    );
    // Normalised to about (-1, 1), doubled, then shifted by 0.5.
    assert_close(&modulated, &[-1.5, 2.5], 1e-3);
}

// Edge: a constant channel normalises to zero, leaving only the shift.
#[test]
#[serial]
fn should_leave_only_the_shift_on_a_constant_channel() {
    let runtime = initialize_test_runtime();
    let modulation = constant_style_modulation(&runtime, 3.0, -0.25);
    let input = upload(&runtime, &[4.0, 4.0, 4.0], &[3, 1]);
    let style = upload(&runtime, &[0.0], &[1, 1]);
    let modulated = read_back(
        &runtime,
        &modulation
            .instance_modulate(&runtime, &input, &style)
            .expect("AdaIN"),
    );
    assert_close(&modulated, &[-0.25, -0.25, -0.25], 1e-5);
}

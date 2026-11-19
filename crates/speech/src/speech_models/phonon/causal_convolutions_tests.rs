use serial_test::serial;

use super::{CausalConv1d, CausalConvTranspose1d, CausalGeometry};
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_test_support::{
    assert_close, checkpoint_from_arrays, initialize_test_runtime, read_back, upload,
};

#[test]
#[serial]
fn should_see_only_the_past_and_keep_the_length() {
    let runtime = initialize_test_runtime();
    // Kernel (1, 10, 100): y[t] = x[t-2] + 10 x[t-1] + 100 x[t], with zeros before the start.
    let tensors = checkpoint_from_arrays(
        &runtime,
        &[(
            "conv.weight",
            upload(&runtime, &[1.0, 10.0, 100.0], &[1, 1, 3]),
        )],
    );
    let checkpoint = CheckpointTensors {
        runtime: &runtime,
        tensors: &tensors,
    };
    let convolution =
        CausalConv1d::load(&checkpoint, "conv", CausalGeometry::PLAIN).expect("loads");
    let output = read_back(
        &runtime,
        &convolution
            .forward(&runtime, &upload(&runtime, &[1.0, 2.0, 3.0], &[3, 1]))
            .expect("forward"),
    );
    assert_close(&output, &[100.0, 210.0, 321.0], 1e-4);
}

#[test]
#[serial]
fn should_upsample_to_exactly_stride_times_the_length() {
    let runtime = initialize_test_runtime();
    // Kernel (1, 2, 3, 4), stride 2: each step lays four samples, half overlapping the next.
    let tensors = checkpoint_from_arrays(
        &runtime,
        &[(
            "up.weight",
            upload(&runtime, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]),
        )],
    );
    let checkpoint = CheckpointTensors {
        runtime: &runtime,
        tensors: &tensors,
    };
    let upsampler = CausalConvTranspose1d::load(&checkpoint, "up", 2, 1).expect("loads");
    let output = read_back(
        &runtime,
        &upsampler
            .forward(&runtime, &upload(&runtime, &[1.0, 10.0], &[2, 1]))
            .expect("forward"),
    );
    // Full output (1, 2, 13, 24, 30, 40); the last two samples await a later step.
    assert_close(&output, &[1.0, 2.0, 13.0, 24.0], 1e-4);
}

// Edge: a prefix of the input gives a prefix of the output.
#[test]
#[serial]
fn should_give_a_prefix_of_the_output_for_a_prefix_of_the_input() {
    let runtime = initialize_test_runtime();
    let tensors = checkpoint_from_arrays(
        &runtime,
        &[(
            "up.weight",
            upload(&runtime, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]),
        )],
    );
    let checkpoint = CheckpointTensors {
        runtime: &runtime,
        tensors: &tensors,
    };
    let upsampler = CausalConvTranspose1d::load(&checkpoint, "up", 2, 1).expect("loads");
    let longer = read_back(
        &runtime,
        &upsampler
            .forward(&runtime, &upload(&runtime, &[1.0, 10.0, 5.0], &[3, 1]))
            .expect("forward"),
    );
    let shorter = read_back(
        &runtime,
        &upsampler
            .forward(&runtime, &upload(&runtime, &[1.0, 10.0], &[2, 1]))
            .expect("forward"),
    );
    assert_close(&longer[..2], &shorter[..2], 1e-4);
}

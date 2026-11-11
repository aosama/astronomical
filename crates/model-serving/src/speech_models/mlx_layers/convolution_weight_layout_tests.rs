//! The weight-layout conversions against hand-computed convolutions: if a
//! transpose or reshape were wrong, the taps would land on the wrong samples.

use serial_test::serial;

use super::{
    convolution_weight_for_mlx, pytorch_convolution_weight, transposed_convolution_weight_for_mlx,
};
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_test_support::{
    assert_close, checkpoint_from_arrays, initialize_test_runtime, read_back, upload,
};

#[test]
#[serial]
fn should_fold_a_weight_norm_pair_to_magnitude_times_unit_direction() {
    let runtime = initialize_test_runtime();
    // One output row with direction (3, 4): norm 5, so magnitude 10 gives (6, 8).
    let tensors = checkpoint_from_arrays(
        &runtime,
        &[
            ("layer.weight_v", upload(&runtime, &[3.0, 4.0], &[1, 2, 1])),
            ("layer.weight_g", upload(&runtime, &[10.0], &[1, 1, 1])),
        ],
    );
    let checkpoint = CheckpointTensors {
        runtime: &runtime,
        tensors: &tensors,
    };
    let folded = pytorch_convolution_weight(&checkpoint, "layer").expect("the pair folds");
    assert_close(&read_back(&runtime, &folded), &[6.0, 8.0], 1e-5);
}

#[test]
#[serial]
fn should_use_a_plain_weight_unchanged() {
    let runtime = initialize_test_runtime();
    let tensors = checkpoint_from_arrays(
        &runtime,
        &[("layer.weight", upload(&runtime, &[1.5, -2.0], &[1, 2, 1]))],
    );
    let checkpoint = CheckpointTensors {
        runtime: &runtime,
        tensors: &tensors,
    };
    let weight = pytorch_convolution_weight(&checkpoint, "layer").expect("the weight reads");
    assert_close(&read_back(&runtime, &weight), &[1.5, -2.0], 0.0);
}

// Negative: neither a weight nor a weight-norm pair.
#[test]
#[serial]
fn should_refuse_a_layer_without_weights() {
    let runtime = initialize_test_runtime();
    let tensors = checkpoint_from_arrays(
        &runtime,
        &[("other.weight", upload(&runtime, &[1.0], &[1, 1, 1]))],
    );
    let checkpoint = CheckpointTensors {
        runtime: &runtime,
        tensors: &tensors,
    };
    assert!(pytorch_convolution_weight(&checkpoint, "layer").is_err());
}

#[test]
#[serial]
fn should_convolve_with_the_pytorch_tap_order() {
    let runtime = initialize_test_runtime();
    // PyTorch weight [out 1, in 1, kernel 2] = (1, 10): y[t] = x[t] + 10 x[t + 1].
    let mlx_weight =
        convolution_weight_for_mlx(&runtime, &upload(&runtime, &[1.0, 10.0], &[1, 1, 2]))
            .expect("layout");
    let input = upload(&runtime, &[1.0, 2.0, 3.0], &[1, 3, 1]);
    let convolved = runtime
        .conv1d(&input, &mlx_weight, 1, 0, 1, 1)
        .expect("convolution");
    assert_close(&read_back(&runtime, &convolved), &[21.0, 32.0], 1e-5);
}

#[test]
#[serial]
fn should_overlap_add_a_transposed_convolution_as_pytorch_does() {
    let runtime = initialize_test_runtime();
    // PyTorch weight [in 1, out 1, kernel 3] = (1, 2, 3), stride 2: input (1, 10)
    // places (1, 2, 3) at 0 and (10, 20, 30) at 2, overlapping at 2.
    let mlx_weight = transposed_convolution_weight_for_mlx(
        &runtime,
        &upload(&runtime, &[1.0, 2.0, 3.0], &[1, 1, 3]),
        1,
    )
    .expect("layout");
    let input = upload(&runtime, &[1.0, 10.0], &[1, 2, 1]);
    let upsampled = runtime
        .conv_transpose1d(&input, &mlx_weight, 2, 0, 1, 0, 1)
        .expect("transposed convolution");
    assert_close(
        &read_back(&runtime, &upsampled),
        &[1.0, 2.0, 13.0, 20.0, 30.0],
        1e-5,
    );
}

// Boundary: depthwise (groups equal to channels) keeps each channel's own kernel.
#[test]
#[serial]
fn should_keep_each_channels_kernel_in_a_depthwise_transposed_convolution() {
    let runtime = initialize_test_runtime();
    // PyTorch [in 2, out/groups 1, kernel 2]: channel 0 kernel (1, 2), channel 1 kernel (5, 7).
    let pytorch_weight = upload(&runtime, &[1.0, 2.0, 5.0, 7.0], &[2, 1, 2]);
    let mlx_weight =
        transposed_convolution_weight_for_mlx(&runtime, &pytorch_weight, 2).expect("layout");
    // One step: channel 0 holds 1, channel 1 holds 10.
    let input = upload(&runtime, &[1.0, 10.0], &[1, 1, 2]);
    let upsampled = runtime
        .conv_transpose1d(&input, &mlx_weight, 2, 0, 1, 0, 2)
        .expect("transposed convolution");
    // [1 batch, 2 samples, 2 channels]: (1, 50), (2, 70).
    assert_close(
        &read_back(&runtime, &upsampled),
        &[1.0, 50.0, 2.0, 70.0],
        1e-5,
    );
}

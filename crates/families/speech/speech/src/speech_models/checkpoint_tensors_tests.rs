use mlx_bridge::MlxDtype;

use super::{CheckpointTensors, weights_file_bytes};
use crate::speech_models::speech_test_support::{
    checkpoint_from_arrays, initialize_test_runtime, read_back, upload,
};

// Positive: a checkpoint published in bfloat16 is read as float32, values
// kept (each is exact in bfloat16).
#[test]
fn should_read_a_bfloat16_tensor_as_float32() {
    let runtime = initialize_test_runtime();
    let exact_in_bfloat16 = [0.5_f32, -1.25, 3.0, 0.0];
    let float_weights = upload(&runtime, &exact_in_bfloat16, &[2, 2]);
    let bfloat16_weights = runtime
        .astype(&float_weights, MlxDtype::BFloat16)
        .expect("the weights cast to bfloat16");
    let safetensors = checkpoint_from_arrays(&runtime, &[("projection.weight", bfloat16_weights)]);
    let checkpoint = CheckpointTensors {
        runtime: &runtime,
        tensors: &safetensors,
    };

    let read_weights = checkpoint
        .tensor("projection.weight")
        .expect("the tensor is in the checkpoint");

    assert_eq!(read_weights.dtype(), MlxDtype::Float32);
    assert_eq!(read_back(&runtime, &read_weights), exact_in_bfloat16);
}

// Negative: a missing tensor is an artifact error that names it.
#[test]
fn should_name_a_missing_tensor() {
    let runtime = initialize_test_runtime();
    let safetensors = checkpoint_from_arrays(
        &runtime,
        &[("projection.weight", upload(&runtime, &[1.0], &[1]))],
    );
    let checkpoint = CheckpointTensors {
        runtime: &runtime,
        tensors: &safetensors,
    };

    let Err(missing_error) = checkpoint.tensor("projection.bias") else {
        panic!("a missing tensor must fail");
    };

    assert!(
        missing_error.to_string().contains("projection.bias"),
        "{missing_error}"
    );
}

// Edge: an optional tensor that is absent is `None`, not an error.
#[test]
fn should_return_none_for_an_absent_optional_tensor() {
    let runtime = initialize_test_runtime();
    let safetensors = checkpoint_from_arrays(
        &runtime,
        &[("projection.weight", upload(&runtime, &[1.0], &[1]))],
    );
    let checkpoint = CheckpointTensors {
        runtime: &runtime,
        tensors: &safetensors,
    };

    assert!(
        checkpoint
            .optional_tensor("projection.bias")
            .expect("an absent optional tensor is not an error")
            .is_none()
    );
    assert!(
        checkpoint
            .optional_tensor("projection.weight")
            .expect("a present tensor reads")
            .is_some()
    );
}

// Positive and edge: a weight file's size, and zero for a missing one.
#[test]
fn should_measure_a_weight_file_and_count_a_missing_one_as_empty() {
    let checkpoint_directory = tempfile::tempdir().expect("a temporary directory");
    let weights_path = checkpoint_directory.path().join("model.safetensors");
    assert_eq!(weights_file_bytes(&weights_path), 0);
    std::fs::write(&weights_path, [0_u8; 1234]).expect("a weight file");
    assert_eq!(weights_file_bytes(&weights_path), 1234);
}

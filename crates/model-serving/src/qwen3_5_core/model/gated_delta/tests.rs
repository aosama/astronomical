use astronomical_mlx_c_rust::MlxArray;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::normalize_linear_attention_convolution_weight;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the convolution layout test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

#[test]
#[serial]
fn should_transpose_published_pytorch_conv1d_weights_into_the_mlx_layout() {
    let runtime = test_runtime();
    let published_weight_values = (0..12)
        .map(|weight_index| weight_index as f32)
        .collect::<Vec<_>>();
    let published_weight = runtime
        .array_from_f32(&published_weight_values, &[4, 1, 3])
        .expect("the published PyTorch convolution weight shape should be valid");

    let mlx_layout_weight =
        normalize_linear_attention_convolution_weight(&runtime, &published_weight, 4, 3)
            .expect("the published Conv1d layout should transpose to the MLX layout");

    assert_eq!(mlx_layout_weight.shape(), [4, 3, 1]);
    assert_eq!(
        mlx_layout_weight
            .to_vec_f32()
            .expect("the normalized convolution weights should be readable"),
        published_weight_values
    );
}

#[test]
#[serial]
fn should_reject_a_conv1d_weight_shape_outside_both_published_layouts() {
    let runtime = test_runtime();
    let malformed_convolution_weight = MlxArray::from_f32(&[0.0; 24], &[4, 2, 3])
        .expect("the malformed shape should still form a valid MLX array");

    assert!(
        normalize_linear_attention_convolution_weight(
            &runtime,
            &malformed_convolution_weight,
            4,
            3,
        )
        .is_err()
    );
}

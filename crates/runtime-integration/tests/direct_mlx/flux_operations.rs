use astronomical_runtime_integration::{MlxDtype, MlxRuntimeError};

use crate::common::runtime_test_support::{assert_f32_close, runtime};

#[test]
fn should_run_flux_vae_asymmetric_padding_and_channel_last_convolution() {
    let runtime = runtime();
    let image = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1])
        .expect("the channel-last VAE image should be valid");
    let padded_image = runtime
        .pad(&image, &[1, 2], &[0, 0], &[1, 1], 0.0)
        .expect("VAE asymmetric padding should build a valid graph");
    let weights = runtime
        .array_from_f32(&[1.0; 4], &[1, 2, 2, 1])
        .expect("the channel-last convolution weights should be valid");

    let convolved_image = runtime
        .conv2d(&padded_image, &weights, [1, 1], [0, 0], [1, 1], 1)
        .expect("the VAE convolution should build a valid graph");

    assert_eq!(padded_image.shape(), vec![1, 3, 3, 1]);
    assert_eq!(convolved_image.shape(), vec![1, 2, 2, 1]);
    assert_eq!(
        convolved_image
            .to_vec_f32()
            .expect("the VAE convolution should evaluate as float32"),
        vec![10.0, 6.0, 7.0, 4.0]
    );
}

#[test]
fn should_generate_repeatable_and_seed_distinct_request_local_flux_noise() {
    let runtime = runtime();
    let request_key = runtime
        .random_key(42)
        .expect("the request seed should create an MLX key");
    let different_request_key = runtime
        .random_key(43)
        .expect("the different request seed should create an MLX key");

    let first_noise = runtime
        .random_normal(&[1, 2, 2, 4], MlxDtype::Float32, 0.0, 1.0, &request_key)
        .expect("request-keyed normal sampling should build a valid graph");
    let repeated_noise = runtime
        .random_normal(&[1, 2, 2, 4], MlxDtype::Float32, 0.0, 1.0, &request_key)
        .expect("reusing the request key should build the same sampling graph");
    let different_noise = runtime
        .random_normal(
            &[1, 2, 2, 4],
            MlxDtype::Float32,
            0.0,
            1.0,
            &different_request_key,
        )
        .expect("a different request key should build a valid sampling graph");

    let first_values = first_noise
        .to_vec_f32()
        .expect("the first noise tensor should evaluate");
    assert_eq!(first_noise.shape(), vec![1, 2, 2, 4]);
    assert_eq!(first_noise.dtype(), MlxDtype::Float32);
    assert_eq!(
        first_values,
        repeated_noise
            .to_vec_f32()
            .expect("the repeated noise tensor should evaluate")
    );
    assert_ne!(
        first_values,
        different_noise
            .to_vec_f32()
            .expect("the different-seed noise tensor should evaluate")
    );
}

#[test]
fn should_apply_flux_normalization_and_elementwise_bounds() {
    let runtime = runtime();
    let hidden_states = runtime
        .array_from_f32(&[1.0, 3.0], &[1, 2])
        .expect("the hidden states should be valid");
    let normalized_states = runtime
        .layer_norm_without_weight_and_bias(&hidden_states, 0.0)
        .expect("non-affine LayerNorm should build a valid graph");
    let squared_values = runtime
        .array_from_f32(&[0.0, 1.0, 4.0, 9.0], &[4])
        .expect("the squared values should be valid");
    let square_roots = runtime
        .sqrt(&squared_values)
        .expect("square root should build a valid graph");
    let clipped_values = runtime
        .clip(&square_roots, 1.0, 2.0)
        .expect("clipping should build a valid graph");
    let filled_values = runtime
        .full(&[2, 2], 7.0, MlxDtype::Float32)
        .expect("full should build a valid graph");

    assert_f32_close(
        &normalized_states
            .to_vec_f32()
            .expect("the normalized states should evaluate"),
        &[-1.0, 1.0],
    );
    assert_eq!(
        clipped_values
            .to_vec_f32()
            .expect("the clipped roots should evaluate"),
        vec![1.0, 1.0, 2.0, 2.0]
    );
    assert_eq!(
        filled_values
            .to_vec_f32()
            .expect("the filled tensor should evaluate"),
        vec![7.0; 4]
    );
}

#[test]
fn should_run_flux_fused_attention_without_exposing_score_matrices() {
    let runtime = runtime();
    let queries = runtime
        .array_from_f32(&[1.0, 1.0], &[1, 1, 2, 1])
        .expect("the FLUX query tensor should be valid");
    let keys = runtime
        .array_from_f32(&[0.0, 0.0], &[1, 1, 2, 1])
        .expect("the FLUX key tensor should be valid");
    let values = runtime
        .array_from_f32(&[2.0, 4.0], &[1, 1, 2, 1])
        .expect("the FLUX value tensor should be valid");

    let attention_output = runtime
        .scaled_dot_product_attention(&queries, &keys, &values, 1.0)
        .expect("fused FLUX attention should build one output graph");

    assert_eq!(attention_output.shape(), vec![1, 1, 2, 1]);
    assert_eq!(
        attention_output
            .to_vec_f32()
            .expect("the fused FLUX attention output should evaluate"),
        vec![3.0, 3.0]
    );
}

#[test]
fn should_transfer_evaluated_flux_pixels_as_uint8_without_scalar_reads() {
    let runtime = runtime();
    let float_pixels = runtime
        .array_from_f32(&[0.0, 127.0, 255.0, 64.0], &[2, 2])
        .expect("the float pixels should be valid");
    let uint8_pixels = runtime
        .astype(&float_pixels, MlxDtype::UInt8)
        .expect("pixel conversion should build a valid graph");
    let contiguous_pixels = runtime
        .build_contiguous_row_major_copy(&uint8_pixels)
        .expect("the pixel transfer should build one contiguous graph");
    contiguous_pixels
        .evaluate()
        .expect("the contiguous pixels should evaluate once");

    assert_eq!(
        contiguous_pixels
            .copy_evaluated_u8_values()
            .expect("the evaluated pixels should copy in one bounded transfer"),
        vec![0, 127, 255, 64]
    );
    assert_eq!(
        runtime
            .copy_u8_values(&uint8_pixels)
            .expect("the convenience transfer should preserve pixel order"),
        vec![0, 127, 255, 64]
    );
}

#[test]
fn should_reject_invalid_flux_operation_arguments_before_building_graphs() {
    let runtime = runtime();
    let request_key = runtime
        .random_key(7)
        .expect("the request seed should create an MLX key");
    let image = runtime
        .zeros(&[1, 2, 2, 1], MlxDtype::Float32)
        .expect("the validation image should be valid");
    let scalar = runtime
        .array_from_f32(&[1.0], &[])
        .expect("the validation scalar should be valid");
    let invalid_conv2d_weights = runtime
        .zeros(&[1, 2, 1], MlxDtype::Float32)
        .expect("the invalid-rank weights should still be valid arrays");

    assert!(matches!(
        runtime.random_normal(&[-1], MlxDtype::Float32, 0.0, 1.0, &request_key),
        Err(MlxRuntimeError::RuntimeOperation { .. })
    ));
    assert!(matches!(
        runtime.pad(&image, &[1, 1], &[0, 0], &[1, 1], 0.0),
        Err(MlxRuntimeError::RuntimeOperation { .. })
    ));
    assert!(matches!(
        runtime.conv2d(&image, &invalid_conv2d_weights, [1, 1], [0, 0], [1, 1], 1),
        Err(MlxRuntimeError::RuntimeOperation { .. })
    ));
    assert!(matches!(
        runtime.layer_norm_without_weight_and_bias(&scalar, f32::NAN),
        Err(MlxRuntimeError::RuntimeOperation { .. })
    ));
    assert!(matches!(
        runtime.clip(&image, 2.0, 1.0),
        Err(MlxRuntimeError::RuntimeOperation { .. })
    ));
}

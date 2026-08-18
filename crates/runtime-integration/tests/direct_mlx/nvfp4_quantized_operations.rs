//! Direct GPU correctness contracts for MLX's native NVFP4 representation.

use astronomical_runtime_integration::MlxDtype;

use crate::common::runtime_test_support::runtime;

#[test]
fn should_match_dense_reference_for_nvfp4_split_k_quantized_matmul() {
    // Sixty-four rows exceed every MLX device-specific vector limit. With this
    // geometry, split-K retains two 32-wide partitions instead of escaping to QMM.
    const ACTIVATION_ROWS: i32 = 64;
    const INPUT_WIDTH: i32 = 64;
    const OUTPUT_WIDTH: i32 = 128;
    const MAX_ABSOLUTE_ERROR: f32 = 1e-3;

    let runtime = runtime();
    let activation_values = (0..ACTIVATION_ROWS * INPUT_WIDTH)
        .map(|activation_index| 0.01 * ((activation_index % 19) as f32 - 9.0))
        .collect::<Vec<_>>();
    let activations = runtime
        .array_from_f32(&activation_values, &[ACTIVATION_ROWS, INPUT_WIDTH])
        .expect("the NVFP4 split-K activations should be valid");
    let source_weight_values = (0..OUTPUT_WIDTH * INPUT_WIDTH)
        .map(|weight_index| 0.025 * ((weight_index % 17) as f32 - 8.0))
        .collect::<Vec<_>>();
    let source_weights = runtime
        .array_from_f32(&source_weight_values, &[OUTPUT_WIDTH, INPUT_WIDTH])
        .expect("the NVFP4 source weights should be valid");
    let (packed_weights, scales) = runtime
        .quantize_nvfp4(&source_weights)
        .expect("the source weights should quantize as NVFP4");

    let quantized_product = runtime
        .quantized_matmul_nvfp4(&activations, &packed_weights, &scales, true)
        .expect("the NVFP4 split-K graph should build");
    let dequantized_weights = runtime
        .dequantize_nvfp4(&packed_weights, &scales, MlxDtype::Float32)
        .expect("the dense reference weights should dequantize");
    let transposed_reference_weights = runtime
        .transpose_axes(&dequantized_weights, &[1, 0])
        .expect("the dense reference weights should transpose");
    let reference_product = runtime
        .matmul(&activations, &transposed_reference_weights)
        .expect("the dense reference graph should build");
    let actual_values = quantized_product
        .to_vec_f32()
        .expect("the NVFP4 split-K product should evaluate");
    let expected_values = reference_product
        .to_vec_f32()
        .expect("the dense reference product should evaluate");

    assert_eq!(
        quantized_product.shape(),
        vec![ACTIVATION_ROWS, OUTPUT_WIDTH]
    );
    assert_eq!(actual_values.len(), expected_values.len());
    assert!(
        actual_values
            .iter()
            .all(|actual_value| actual_value.is_finite()),
        "NVFP4 split-K must not return NaN or infinity"
    );
    for (output_index, (actual_value, expected_value)) in
        actual_values.iter().zip(expected_values.iter()).enumerate()
    {
        let absolute_error = (*actual_value - *expected_value).abs();
        assert!(
            absolute_error <= MAX_ABSOLUTE_ERROR,
            "NVFP4 split-K output {output_index} was {actual_value}, expected {expected_value}, absolute error {absolute_error}"
        );
    }
}

#[test]
fn should_compute_nvfp4_scales_independently_for_each_sixteen_value_group() {
    const GROUP_COUNT: i32 = 64;
    const GROUP_SIZE: i32 = 16;
    const MAX_ABSOLUTE_ERROR: f32 = 1e-6;

    let runtime = runtime();
    let mut source_weight_values = Vec::with_capacity((GROUP_COUNT * GROUP_SIZE) as usize);
    for group_index in 0..GROUP_COUNT {
        // Adjacent extremes expose an accidental 32-lane maximum immediately:
        // the small group becomes unrepresentable when it shares the large scale.
        let group_weight = if group_index % 2 == 0 {
            6.0 * 2.0_f32.powi(-9)
        } else {
            6.0
        };
        source_weight_values.extend([group_weight; GROUP_SIZE as usize]);
    }
    let float32_source_weights = runtime
        .array_from_f32(&source_weight_values, &[GROUP_COUNT, GROUP_SIZE])
        .expect("the alternating NVFP4 scale groups should be valid");
    let source_weights = runtime
        .astype(&float32_source_weights, MlxDtype::BFloat16)
        .expect("the NVFP4 scale source should use the model execution dtype");
    let (packed_weights, scales) = runtime
        .quantize_nvfp4(&source_weights)
        .expect("the alternating groups should quantize as NVFP4");
    let restored_weights = runtime
        .dequantize_nvfp4(&packed_weights, &scales, MlxDtype::BFloat16)
        .expect("the alternating NVFP4 groups should dequantize");
    let float32_restored_weights = runtime
        .astype(&restored_weights, MlxDtype::Float32)
        .expect("the restored NVFP4 groups should cast for comparison");
    let actual_values = float32_restored_weights
        .to_vec_f32()
        .expect("the restored NVFP4 groups should evaluate");

    assert!(
        actual_values
            .iter()
            .all(|actual_value| actual_value.is_finite()),
        "NVFP4 scale reduction must not return NaN or infinity"
    );
    for (weight_index, (actual_value, expected_value)) in actual_values
        .iter()
        .zip(source_weight_values.iter())
        .enumerate()
    {
        let absolute_error = (*actual_value - *expected_value).abs();
        assert!(
            absolute_error <= MAX_ABSOLUTE_ERROR,
            "NVFP4 restored weight {weight_index} was {actual_value}, expected {expected_value}, absolute error {absolute_error}"
        );
    }
}

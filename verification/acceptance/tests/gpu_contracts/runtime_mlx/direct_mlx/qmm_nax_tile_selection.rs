use astronomical_runtime_integration::MlxRuntime;

use crate::common::runtime_test_support;
use mlx_c_rust::{MlxArray, MlxDtype};

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

/// Attention projection shapes: (output_rows, input_columns).
const PROJECTION_SHAPES: [(usize, usize); 3] = [(8_192, 2_048), (4_096, 2_048), (2_048, 4_096)];
const TOKEN_COUNTS: [usize; 5] = [32, 33, 64, 65, 512];

#[test]
#[serial_test::serial]
#[ignore = "checks production transposed-prefill kernels against a dense reference on a real graphics processor"]
fn should_match_dense_matmul_across_production_prefill_tile_boundaries() {
    let runtime = runtime_test_support::runtime();

    for (output_rows, input_columns) in PROJECTION_SHAPES {
        let quantized_weights = build_quantized_weights(&runtime, output_rows, input_columns);
        let dequantized_weights = runtime
            .dequantize_affine(
                &quantized_weights.packed_words,
                &quantized_weights.scales,
                &quantized_weights.biases,
                GROUP_SIZE,
                BITS,
            )
            .expect("the reference weights should dequantize");
        let float32_reference_weights = runtime
            .astype(&dequantized_weights, MlxDtype::Float32)
            .expect("the reference weights should cast to float32");
        let transposed_reference_weights = runtime
            .transpose_axes(&float32_reference_weights, &[1, 0])
            .expect("the dense reference weights should transpose");

        for token_count in TOKEN_COUNTS {
            let activations = build_activations(&runtime, token_count, input_columns);
            let quantized_product =
                build_quantized_product(&runtime, &activations, &quantized_weights);
            let float32_reference_activations = runtime
                .astype(&activations, MlxDtype::Float32)
                .expect("the reference activations should cast to float32");
            let reference_product = runtime
                .matmul(
                    &float32_reference_activations,
                    &transposed_reference_weights,
                )
                .expect("the independent dense reference should build");
            let actual_values = materialize_output(&runtime, &quantized_product);
            let reference_values = materialize_output(&runtime, &reference_product);
            assert_eq!(
                quantized_product.shape(),
                vec![token_count as i32, output_rows as i32]
            );
            assert_eq!(actual_values.len(), reference_values.len());
            assert!(
                actual_values
                    .iter()
                    .chain(&reference_values)
                    .all(|product_element| product_element.is_finite())
            );
            let reference_magnitude = reference_values
                .iter()
                .fold(0.0f32, |largest, reference| largest.max(reference.abs()));
            let maximum_absolute_error = actual_values
                .iter()
                .zip(&reference_values)
                .fold(0.0f32, |largest, (actual, reference)| {
                    largest.max((actual - reference).abs())
                });
            let relative_error =
                maximum_absolute_error / reference_magnitude.max(f32::MIN_POSITIVE);
            assert!(
                relative_error < 1e-2,
                "production prefill diverged from dense matmul on shape ({output_rows},{input_columns}) tokens={token_count}: relative_error={relative_error}"
            );
        }
    }
}

struct QuantizedWeights {
    packed_words: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

fn build_quantized_weights(
    runtime: &MlxRuntime,
    output_rows: usize,
    input_columns: usize,
) -> QuantizedWeights {
    let words_per_row = input_columns * BITS as usize / 32;
    let weight_words = deterministic_weight_words(output_rows * words_per_row);
    let packed_words = runtime
        .array_from_u32(&weight_words, &[output_rows as i32, words_per_row as i32])
        .expect("the packed quantized weights should be valid");

    let groups_per_row = input_columns / GROUP_SIZE as usize;
    let scale_values = vec![0.01f32; output_rows * groups_per_row];
    let float32_scales = runtime
        .array_from_f32(&scale_values, &[output_rows as i32, groups_per_row as i32])
        .expect("the affine scales should be valid");
    let scales = runtime
        .astype(&float32_scales, MlxDtype::Float16)
        .expect("the scales should cast to float16");
    let biases = runtime
        .astype(&float32_scales, MlxDtype::Float16)
        .expect("the affine biases should cast to float16");

    runtime
        .evaluate_arrays(&[&packed_words, &scales, &biases])
        .expect("the quantized weights should materialize");
    QuantizedWeights {
        packed_words,
        scales,
        biases,
    }
}

fn build_activations(runtime: &MlxRuntime, token_count: usize, input_columns: usize) -> MlxArray {
    let activation_values = deterministic_activation_values(token_count * input_columns);
    let float32_activations = runtime
        .array_from_f32(
            &activation_values,
            &[token_count as i32, input_columns as i32],
        )
        .expect("the activation matrix should be valid");
    let activations = runtime
        .astype(&float32_activations, MlxDtype::Float16)
        .expect("the activations should cast to float16");
    runtime
        .evaluate_arrays(&[&activations])
        .expect("the activations should materialize");
    activations
}

fn build_quantized_product(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    quantized_weights: &QuantizedWeights,
) -> MlxArray {
    runtime
        .quantized_matmul_affine(
            activations,
            &quantized_weights.packed_words,
            &quantized_weights.scales,
            &quantized_weights.biases,
            true,
            GROUP_SIZE,
            BITS,
        )
        .expect("the transposed affine quantized matmul should build")
}

fn materialize_output(runtime: &MlxRuntime, output: &MlxArray) -> Vec<f32> {
    let float32_output = runtime
        .astype(output, MlxDtype::Float32)
        .expect("the output should cast to float32");
    float32_output
        .to_vec_f32()
        .expect("the output should read back as float32")
}

fn deterministic_activation_values(element_count: usize) -> Vec<f32> {
    let mut generator_state = 0x2545_F491_4F6C_DD1Du64;
    (0..element_count)
        .map(|_| {
            generator_state = generator_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit_interval = ((generator_state >> 33) as f32) / ((1u32 << 31) as f32);
            unit_interval * 2.0 - 1.0
        })
        .collect()
}

fn deterministic_weight_words(word_count: usize) -> Vec<u32> {
    let mut generator_state = 0x9E37_79B9_7F4A_7C15u64;
    (0..word_count)
        .map(|_| {
            generator_state = generator_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (generator_state >> 32) as u32
        })
        .collect()
}

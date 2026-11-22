use mlx_bridge::{MlxArray, MlxBindingsContext, MlxDtype};

use super::quantized_matmul_affine_through_float32;

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
const INPUT_WIDTH: usize = 128;
const OUTPUT_WIDTH: usize = 32;

/// Deterministic, bounded values from a fixed integer pattern.
fn patterned_values(value_count: usize, pattern_offset: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(value_count);
    for value_index in 0..value_count {
        let pattern_step = (value_index * 37 + pattern_offset) % 97;
        values.push((pattern_step as f32 - 48.0) / 32.0);
    }
    values
}

fn quantized_weights(context: &MlxBindingsContext) -> (MlxArray, MlxArray, MlxArray) {
    let weights = context
        .array_from_f32(
            &patterned_values(OUTPUT_WIDTH * INPUT_WIDTH, 11),
            &[OUTPUT_WIDTH as i32, INPUT_WIDTH as i32],
        )
        .expect("the weights build");
    let bfloat16_weights = context
        .astype(&weights, MlxDtype::BFloat16)
        .expect("the weights cast");
    let quantized = context
        .quantize_affine(&bfloat16_weights, GROUP_SIZE, BITS)
        .expect("the weights quantize");
    (quantized.0, quantized.1, quantized.2)
}

fn activations(context: &MlxBindingsContext, row_count: usize, dtype: MlxDtype) -> MlxArray {
    let float32_activations = context
        .array_from_f32(
            &patterned_values(row_count * INPUT_WIDTH, 5),
            &[row_count as i32, INPUT_WIDTH as i32],
        )
        .expect("the activations build");
    context
        .astype(&float32_activations, dtype)
        .expect("the activations cast")
}

fn float32_values(context: &MlxBindingsContext, array: &MlxArray) -> Vec<f32> {
    context
        .astype(array, MlxDtype::Float32)
        .and_then(|float32_array| float32_array.to_vec_f32())
        .expect("the array evaluates")
}

/// The exact projection: float32 dequantized weights times the BF16-rounded
/// activations widened to float32, with nothing rounded until the end.
fn exact_reference(
    context: &MlxBindingsContext,
    bfloat16_activations: &MlxArray,
    quantized: (&MlxArray, &MlxArray, &MlxArray),
) -> Vec<f32> {
    let (packed_weights, scales, biases) = quantized;
    let float32_weights = context
        .dequantize_affine(
            packed_weights,
            &context
                .astype(scales, MlxDtype::Float32)
                .expect("scales widen"),
            &context
                .astype(biases, MlxDtype::Float32)
                .expect("biases widen"),
            GROUP_SIZE,
            BITS,
        )
        .expect("the weights dequantize");
    let float32_activations = context
        .astype(bfloat16_activations, MlxDtype::Float32)
        .expect("the activations widen");
    let transposed_weights = context
        .transpose_axes(&float32_weights, &[1, 0])
        .expect("the weights transpose");
    float32_values(
        context,
        &context
            .matmul(&float32_activations, &transposed_weights)
            .expect("the reference runs"),
    )
}

fn largest_difference(left_values: &[f32], right_values: &[f32]) -> f32 {
    let mut largest_difference = 0.0_f32;
    for (left_value, right_value) in left_values.iter().zip(right_values) {
        largest_difference = largest_difference.max((left_value - right_value).abs());
    }
    largest_difference
}

#[test]
fn should_keep_the_activations_dtype_and_round_the_exact_projection_once() {
    let context = MlxBindingsContext::on_cpu().expect("the CPU stream exists");
    let (packed_weights, scales, biases) = quantized_weights(&context);
    let bfloat16_activations = activations(&context, 3, MlxDtype::BFloat16);

    let through_float32 = quantized_matmul_affine_through_float32(
        &context,
        &bfloat16_activations,
        &packed_weights,
        &scales,
        &biases,
        true,
        GROUP_SIZE,
        BITS,
    )
    .expect("the float32 route runs");
    let half_precision = context
        .quantized_matmul_affine(
            &bfloat16_activations,
            &packed_weights,
            &scales,
            &biases,
            true,
            GROUP_SIZE,
            BITS,
        )
        .expect("the half-precision route runs");
    let exact_values = exact_reference(
        &context,
        &bfloat16_activations,
        (&packed_weights, &scales, &biases),
    );

    assert_eq!(through_float32.dtype(), MlxDtype::BFloat16);
    assert_eq!(through_float32.shape(), vec![3, OUTPUT_WIDTH as i32]);
    let mut largest_magnitude = 0.0_f32;
    for exact_value in &exact_values {
        largest_magnitude = largest_magnitude.max(exact_value.abs());
    }
    let float32_route_error =
        largest_difference(&float32_values(&context, &through_float32), &exact_values);
    let half_route_error =
        largest_difference(&float32_values(&context, &half_precision), &exact_values);
    // One BF16 rounding of the result moves it at most half a step, a 1/256
    // share of the largest output; the CPU's BF16 kernel also rounds each
    // dequantized weight, so the float32 route may only be closer.
    assert!(
        float32_route_error <= largest_magnitude / 256.0,
        "float32 route error {float32_route_error} at magnitude {largest_magnitude}"
    );
    assert!(
        float32_route_error <= half_route_error,
        "float32 route error {float32_route_error} exceeds the BF16 route's {half_route_error}"
    );
}

#[test]
fn should_take_the_ordinary_route_for_float32_activations() {
    let context = MlxBindingsContext::on_cpu().expect("the CPU stream exists");
    let (packed_weights, scales, biases) = quantized_weights(&context);
    let float32_activations = activations(&context, 1, MlxDtype::Float32);

    let through_float32 = quantized_matmul_affine_through_float32(
        &context,
        &float32_activations,
        &packed_weights,
        &scales,
        &biases,
        true,
        GROUP_SIZE,
        BITS,
    )
    .expect("the route runs");
    let ordinary = context
        .quantized_matmul_affine(
            &float32_activations,
            &packed_weights,
            &scales,
            &biases,
            true,
            GROUP_SIZE,
            BITS,
        )
        .expect("the ordinary route runs");

    assert_eq!(
        float32_values(&context, &through_float32),
        float32_values(&context, &ordinary),
        "float32 activations must not be cast at all"
    );
}

#[test]
fn should_reject_mismatched_arguments_like_mlx_does() {
    let context = MlxBindingsContext::on_cpu().expect("the CPU stream exists");
    let (packed_weights, scales, biases) = quantized_weights(&context);
    let too_narrow_activations = context
        .array_from_f32(
            &patterned_values(INPUT_WIDTH / 2, 0),
            &[1, (INPUT_WIDTH / 2) as i32],
        )
        .expect("the activations build");

    assert!(
        quantized_matmul_affine_through_float32(
            &context,
            &too_narrow_activations,
            &packed_weights,
            &scales,
            &biases,
            true,
            GROUP_SIZE,
            BITS,
        )
        .is_err(),
        "an input width that does not match the weights must fail"
    );
}

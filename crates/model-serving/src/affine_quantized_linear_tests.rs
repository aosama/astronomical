//! Contracts for the shared 4-bit projection: binding refuses any other
//! packing, projection matches the dense form, and an embedding gathers only
//! the rows it needs. Values come from the Romeo and Juliet fixture.

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

use super::{
    AFFINE_QUANTIZATION_BITS, AFFINE_QUANTIZATION_GROUP_SIZE, AffineQuantizedLinear,
    AffineQuantizedLinearError,
};

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../verification/acceptance/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);
const OUTPUT_WIDTH: usize = 32;
const INPUT_WIDTH: usize = 128;

fn initialize_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// BF16 values in about [-1.5, 1.5] drawn from the play's bytes.
fn shakespeare_bf16(runtime: &MlxRuntime, shape: &[i32], byte_offset: usize) -> MlxArray {
    let value_count = shape.iter().product::<i32>() as usize;
    let values = ROMEO_AND_JULIET_SOURCE
        .bytes()
        .skip(byte_offset)
        .take(value_count)
        .map(|source_byte| (f32::from(source_byte) - 96.0) / 32.0)
        .collect::<Vec<_>>();
    let float32_values = runtime
        .array_from_f32(&values, shape)
        .expect("the fixture values should upload");
    runtime
        .astype(&float32_values, MlxDtype::BFloat16)
        .expect("the fixture values should convert to BF16")
}

fn packed_projection(runtime: &MlxRuntime) -> ([MlxArray; 3], MlxArray) {
    let dense_weight = shakespeare_bf16(runtime, &[OUTPUT_WIDTH as i32, INPUT_WIDTH as i32], 0);
    let (packed_weight, scales, biases) = runtime
        .quantize_affine(
            &dense_weight,
            AFFINE_QUANTIZATION_GROUP_SIZE as i32,
            AFFINE_QUANTIZATION_BITS,
        )
        .expect("the fixture weight should quantize");
    let dequantized_weight = runtime
        .dequantize_affine(
            &packed_weight,
            &scales,
            &biases,
            AFFINE_QUANTIZATION_GROUP_SIZE as i32,
            AFFINE_QUANTIZATION_BITS,
        )
        .expect("the packed weight should dequantize");
    ([packed_weight, scales, biases], dequantized_weight)
}

fn float32_values(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    runtime
        .astype(array, MlxDtype::Float32)
        .and_then(|float32_array| float32_array.to_vec_f32())
        .expect("the array should evaluate")
}

#[test]
#[serial]
fn should_project_on_packed_weights_like_the_dequantized_dense_form() {
    let runtime = initialize_test_runtime();
    let (packed_tensors, dequantized_weight) = packed_projection(&runtime);
    let projection =
        AffineQuantizedLinear::bind("proj_out", packed_tensors, INPUT_WIDTH, OUTPUT_WIDTH)
            .expect("a correctly packed projection should bind");
    let activations = shakespeare_bf16(&runtime, &[1, 3, INPUT_WIDTH as i32], 4_096);

    let packed_output = projection
        .forward(&runtime, &activations)
        .expect("the packed projection should run");
    let transposed_weight = runtime
        .transpose_axes(&dequantized_weight, &[1, 0])
        .expect("the dense weight should transpose");
    let dense_output = runtime
        .matmul(&activations, &transposed_weight)
        .expect("the dense projection should run");

    assert_eq!(packed_output.shape(), vec![1, 3, OUTPUT_WIDTH as i32]);
    let largest_difference = float32_values(&runtime, &packed_output)
        .iter()
        .zip(float32_values(&runtime, &dense_output))
        .map(|(packed_value, dense_value)| (packed_value - dense_value).abs())
        .fold(0.0_f32, f32::max);
    // Both paths round to BF16 once; anything larger means a packing mismatch.
    assert!(largest_difference < 0.5, "difference {largest_difference}");
}

#[test]
#[serial]
fn should_gather_rows_exactly_as_dequantize_then_take() {
    let runtime = initialize_test_runtime();
    let (packed_tensors, dequantized_weight) = packed_projection(&runtime);
    let embedding =
        AffineQuantizedLinear::bind("embed_tokens", packed_tensors, INPUT_WIDTH, OUTPUT_WIDTH)
            .expect("the packed table should bind");
    let row_ids = MlxArray::from_u32(&[31, 0, 7], &[3]).expect("the row ids should upload");

    let gathered_rows = embedding
        .gather_dequantized_rows(&runtime, &row_ids)
        .expect("the rows should gather");
    let reference_rows = runtime
        .take_axis(&dequantized_weight, &row_ids, 0)
        .expect("the reference rows should gather");

    assert_eq!(gathered_rows.shape(), vec![3, INPUT_WIDTH as i32]);
    assert_eq!(
        float32_values(&runtime, &gathered_rows),
        float32_values(&runtime, &reference_rows)
    );
}

#[test]
#[serial]
fn should_refuse_a_triple_packed_for_another_input_width() {
    let runtime = initialize_test_runtime();
    let (packed_tensors, _) = packed_projection(&runtime);

    let refusal =
        AffineQuantizedLinear::bind("proj_out", packed_tensors, INPUT_WIDTH * 2, OUTPUT_WIDTH);

    assert!(matches!(
        refusal,
        Err(AffineQuantizedLinearError::TensorShape { tensor_name, .. }) if tensor_name == "proj_out.weight"
    ));
}

#[test]
#[serial]
fn should_refuse_activations_of_the_wrong_width_or_rank() {
    let runtime = initialize_test_runtime();
    let (packed_tensors, _) = packed_projection(&runtime);
    let projection =
        AffineQuantizedLinear::bind("proj_out", packed_tensors, INPUT_WIDTH, OUTPUT_WIDTH)
            .expect("a correctly packed projection should bind");

    for activations in [
        shakespeare_bf16(&runtime, &[1, 3, 64], 0),
        shakespeare_bf16(&runtime, &[INPUT_WIDTH as i32], 0),
    ] {
        assert!(matches!(
            projection.forward(&runtime, &activations),
            Err(AffineQuantizedLinearError::InputWidth { .. })
        ));
    }
}

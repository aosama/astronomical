//! One dense operation in the artifact's quantized form, prepared for both
//! engines, shared by the whole-model and hand-off journeys.

use astronomical_neural_engine::deterministic_values::{
    PROJECTION_WEIGHT_STANDARD_DEVIATION, bell_shaped_values,
};
use astronomical_neural_engine::half_precision::f32_slice_to_f16_bits;
use astronomical_neural_engine::projection_shape::ProjectionShape;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxDtype};

use super::journey_constants::PREFILL_CHUNK_TOKEN_COUNT;
use super::mlx_projection::{
    MlxProjectionForm, PRODUCTION_QUANTIZATION_GROUP_SIZE, token_major, upload,
};
use super::projection_fixture::ProjectionFixture;

pub(crate) struct DenseOperation {
    pub(crate) label: &'static str,
    pub(crate) input_channels: usize,
    pub(crate) output_channels: usize,
    pub(crate) layer_count: usize,
}

pub(crate) struct QuantizedOperation {
    pub(crate) fixture: ProjectionFixture,
    pub(crate) packed_weights: MlxArray,
    pub(crate) scales: MlxArray,
    pub(crate) biases: MlxArray,
    pub(crate) token_major_activations: MlxArray,
}

pub(crate) fn quantize_and_round_trip(
    runtime: &MlxRuntime,
    shape: ProjectionShape,
) -> (MlxArray, MlxArray, MlxArray, Vec<u16>) {
    let trained_like_weights = bell_shaped_values(
        ORNITH_WEIGHT_SEED,
        shape.weight_element_count(),
        PROJECTION_WEIGHT_STANDARD_DEVIATION,
    );
    let weight_shape = [
        shape.output_channels() as i32,
        shape.input_channels() as i32,
    ];
    let fp16_weights = upload(
        runtime,
        &trained_like_weights,
        &weight_shape,
        MlxDtype::Float16,
    );
    let (packed_weights, scales, biases) = runtime
        .quantize_affine(
            &fp16_weights,
            PRODUCTION_QUANTIZATION_GROUP_SIZE,
            ORNITH_QUANTIZATION_BITS,
        )
        .expect("weights quantize");
    let dequantized = runtime
        .dequantize_affine(
            &packed_weights,
            &scales,
            &biases,
            PRODUCTION_QUANTIZATION_GROUP_SIZE,
            ORNITH_QUANTIZATION_BITS,
        )
        .expect("weights dequantize");
    let dequantized_fp16 = runtime
        .astype(&dequantized, MlxDtype::Float16)
        .expect("fp16");
    let dequantized_single = runtime
        .astype(&dequantized_fp16, MlxDtype::Float32)
        .expect("f32");
    runtime
        .evaluate_arrays(&[&packed_weights, &scales, &biases, &dequantized_single])
        .expect("quantization evaluates");
    let dequantized_values = runtime
        .array_to_vec_f32(&dequantized_single)
        .expect("dequantized weights read back");
    (
        packed_weights,
        scales,
        biases,
        f32_slice_to_f16_bits(&dequantized_values),
    )
}

pub(crate) fn prepare_quantized_operation(
    runtime: &MlxRuntime,
    operation: &DenseOperation,
) -> QuantizedOperation {
    let shape = ProjectionShape::new(
        operation.input_channels,
        operation.output_channels,
        PREFILL_CHUNK_TOKEN_COUNT,
    )
    .expect("the Ornith operation shape is valid");
    let (packed_weights, scales, biases, fp16_weights) = quantize_and_round_trip(runtime, shape);
    let fixture = ProjectionFixture::with_fp16_weights(shape, fp16_weights);
    let token_major_activations = upload(
        runtime,
        &token_major(&fixture),
        &[1, shape.token_count() as i32, shape.input_channels() as i32],
        MlxDtype::Float16,
    );
    QuantizedOperation {
        fixture,
        packed_weights,
        scales,
        biases,
        token_major_activations,
    }
}

/// Rows `first_row..` of a quantized tensor, copied contiguous. Affine groups
/// run along the input axis, so whole rows slice cleanly.
pub(crate) fn trailing_rows(
    runtime: &MlxRuntime,
    quantized_tensor: &MlxArray,
    first_row: usize,
) -> MlxArray {
    let tensor_shape = quantized_tensor.shape();
    let sliced = runtime
        .slice(
            quantized_tensor,
            &[first_row as i32, 0],
            &tensor_shape,
            &[1, 1],
        )
        .expect("rows slice");
    let contiguous = runtime.copy(&sliced).expect("rows copy");
    runtime
        .evaluate_arrays(&[&contiguous])
        .expect("rows evaluate");
    contiguous
}

pub(crate) fn quantized_form<'arrays>(
    activations: &'arrays MlxArray,
    packed_weights: &'arrays MlxArray,
    scales: &'arrays MlxArray,
    biases: &'arrays MlxArray,
) -> MlxProjectionForm<'arrays> {
    MlxProjectionForm::QuantizedAffine {
        token_major_activations: activations,
        packed_weights,
        scales,
        biases,
        bits: ORNITH_QUANTIZATION_BITS,
    }
}

/// An evaluated or lazy `[1, tokens, channels]` array as `[channels][tokens]`.
pub(crate) fn channel_major_values(
    runtime: &MlxRuntime,
    projected: &MlxArray,
    channel_count: usize,
) -> Vec<f32> {
    let single = runtime
        .astype(projected, MlxDtype::Float32)
        .expect("result converts");
    let token_major_values = runtime
        .array_to_vec_f32(&single)
        .expect("result reads back");
    let token_count = token_major_values.len() / channel_count;
    let mut channel_major_values = vec![0.0_f32; token_major_values.len()];
    for token_index in 0..token_count {
        for channel_index in 0..channel_count {
            channel_major_values[channel_index * token_count + token_index] =
                token_major_values[token_index * channel_count + channel_index];
        }
    }
    channel_major_values
}

const ORNITH_QUANTIZATION_BITS: i32 = 4;
const ORNITH_WEIGHT_SEED: u64 = 0x5eed_0909;

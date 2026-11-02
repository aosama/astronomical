//! A/B for a dense model: Ornith-1.5-9B (`qwen3_5`, 32 layers: 24 DeltaNet
//! and 8 full attention, hidden 4,096, MLP intermediate 12,288).
//!
//! - **A**: MLX alone on the GPU, in the artifact's dominant format (4-bit
//!   affine, group 64): the serving path today.
//! - **B**: the same operation with its first N output channels on the
//!   Neural Engine and the rest on the GPU in the same 4-bit format, both
//!   started at the same instant.
//!
//! The Neural Engine slice holds the *dequantized* 4-bit weights as fp16.
//! Every 4-bit affine value is exactly representable in fp16 here, so B
//! computes with the same weights as A; the cost is memory, four times the
//! 4-bit bytes for the Neural Engine's share. Its contraction is split into
//! 2,048-deep parts, which the contraction-split journeys measured at the
//! Neural Engine's full rate.
//!
//! Not included in B: merging the two slices into one tensor (rung 4).

use std::time::Duration;

use astronomical_neural_engine::deterministic_values::{
    PROJECTION_WEIGHT_STANDARD_DEVIATION, bell_shaped_values,
};
use astronomical_neural_engine::half_precision::{f16_bits_slice_to_f32, f32_slice_to_f16_bits};
use astronomical_neural_engine::projection_shape::ProjectionShape;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

use super::concurrent_split_journeys::{
    MeasuredEngines, assert_slice_matches_reference, load_neural_engine_slice, milliseconds,
    time_split,
};
use super::mlx_projection_journeys::{
    MlxProjectionForm, PRODUCTION_QUANTIZATION_GROUP_SIZE, build_projection, initialize_runtime,
    time_projection, token_major, upload,
};
use crate::hello::projection_fixture::ProjectionFixture;

const ORNITH_QUANTIZATION_BITS: i32 = 4;
const PREFILL_CHUNK_TOKEN_COUNT: usize = 2048;
const NEURAL_ENGINE_CONTRACTION_PART_WIDTH: usize = 2048;
const ORNITH_WEIGHT_SEED: u64 = 0x5eed_0909;
// Neural Engine shares in sixteenths of the output channels. With the
// contraction split the Neural Engine runs near 18.6 TFLOP/s against the
// GPU's 4-bit rate, which puts the balance near one half; the sweep brackets it.
const NEURAL_ENGINE_SHARE_SIXTEENTHS: [usize; 4] = [6, 7, 8, 9];

struct DenseOperation {
    label: &'static str,
    input_channels: usize,
    output_channels: usize,
    layer_count: usize,
}

struct QuantizedOperation {
    fixture: ProjectionFixture,
    packed_weights: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    token_major_activations: MlxArray,
}

fn quantize_and_round_trip(
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

fn prepare_quantized_operation(
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
fn trailing_rows(runtime: &MlxRuntime, quantized_tensor: &MlxArray, first_row: usize) -> MlxArray {
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

fn quantized_form<'arrays>(
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

/// MLX's `[1, tokens, channels]` result as `[channels][tokens]`.
fn channel_major_projection(
    runtime: &MlxRuntime,
    gpu_form: &MlxProjectionForm<'_>,
    channel_count: usize,
) -> Vec<f32> {
    let projected = build_projection(runtime, gpu_form);
    let single = runtime
        .astype(&projected, MlxDtype::Float32)
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

/// One B configuration; returns the median time of the two engines together.
fn measure_b(
    runtime: &MlxRuntime,
    operation: &QuantizedOperation,
    neural_engine_channels: usize,
    label: &str,
) -> Duration {
    let output_channels = operation.fixture.shape.output_channels();
    let gpu_channels = output_channels - neural_engine_channels;
    let neural_engine_fixture = operation
        .fixture
        .output_channel_slice(0, neural_engine_channels);
    let gpu_fixture = operation
        .fixture
        .output_channel_slice(neural_engine_channels, gpu_channels);
    let contraction_part_count =
        operation.fixture.shape.input_channels() / NEURAL_ENGINE_CONTRACTION_PART_WIDTH;
    let model_directory = tempfile::tempdir().expect("a scratch directory for the model");
    let mut neural_engine_projection = load_neural_engine_slice(
        &neural_engine_fixture,
        contraction_part_count,
        model_directory.path(),
    );
    let gpu_packed = trailing_rows(runtime, &operation.packed_weights, neural_engine_channels);
    let gpu_scales = trailing_rows(runtime, &operation.scales, neural_engine_channels);
    let gpu_biases = trailing_rows(runtime, &operation.biases, neural_engine_channels);
    let gpu_form = quantized_form(
        &operation.token_major_activations,
        &gpu_packed,
        &gpu_scales,
        &gpu_biases,
    );
    let [together, _, _] = time_split(
        runtime,
        &gpu_form,
        &mut neural_engine_projection,
        &MeasuredEngines::Together,
        label,
    );
    let neural_engine_projected =
        f16_bits_slice_to_f32(&neural_engine_projection.read_projected().expect("readable"));
    assert_slice_matches_reference(
        &neural_engine_fixture,
        &neural_engine_projected,
        "Neural Engine",
    );
    let gpu_projected = channel_major_projection(runtime, &gpu_form, gpu_channels);
    assert_slice_matches_reference(&gpu_fixture, &gpu_projected, "GPU 4-bit");
    together.median
}

fn run_ab(operation: &DenseOperation) {
    let runtime = initialize_runtime();
    let quantized = prepare_quantized_operation(&runtime, operation);
    let full_form = quantized_form(
        &quantized.token_major_activations,
        &quantized.packed_weights,
        &quantized.scales,
        &quantized.biases,
    );
    let a_label = format!("A {}: MLX GPU 4-bit alone", operation.label);
    let a_median = time_projection(&runtime, &full_form, &a_label).median;
    let full_gpu_projected =
        channel_major_projection(&runtime, &full_form, operation.output_channels);
    assert_slice_matches_reference(&quantized.fixture, &full_gpu_projected, "A GPU 4-bit");
    eprintln!(
        "[dense-ab] {}: A = {:.3} ms",
        operation.label,
        milliseconds(a_median)
    );
    let mut best_b = (Duration::MAX, 0_usize);
    for share_sixteenths in NEURAL_ENGINE_SHARE_SIXTEENTHS {
        let neural_engine_channels = operation.output_channels * share_sixteenths / 16;
        let b_label = format!(
            "B {} with {neural_engine_channels} NE channels",
            operation.label
        );
        let b_median = measure_b(&runtime, &quantized, neural_engine_channels, &b_label);
        eprintln!(
            "[dense-ab] {}: B ({neural_engine_channels} of {} channels on the NE) = {:.3} ms → {:.3}×",
            operation.label,
            operation.output_channels,
            milliseconds(b_median),
            a_median.as_secs_f64() / b_median.as_secs_f64()
        );
        if b_median < best_b.0 {
            best_b = (b_median, neural_engine_channels);
        }
    }
    print_ab_summary(operation, a_median, best_b);
}

fn print_ab_summary(operation: &DenseOperation, a_median: Duration, best_b: (Duration, usize)) {
    let (b_median, neural_engine_channels) = best_b;
    eprintln!(
        "[dense-ab] SUMMARY {} ×{} layers | A {:.3} ms | best B {:.3} ms ({neural_engine_channels} NE \
         channels, {:.0}%) | {:.3}× | per prefill chunk: A {:.1} ms → B {:.1} ms",
        operation.label,
        operation.layer_count,
        milliseconds(a_median),
        milliseconds(b_median),
        100.0 * neural_engine_channels as f64 / operation.output_channels as f64,
        a_median.as_secs_f64() / b_median.as_secs_f64(),
        milliseconds(a_median) * operation.layer_count as f64,
        milliseconds(b_median) * operation.layer_count as f64,
    );
}

#[test]
#[serial(neural_engine)]
#[ignore = "A/B of Ornith-1.5-9B's DeltaNet fused input on the real Neural Engine and GPU"]
fn should_compare_the_dense_deltanet_fused_input_alone_and_split() {
    run_ab(&DenseOperation {
        label: "DeltaNet fused input 4096→12352",
        input_channels: 4096,
        output_channels: 12_352,
        layer_count: 24,
    });
}

#[test]
#[serial(neural_engine)]
#[ignore = "A/B of Ornith-1.5-9B's attention fused input on the real Neural Engine and GPU"]
fn should_compare_the_dense_attention_fused_input_alone_and_split() {
    run_ab(&DenseOperation {
        label: "attention fused input 4096→10240",
        input_channels: 4096,
        output_channels: 10_240,
        layer_count: 8,
    });
}

#[test]
#[serial(neural_engine)]
#[ignore = "A/B of Ornith-1.5-9B's mixer output on the real Neural Engine and GPU"]
fn should_compare_the_dense_mixer_output_alone_and_split() {
    run_ab(&DenseOperation {
        label: "mixer output 4096→4096",
        input_channels: 4096,
        output_channels: 4096,
        layer_count: 32,
    });
}

#[test]
#[serial(neural_engine)]
#[ignore = "A/B of Ornith-1.5-9B's MLP gate and up projection on the real Neural Engine and GPU"]
fn should_compare_the_dense_mlp_gate_and_up_alone_and_split() {
    run_ab(&DenseOperation {
        label: "MLP gate+up 4096→24576",
        input_channels: 4096,
        output_channels: 24_576,
        layer_count: 32,
    });
}

#[test]
#[serial(neural_engine)]
#[ignore = "A/B of Ornith-1.5-9B's MLP down projection on the real Neural Engine and GPU"]
fn should_compare_the_dense_mlp_down_alone_and_split() {
    run_ab(&DenseOperation {
        label: "MLP down 12288→4096",
        input_channels: 12_288,
        output_channels: 4096,
        layer_count: 32,
    });
}

//! A/B for a dense model: Ornith-1.5-9B (`qwen3_5`, 32 layers: 24 DeltaNet
//! and 8 full attention, hidden 4,096, MLP intermediate 12,288).
//!
//! - **A**: MLX alone on the GPU, in the artifact's dominant format (4-bit
//!   affine, group 64): the serving path today.
//! - **B**: the same operation with its first N output channels on the
//!   Neural Engine and the rest on the GPU in the same 4-bit format, both
//!   started at the same instant.
//!
//! The Neural Engine slice holds the *dequantized* 4-bit weights as fp16,
//! rounded to within 0.3 percent of the weight RMS (the compressed-weight
//! journeys measure this), so B computes with nearly but not exactly A's
//! weights; the cost is memory, four times the 4-bit bytes for the Neural
//! Engine's share. Its contraction is split into
//! 2,048-deep parts, which the contraction-split journeys measured at the
//! Neural Engine's full rate.
//!
//! Not included in B: passing data between the engines and merging the two
//! slices; the hand-off journeys measure the whole call.

use std::time::Duration;

use astronomical_neural_engine::half_precision::f16_bits_slice_to_f32;
use astronomical_runtime_integration::MlxRuntime;
use serial_test::serial;

use crate::support::engine_split::{
    MeasuredEngines, assert_slice_matches_reference, load_neural_engine_slice, milliseconds,
    time_split,
};
use crate::support::mlx_projection::{
    MlxProjectionForm, build_projection, initialize_runtime, time_projection,
};
use crate::support::quantized_operation::{
    DenseOperation, QuantizedOperation, channel_major_values, prepare_quantized_operation,
    quantized_form, trailing_rows,
};

const NEURAL_ENGINE_CONTRACTION_PART_WIDTH: usize = 2048;
// Neural Engine shares in sixteenths of the output channels. With the
// contraction split the Neural Engine runs near 18.6 TFLOP/s against the
// GPU's 4-bit rate, which puts the balance near one half; the sweep brackets it.
const NEURAL_ENGINE_SHARE_SIXTEENTHS: [usize; 4] = [6, 7, 8, 9];
const DELTANET_OUTPUT_GATE_ROWS: usize = 4096;

/// MLX's `[1, tokens, channels]` result as `[channels][tokens]`.
fn channel_major_projection(
    runtime: &MlxRuntime,
    gpu_form: &MlxProjectionForm<'_>,
    channel_count: usize,
) -> Vec<f32> {
    let projected = build_projection(runtime, gpu_form);
    channel_major_values(runtime, &projected, channel_count)
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

/// The balance sweep every whole operation uses, around an even split.
fn run_ab(operation: &DenseOperation) {
    let mut neural_engine_channel_candidates = Vec::new();
    for share_sixteenths in NEURAL_ENGINE_SHARE_SIXTEENTHS {
        neural_engine_channel_candidates.push(operation.output_channels * share_sixteenths / 16);
    }
    run_ab_for_channels(operation, &neural_engine_channel_candidates);
}

fn run_ab_for_channels(operation: &DenseOperation, neural_engine_channel_candidates: &[usize]) {
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
    for neural_engine_channels in neural_engine_channel_candidates.iter().copied() {
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

/// Only the output gate's rows, which never enter the recurrent state. The whole
/// fused input is 12,352 rows; `z` is the 4,096 of them at value width.
#[test]
#[serial(neural_engine)]
#[ignore = "A/B of Ornith-1.5-9B's DeltaNet output gate rows on the real Neural Engine and GPU"]
fn should_compare_the_dense_deltanet_output_gate_rows_alone_and_split() {
    run_ab_for_channels(
        &DenseOperation {
            label: "DeltaNet fused input 4096→12352, output gate rows only",
            input_channels: 4096,
            output_channels: 12_352,
            layer_count: 24,
        },
        &[DELTANET_OUTPUT_GATE_ROWS],
    );
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

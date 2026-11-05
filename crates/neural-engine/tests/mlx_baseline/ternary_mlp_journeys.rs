//! Can one real MLP layer of the folded ternary 27B model run on the Neural
//! Engine (issue #161)? `ternary_mlp_memory_journeys` then asks whether many
//! of them fit.
//!
//! Earlier work ruled these models out on memory, assuming their 2-bit weights
//! had to expand to fp16 (about 57 GB across 64 layers). Their values are so few
//! that 8-bit tables shared by 16 rows hold them exactly, and that form used far
//! less memory at the same speed on an Ornith-1.5-9B projection. These journeys
//! test the idea on real weights.
//!
//! The gates print a verdict rather than fail the test, because a STOP is a
//! finding about the hardware that the run exists to report, not a broken build.
//!
//! Activations are seeded unit-variance values, because a layer-level measure
//! needs no model and post-norm activations are unit-variance by construction.
//! They are not Shakespeare, so accuracy here is a filter and not a final answer:
//! real prompts still have to be compared end to end across all 64 layers.
//!
//! The GPU chain composes the rotation from MLX operations rather than
//! production's one-launch kernel. Both produce the same bits, but the composed
//! form is slower, so the gain printed here can be somewhat flattering.

use std::time::{Duration, Instant};

use astronomical_neural_engine::core_ml_projection::{
    ComputePlacement, CoreMlComputeUnits, CoreMlProjection,
};
use astronomical_neural_engine::deterministic_values::bell_shaped_values;
use astronomical_neural_engine::half_precision::{f16_bits_slice_to_f32, f32_to_f16_bits};
use astronomical_neural_engine::ml_program_mlp_document::StageWeightEncoding;
use astronomical_neural_engine::ml_program_rotated_mlp_document::{
    RotatedMlpLayout, RotatedMlpWeights,
};
use astronomical_neural_engine::mlx_hand_off::{
    OutputHandOff, SplitAxis, channel_major_fp16, merge_surface_with_gpu_slice,
    write_prepared_input,
};
use astronomical_neural_engine::numerical_comparison::{NumericalComparison, compare_values};
use astronomical_neural_engine::process_memory::{MemoryGrowth, MemorySnapshot, mebibytes};
use astronomical_neural_engine::projection_shape::ProjectionShape;
use astronomical_neural_engine::timing_summary::TimingSummary;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxCompiledSwiGlu, MlxDtype};
use serial_test::serial;

use super::concurrent_split_journeys::milliseconds;
use super::dense_model_ab_journeys::channel_major_values;
use super::hand_off_stage_timing::{StageDurations, summarize_stages};
use super::mlx_projection_journeys::{initialize_runtime, upload};
use super::ternary_checkpoint::{
    BLOCK_INPUT_COLUMNS, TernaryCheckpoint, TernaryMlpLayer, TernaryProjection,
};
use crate::hello::measured_projection::{
    MEASURED_CALL_COUNT, MINIMUM_WARM_UP_CALL_COUNT, MINIMUM_WARM_UP_DURATION,
};

// A middle layer, so neither the first nor the last layer's scale spread decides.
pub(super) const PROBED_LAYER_INDEX: usize = 31;
pub(super) const HIDDEN_CHANNELS: usize = 5120;
pub(super) const INTERMEDIATE_CHANNELS: usize = 17_408;
// The deepest width dividing both 5,120 and 17,408.
pub(super) const CONTRACTION_PART_WIDTH: usize = 1024;
pub(super) const ROTATION_BLOCK_SIZE: usize = 1024;
pub(super) const ROWS_PER_LOOKUP_TABLE: usize = 16;
pub(super) const CHUNK_TOKEN_COUNT: usize = 2048;
// The share serving uses on the 9B, and a multiple of 64 so MLX can read the
// surface directly. The journey prints enough to choose a better balance.
pub(super) const NEURAL_ENGINE_TOKEN_COUNT: usize = 960;
pub(super) const LAYER_COUNT: f64 = 64.0;
const QUANTIZATION_BITS: i32 = 2;
pub(super) const ACTIVATION_SEED: u64 = 0x5eed_2700;
pub(super) const NEURAL_ENGINE_FINISH_TIMEOUT: Duration = Duration::from_secs(30);
const STEP_COUNT: usize = 6;

const MINIMUM_NEURAL_ENGINE_TFLOPS: f64 = 13.0;
const MINIMUM_COSINE_SIMILARITY: f64 = 0.9999;

/// Reports a numbered step with elapsed time, since the journey loads real
/// weights and compiles a large program between its measurements.
struct JourneyProgress {
    started_at: Instant,
    completed_steps: usize,
}

impl JourneyProgress {
    fn finished(&mut self, step_label: &str) {
        self.completed_steps += 1;
        let elapsed_seconds = self.started_at.elapsed().as_secs_f64();
        let remaining_seconds = elapsed_seconds / self.completed_steps as f64
            * (STEP_COUNT - self.completed_steps) as f64;
        eprintln!(
            "[ternary-mlp] step {}/{STEP_COUNT} ({:.0}%) {step_label}: elapsed {elapsed_seconds:.1} s, ETA {remaining_seconds:.0} s",
            self.completed_steps,
            100.0 * self.completed_steps as f64 / STEP_COUNT as f64,
        );
    }
}

/// The GPU's copy of one packed projection.
pub(super) struct PackedArrays {
    pub(super) packed_codes: MlxArray,
    pub(super) scales: MlxArray,
    pub(super) biases: MlxArray,
}

pub(super) struct GpuMlp {
    gate_and_up: PackedArrays,
    pub(super) down: PackedArrays,
    pub(super) hidden_signs: MlxArray,
    pub(super) intermediate_signs: MlxArray,
    pub(super) compiled_swiglu: MlxCompiledSwiGlu,
}

pub(super) fn fp16_array(runtime: &MlxRuntime, fp16_values: &[u16], shape: &[i32]) -> MlxArray {
    upload(
        runtime,
        &f16_bits_slice_to_f32(fp16_values),
        shape,
        MlxDtype::Float16,
    )
}

fn packed_arrays(runtime: &MlxRuntime, members: &[&TernaryProjection]) -> PackedArrays {
    let mut packed_codes = Vec::new();
    let mut fp16_scales = Vec::new();
    let mut fp16_biases = Vec::new();
    let mut row_count = 0;
    for member in members {
        packed_codes.extend_from_slice(&member.packed_codes);
        fp16_scales.extend_from_slice(&member.fp16_scales);
        fp16_biases.extend_from_slice(&member.fp16_biases);
        row_count += member.row_count;
    }
    let first_member = members[0];
    let block_shape = [row_count as i32, first_member.blocks_per_row() as i32];
    let packed_shape = [row_count as i32, first_member.packed_words_per_row() as i32];
    let packed_codes = runtime
        .array_from_u32(&packed_codes, &packed_shape)
        .expect("the codes upload");
    let scales = fp16_array(runtime, &fp16_scales, &block_shape);
    let biases = fp16_array(runtime, &fp16_biases, &block_shape);
    runtime
        .evaluate_arrays(&[&packed_codes])
        .expect("the codes evaluate");
    PackedArrays {
        packed_codes,
        scales,
        biases,
    }
}

pub(super) fn gpu_mlp(runtime: &MlxRuntime, layer: &TernaryMlpLayer) -> GpuMlp {
    let hidden_signs = runtime
        .array_from_f32(&layer.gate.signs, &[HIDDEN_CHANNELS as i32])
        .expect("the signs upload");
    let intermediate_signs = runtime
        .array_from_f32(&layer.down.signs, &[INTERMEDIATE_CHANNELS as i32])
        .expect("the signs upload");
    GpuMlp {
        gate_and_up: packed_arrays(runtime, &[&layer.gate, &layer.up]),
        down: packed_arrays(runtime, &[&layer.down]),
        hidden_signs,
        intermediate_signs,
        compiled_swiglu: MlxCompiledSwiGlu::new().expect("the activation compiles"),
    }
}

/// Signs in fp32, the normalized block Hadamard, and the cast back, mirroring
/// the reference's arithmetic (`fwht.rs` in model serving).
pub(super) fn rotate(runtime: &MlxRuntime, activations: &MlxArray, signs: &MlxArray) -> MlxArray {
    let activation_shape = activations.shape();
    let width = activation_shape[activation_shape.len() - 1];
    let token_count = activation_shape[1];
    let block_size = ROTATION_BLOCK_SIZE as i32;
    let single_precision = runtime
        .astype(activations, MlxDtype::Float32)
        .expect("the cast builds");
    let signed = runtime
        .multiply(&single_precision, signs)
        .expect("the signs apply");
    let blocked = runtime
        .reshape(&signed, &[token_count * width / block_size, block_size])
        .expect("the blocks reshape");
    let transformed = runtime
        .hadamard_transform(&blocked, Some(1.0 / (ROTATION_BLOCK_SIZE as f32).sqrt()))
        .expect("the rotation builds");
    let restored = runtime
        .reshape(&transformed, &activation_shape)
        .expect("the blocks restore");
    runtime
        .astype(&restored, MlxDtype::Float16)
        .expect("the cast back builds")
}

pub(super) fn packed_projection(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    packed: &PackedArrays,
) -> MlxArray {
    runtime
        .quantized_matmul_affine(
            activations,
            &packed.packed_codes,
            &packed.scales,
            &packed.biases,
            true,
            BLOCK_INPUT_COLUMNS as i32,
            QUANTIZATION_BITS,
        )
        .expect("the packed projection builds")
}

/// Everything after the first rotation, as one lazy graph.
fn gpu_chain_from_rotated(
    runtime: &MlxRuntime,
    mlp: &GpuMlp,
    rotated_input: &MlxArray,
) -> MlxArray {
    let token_count = rotated_input.shape()[1];
    let intermediate = INTERMEDIATE_CHANNELS as i32;
    let gate_and_up = packed_projection(runtime, rotated_input, &mlp.gate_and_up);
    let gate = runtime
        .slice(
            &gate_and_up,
            &[0, 0, 0],
            &[1, token_count, intermediate],
            &[1, 1, 1],
        )
        .expect("the gate slices");
    let up = runtime
        .slice(
            &gate_and_up,
            &[0, 0, intermediate],
            &[1, token_count, 2 * intermediate],
            &[1, 1, 1],
        )
        .expect("the up projection slices");
    let gated = runtime
        .apply_compiled_swiglu(&mlp.compiled_swiglu, &gate, &up)
        .expect("the activation applies");
    let rotated_gated = rotate(runtime, &gated, &mlp.intermediate_signs);
    packed_projection(runtime, &rotated_gated, &mlp.down)
}

fn token_range(
    runtime: &MlxRuntime,
    token_major_values: &MlxArray,
    first_token: usize,
    end_token: usize,
    channel_count: usize,
) -> MlxArray {
    runtime
        .slice(
            token_major_values,
            &[0, first_token as i32, 0],
            &[1, end_token as i32, channel_count as i32],
            &[1, 1, 1],
        )
        .expect("the tokens slice")
}

pub(super) fn time_gpu_alone(
    runtime: &MlxRuntime,
    mlp: &GpuMlp,
    input: &MlxArray,
) -> TimingSummary {
    let warm_up_started_at = Instant::now();
    while warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION {
        run_gpu_alone(runtime, mlp, input);
    }
    let mut call_durations = Vec::with_capacity(MEASURED_CALL_COUNT);
    for _ in 0..MEASURED_CALL_COUNT {
        let call_started_at = Instant::now();
        run_gpu_alone(runtime, mlp, input);
        call_durations.push(call_started_at.elapsed());
    }
    TimingSummary::from_samples(&call_durations).expect("calls were timed")
}

pub(super) fn run_gpu_alone(runtime: &MlxRuntime, mlp: &GpuMlp, input: &MlxArray) -> MlxArray {
    let rotated_input = rotate(runtime, input, &mlp.hidden_signs);
    let projected = gpu_chain_from_rotated(runtime, mlp, &rotated_input);
    runtime
        .evaluate_arrays(&[&projected])
        .expect("the GPU chain evaluates");
    projected
}

pub(super) fn load_program(
    layer: &TernaryMlpLayer,
    weight_encoding: StageWeightEncoding,
    directory: &std::path::Path,
) -> (CoreMlProjection, usize) {
    let mut gate_and_up_weights = layer.gate.fp16_weights();
    gate_and_up_weights.extend(layer.up.fp16_weights());
    let down_weights = layer.down.fp16_weights();
    let mut fp16_signs = Vec::with_capacity(layer.down.signs.len());
    for sign in &layer.down.signs {
        fp16_signs.push(f32_to_f16_bits(*sign));
    }
    let shape = ProjectionShape::new(HIDDEN_CHANNELS, HIDDEN_CHANNELS, NEURAL_ENGINE_TOKEN_COUNT)
        .expect("valid shape");
    let weight_count = gate_and_up_weights.len() + down_weights.len();
    let projection = CoreMlProjection::load_ml_program_rotated_mlp(
        RotatedMlpLayout {
            shape,
            intermediate_channels: INTERMEDIATE_CHANNELS,
            contraction_part_width: CONTRACTION_PART_WIDTH,
            rotation_block_size: ROTATION_BLOCK_SIZE,
            weight_encoding,
        },
        RotatedMlpWeights {
            gate_and_up: &gate_and_up_weights,
            down: &down_weights,
            intermediate_signs: &fp16_signs,
        },
        CoreMlComputeUnits::CpuAndNeuralEngine,
        directory,
    )
    .expect("the rotated MLP program loads");
    (projection, weight_count)
}

fn write_rotated_tokens(
    runtime: &MlxRuntime,
    mlp: &GpuMlp,
    input: &MlxArray,
    projection: &mut CoreMlProjection,
) {
    let rotated_input = rotate(runtime, input, &mlp.hidden_signs);
    let neural_engine_tokens = token_range(
        runtime,
        &rotated_input,
        0,
        NEURAL_ENGINE_TOKEN_COUNT,
        HIDDEN_CHANNELS,
    );
    let prepared = channel_major_fp16(runtime, &neural_engine_tokens).expect("the reorder builds");
    runtime
        .evaluate_arrays(&[&prepared])
        .expect("the activations prepare");
    write_prepared_input(&prepared, projection).expect("the activations fit");
}

fn time_neural_engine_alone(projection: &mut CoreMlProjection) -> TimingSummary {
    let warm_up_started_at = Instant::now();
    let mut warm_up_calls = 0;
    while warm_up_calls < MINIMUM_WARM_UP_CALL_COUNT
        || warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION
    {
        projection.predict().expect("the warm-up prediction runs");
        warm_up_calls += 1;
    }
    let mut prediction_durations = Vec::with_capacity(MEASURED_CALL_COUNT);
    for _ in 0..MEASURED_CALL_COUNT {
        prediction_durations.push(projection.predict().expect("the prediction runs"));
    }
    TimingSummary::from_samples(&prediction_durations).expect("predictions were timed")
}

fn run_token_split_call(
    runtime: &MlxRuntime,
    mlp: &GpuMlp,
    input: &MlxArray,
    projection: &mut CoreMlProjection,
) -> (StageDurations, MlxArray) {
    let call_started_at = Instant::now();
    let rotated_input = rotate(runtime, input, &mlp.hidden_signs);
    let neural_engine_tokens = token_range(
        runtime,
        &rotated_input,
        0,
        NEURAL_ENGINE_TOKEN_COUNT,
        HIDDEN_CHANNELS,
    );
    let prepared = channel_major_fp16(runtime, &neural_engine_tokens).expect("the reorder builds");
    runtime
        .evaluate_arrays(&[&prepared])
        .expect("the activations prepare");
    let input_preparation = call_started_at.elapsed();
    let gpu_tokens = token_range(
        runtime,
        &rotated_input,
        NEURAL_ENGINE_TOKEN_COUNT,
        CHUNK_TOKEN_COUNT,
        HIDDEN_CHANNELS,
    );
    let gpu_slice = gpu_chain_from_rotated(runtime, mlp, &gpu_tokens);
    runtime
        .async_eval_arrays(&[&gpu_slice])
        .expect("the GPU slice starts");
    let write_started_at = Instant::now();
    write_prepared_input(&prepared, projection).expect("the activations fit");
    let surface_write = write_started_at.elapsed();
    let engines_started_at = Instant::now();
    let pending = projection.begin_prediction().expect("the program starts");
    runtime
        .evaluate_arrays(&[&gpu_slice])
        .expect("the GPU slice evaluates");
    pending
        .finish(NEURAL_ENGINE_FINISH_TIMEOUT)
        .expect("the program finishes");
    let engines_after_write = engines_started_at.elapsed();
    let merge_started_at = Instant::now();
    let merged = merge_surface_with_gpu_slice(
        runtime,
        projection,
        &gpu_slice,
        SplitAxis::Tokens,
        OutputHandOff::ZeroCopySurface,
    )
    .expect("the token slices merge");
    let stages = StageDurations {
        input_preparation,
        surface_write,
        engines_after_write,
        merge: merge_started_at.elapsed(),
        total: call_started_at.elapsed(),
    };
    (stages, merged)
}

/// Compares only the tokens the Neural Engine produced, because the GPU's own
/// tokens match by construction and would flatter the result.
fn compare_neural_engine_tokens(
    runtime: &MlxRuntime,
    gpu_alone: &MlxArray,
    merged: &MlxArray,
) -> NumericalComparison {
    let reference_tokens = token_range(
        runtime,
        gpu_alone,
        0,
        NEURAL_ENGINE_TOKEN_COUNT,
        HIDDEN_CHANNELS,
    );
    let candidate_tokens = token_range(
        runtime,
        merged,
        0,
        NEURAL_ENGINE_TOKEN_COUNT,
        HIDDEN_CHANNELS,
    );
    let reference_values = channel_major_values(runtime, &reference_tokens, HIDDEN_CHANNELS);
    let candidate_values = channel_major_values(runtime, &candidate_tokens, HIDDEN_CHANNELS);
    compare_values(&reference_values, &candidate_values).expect("same length")
}

pub(super) fn gate_verdict(has_passed: bool) -> &'static str {
    if has_passed { "PASS" } else { "STOP" }
}

fn print_stop_gates(
    projection: &CoreMlProjection,
    neural_engine_tflops: f64,
    comparison: &NumericalComparison,
) {
    let placement = projection.placement();
    let is_placed = placement.preferred == ComputePlacement::NeuralEngine
        && placement.operations_off_neural_engine == 0;
    eprintln!(
        "[ternary-mlp] GATE placement {}: preferred {:?}, {} operations off the Neural Engine",
        gate_verdict(is_placed),
        placement.preferred,
        placement.operations_off_neural_engine,
    );
    eprintln!(
        "[ternary-mlp] GATE speed {}: {neural_engine_tflops:.1} TFLOP/s (needs {MINIMUM_NEURAL_ENGINE_TFLOPS})",
        gate_verdict(neural_engine_tflops >= MINIMUM_NEURAL_ENGINE_TFLOPS),
    );
    eprintln!(
        "[ternary-mlp] GATE accuracy {}: cosine {:.7} against the GPU (needs {MINIMUM_COSINE_SIMILARITY}), max error {:.2e} of RMS, non-finite {}",
        gate_verdict(comparison.cosine_similarity >= MINIMUM_COSINE_SIMILARITY),
        comparison.cosine_similarity,
        comparison.maximum_error_relative_to_reference_rms,
        comparison.non_finite_candidate_count,
    );
}

fn measure_token_split(
    runtime: &MlxRuntime,
    mlp: &GpuMlp,
    input: &MlxArray,
    projection: &mut CoreMlProjection,
) -> (TimingSummary, MlxArray) {
    let warm_up_started_at = Instant::now();
    while warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION {
        run_token_split_call(runtime, mlp, input, projection);
    }
    let mut stage_samples = Vec::with_capacity(MEASURED_CALL_COUNT);
    for _ in 0..MEASURED_CALL_COUNT {
        stage_samples.push(run_token_split_call(runtime, mlp, input, projection).0);
    }
    let summaries = summarize_stages(&stage_samples);
    eprintln!(
        "[ternary-mlp] split stages: prepare {:.1} ms, write {:.1} ms, engines {:.1} ms, merge {:.1} ms",
        milliseconds(summaries.input_preparation.median),
        milliseconds(summaries.surface_write.median),
        milliseconds(summaries.engines_after_write.median),
        milliseconds(summaries.merge.median),
    );
    let (_, merged) = run_token_split_call(runtime, mlp, input, projection);
    (summaries.total, merged)
}

fn load_and_report_program(
    layer: &TernaryMlpLayer,
    weight_encoding: StageWeightEncoding,
    directory: &std::path::Path,
) -> (CoreMlProjection, usize) {
    let before_load = MemorySnapshot::capture();
    let load_started_at = Instant::now();
    let (mut projection, weight_count) = load_program(layer, weight_encoding, directory);
    projection.predict().expect("the first prediction runs");
    let growth = MemoryGrowth::between(before_load, MemorySnapshot::capture());
    eprintln!(
        "[ternary-mlp] program loaded in {:.1} s ({:.1} million weights), footprint {:+.0} MiB",
        load_started_at.elapsed().as_secs_f64(),
        weight_count as f64 / 1e6,
        mebibytes(growth.process_physical_footprint_bytes),
    );
    (projection, weight_count)
}

/// Returns the program's rate, counting the rotation's weights as work because
/// it runs inside the program.
fn measure_neural_engine_tflops(projection: &mut CoreMlProjection, weight_count: usize) -> f64 {
    let neural_engine_timing = time_neural_engine_alone(projection);
    let rotation_weights = (INTERMEDIATE_CHANNELS * ROTATION_BLOCK_SIZE) as f64;
    let floating_point_operations =
        2.0 * (weight_count as f64 + rotation_weights) * NEURAL_ENGINE_TOKEN_COUNT as f64;
    let neural_engine_tflops =
        floating_point_operations / neural_engine_timing.median.as_secs_f64() / 1e12;
    eprintln!(
        "[ternary-mlp] Neural Engine alone, {NEURAL_ENGINE_TOKEN_COUNT} tokens: median {:.1} ms (p90 {:.1}), {neural_engine_tflops:.1} TFLOP/s",
        milliseconds(neural_engine_timing.median),
        milliseconds(neural_engine_timing.ninetieth_percentile),
    );
    neural_engine_tflops
}

fn measure_ternary_mlp_layer(weight_encoding: StageWeightEncoding) {
    let mut progress = JourneyProgress {
        started_at: Instant::now(),
        completed_steps: 0,
    };
    eprintln!("[ternary-mlp] layer {PROBED_LAYER_INDEX}, {weight_encoding:?}");
    let layer = TernaryCheckpoint::open().mlp_layer(PROBED_LAYER_INDEX);
    assert_eq!(
        layer.gate.signs, layer.up.signs,
        "gate and up share one rotated input"
    );
    let runtime = initialize_runtime();
    let mlp = gpu_mlp(&runtime, &layer);
    let activations = bell_shaped_values(ACTIVATION_SEED, HIDDEN_CHANNELS * CHUNK_TOKEN_COUNT, 1.0);
    let input = upload(
        &runtime,
        &activations,
        &[1, CHUNK_TOKEN_COUNT as i32, HIDDEN_CHANNELS as i32],
        MlxDtype::Float16,
    );
    progress.finished("real weights read");

    let gpu_alone_timing = time_gpu_alone(&runtime, &mlp, &input);
    let gpu_alone = run_gpu_alone(&runtime, &mlp, &input);
    eprintln!(
        "[ternary-mlp] GPU alone, {CHUNK_TOKEN_COUNT} tokens: median {:.1} ms (p90 {:.1}), {:.1} ms per chunk across {LAYER_COUNT} layers",
        milliseconds(gpu_alone_timing.median),
        milliseconds(gpu_alone_timing.ninetieth_percentile),
        milliseconds(gpu_alone_timing.median) * LAYER_COUNT,
    );
    progress.finished("GPU baseline");

    let directory = tempfile::tempdir().expect("a scratch directory for the model");
    let (mut projection, weight_count) =
        load_and_report_program(&layer, weight_encoding, directory.path());
    progress.finished("program compiled and loaded");

    write_rotated_tokens(&runtime, &mlp, &input, &mut projection);
    let neural_engine_tflops = measure_neural_engine_tflops(&mut projection, weight_count);
    progress.finished("Neural Engine alone");

    let (split_timing, merged) = measure_token_split(&runtime, &mlp, &input, &mut projection);
    eprintln!(
        "[ternary-mlp] both engines, {CHUNK_TOKEN_COUNT} tokens: median {:.1} ms against {:.1} ms alone, {:.3}× on this layer's MLP",
        milliseconds(split_timing.median),
        milliseconds(gpu_alone_timing.median),
        gpu_alone_timing.median.as_secs_f64() / split_timing.median.as_secs_f64(),
    );
    progress.finished("both engines");

    let comparison = compare_neural_engine_tokens(&runtime, &gpu_alone, &merged);
    print_stop_gates(&projection, neural_engine_tflops, &comparison);
    assert_eq!(comparison.non_finite_candidate_count, 0);
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads one real 27B MLP layer on the real Neural Engine as shared lookup tables"]
fn should_measure_one_real_ternary_mlp_layer_as_shared_lookup_tables() {
    measure_ternary_mlp_layer(StageWeightEncoding::SharedLookupTables {
        rows_per_table: ROWS_PER_LOOKUP_TABLE,
    });
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads one real 27B MLP layer on the real Neural Engine as fp16, for comparison"]
fn should_measure_one_real_ternary_mlp_layer_as_fp16() {
    measure_ternary_mlp_layer(StageWeightEncoding::Fp16);
}

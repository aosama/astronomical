//! Does running a whole dense MLP on the Neural Engine for a slice of tokens
//! beat splitting each projection's output channels (issue #161)?
//!
//! The channel split pays a hand-off per projection. This journey sends the
//! first N tokens through `down(silu(gate) · up)` as one Core ML program while
//! the GPU runs the remaining tokens through the same chain in the artifact's
//! 4-bit form, so one MLP needs one hand-off. The price is memory: the Neural
//! Engine holds every weight of the MLP as fp16, not only a share of them.
//!
//! Weights are synthetic, shaped like Ornith-1.5-9B's (hidden 4,096,
//! intermediate 12,288). The GPU baseline evaluates the whole chain as one graph
//! through production's compiled SwiGLU.

use std::time::{Duration, Instant};

use astronomical_neural_engine::accelerate_reference::reference_projection;
use astronomical_neural_engine::core_ml::{ComputePlacement, CoreMlComputeUnits, CoreMlProjection};
use astronomical_neural_engine::half_precision::f16_bits_slice_to_f32;
use astronomical_neural_engine::ml_program_mlp_document::MlpFp16Weights;
use astronomical_neural_engine::mlx_hand_off::{
    OutputHandOff, SplitAxis, channel_major_fp16, merge_surface_with_gpu_slice,
    write_prepared_input,
};
use astronomical_neural_engine::numerical_comparison::compare_values;
use astronomical_neural_engine::process_memory::{
    MemoryGrowth, MemorySnapshot, mebibytes, physical_memory_bytes,
};
use astronomical_neural_engine::projection_shape::ProjectionShape;
use astronomical_neural_engine::timing_summary::TimingSummary;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxCompiledSwiGlu, MlxDtype};
use serial_test::serial;

use crate::support::engine_split::milliseconds;
use crate::support::hand_off_stage_timing::{StageDurations, summarize_stages};
use crate::support::journey_constants::{NEURAL_ENGINE_FINISH_TIMEOUT, PREFILL_CHUNK_TOKEN_COUNT};
use crate::support::measured_projection::{
    MEASURED_CALL_COUNT, MINIMUM_WARM_UP_CALL_COUNT, MINIMUM_WARM_UP_DURATION, report_progress,
};
use crate::support::mlx_projection::{
    build_projection, initialize_runtime, token_major, token_range, upload,
};
use crate::support::projection_fixture::ProjectionFixture;
use crate::support::quantized_operation::{
    channel_major_values, quantize_and_round_trip, quantized_form,
};

const HIDDEN_CHANNELS: usize = 4096;
const INTERMEDIATE_CHANNELS: usize = 12_288;
const CONTRACTION_PART_WIDTH: usize = 2048;
const MLP_LAYER_COUNT: usize = 32;
const MINIMUM_MERGED_COSINE: f64 = 0.999;
// A surface row must fill its alignment exactly for MLX to read it as one
// array, and multiples of 64 fp16 tokens do. The chain's balance should sit
// near the 37 to 44 percent the per-projection splits found.
const NEURAL_ENGINE_TOKEN_COUNTS: [usize; 7] = [640, 768, 832, 896, 960, 1024, 1088];
// The share serving uses for a 2,048-token chunk, so these programs match it.
const LAYER_PROGRAM_TOKEN_COUNT: usize = 960;
const WARM_UP_LAYER_CYCLES: usize = 2;
const MEASURED_LAYER_CYCLES: usize = 4;

struct QuantizedProjection {
    packed_weights: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

struct MlpFixture {
    gate_and_up: QuantizedProjection,
    down: QuantizedProjection,
    gate_and_up_fp16_weights: Vec<u16>,
    down_fp16_weights: Vec<u16>,
    token_major_bf16: MlxArray,
    reference_channel_major: Vec<f32>,
}

/// Quantizes synthetic weights, then stores the affine parameters as bf16, the
/// dtype real artifacts declare (see `Bf16Operation` in the hand-off journeys).
fn quantized_projection(
    runtime: &MlxRuntime,
    shape: ProjectionShape,
) -> (QuantizedProjection, Vec<u16>) {
    let (packed_weights, fp16_scales, fp16_biases, fp16_weights) =
        quantize_and_round_trip(runtime, shape);
    let scales = runtime
        .astype(&fp16_scales, MlxDtype::BFloat16)
        .expect("scales convert");
    let biases = runtime
        .astype(&fp16_biases, MlxDtype::BFloat16)
        .expect("biases convert");
    runtime
        .evaluate_arrays(&[&scales, &biases])
        .expect("parameters evaluate");
    (
        QuantizedProjection {
            packed_weights,
            scales,
            biases,
        },
        fp16_weights,
    )
}

/// `silu(gate) · up` over `[2 · intermediate][tokens]` fp32 values.
fn gated_reference(gate_and_up_channel_major: &[f32], token_count: usize) -> Vec<f32> {
    let (gate_values, up_values) =
        gate_and_up_channel_major.split_at(INTERMEDIATE_CHANNELS * token_count);
    let mut gated_values = Vec::with_capacity(gate_values.len());
    for (gate_value, up_value) in gate_values.iter().zip(up_values) {
        let sigmoid_value = 1.0 / (1.0 + (-gate_value).exp());
        gated_values.push(gate_value * sigmoid_value * up_value);
    }
    gated_values
}

fn mlp_fixture(runtime: &MlxRuntime) -> MlpFixture {
    let gate_and_up_shape = ProjectionShape::new(
        HIDDEN_CHANNELS,
        2 * INTERMEDIATE_CHANNELS,
        PREFILL_CHUNK_TOKEN_COUNT,
    )
    .expect("valid shape");
    let down_shape = ProjectionShape::new(
        INTERMEDIATE_CHANNELS,
        HIDDEN_CHANNELS,
        PREFILL_CHUNK_TOKEN_COUNT,
    )
    .expect("valid shape");
    let (gate_and_up, gate_and_up_fp16_weights) = quantized_projection(runtime, gate_and_up_shape);
    let (down, down_fp16_weights) = quantized_projection(runtime, down_shape);
    let gate_and_up_fixture =
        ProjectionFixture::with_fp16_weights(gate_and_up_shape, gate_and_up_fp16_weights.clone());
    let gated_values = gated_reference(
        &gate_and_up_fixture.reference_projected,
        PREFILL_CHUNK_TOKEN_COUNT,
    );
    let reference_channel_major = reference_projection(
        down_shape,
        &f16_bits_slice_to_f32(&down_fp16_weights),
        &gated_values,
    )
    .expect("the reference runs");
    let token_major_bf16 = upload(
        runtime,
        &token_major(&gate_and_up_fixture),
        &[1, PREFILL_CHUNK_TOKEN_COUNT as i32, HIDDEN_CHANNELS as i32],
        MlxDtype::BFloat16,
    );
    MlpFixture {
        gate_and_up,
        down,
        gate_and_up_fp16_weights,
        down_fp16_weights,
        token_major_bf16,
        reference_channel_major,
    }
}

/// The whole chain on the GPU in the artifact's 4-bit form, as one lazy graph.
fn gpu_mlp(
    runtime: &MlxRuntime,
    compiled_swiglu: &MlxCompiledSwiGlu,
    fixture: &MlpFixture,
    token_major_activations: &MlxArray,
) -> MlxArray {
    let token_count = token_major_activations.shape()[1];
    let intermediate = INTERMEDIATE_CHANNELS as i32;
    let gate_and_up = build_projection(
        runtime,
        &quantized_form(
            token_major_activations,
            &fixture.gate_and_up.packed_weights,
            &fixture.gate_and_up.scales,
            &fixture.gate_and_up.biases,
        ),
    );
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
        .apply_compiled_swiglu(compiled_swiglu, &gate, &up)
        .expect("the activation applies");
    build_projection(
        runtime,
        &quantized_form(
            &gated,
            &fixture.down.packed_weights,
            &fixture.down.scales,
            &fixture.down.biases,
        ),
    )
}

fn time_gpu_baseline(
    runtime: &MlxRuntime,
    compiled_swiglu: &MlxCompiledSwiGlu,
    fixture: &MlpFixture,
) -> TimingSummary {
    let warm_up_started_at = Instant::now();
    while warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION {
        let projected = gpu_mlp(runtime, compiled_swiglu, fixture, &fixture.token_major_bf16);
        runtime
            .evaluate_arrays(&[&projected])
            .expect("warm-up evaluates");
    }
    let mut call_durations = Vec::with_capacity(MEASURED_CALL_COUNT);
    for _ in 0..MEASURED_CALL_COUNT {
        let call_started_at = Instant::now();
        let projected = gpu_mlp(runtime, compiled_swiglu, fixture, &fixture.token_major_bf16);
        runtime
            .evaluate_arrays(&[&projected])
            .expect("the baseline evaluates");
        call_durations.push(call_started_at.elapsed());
    }
    TimingSummary::from_samples(&call_durations).expect("calls were timed")
}

fn load_mlp_program(
    fixture: &MlpFixture,
    neural_engine_tokens: usize,
    directory: &std::path::Path,
) -> CoreMlProjection {
    let shape = ProjectionShape::new(HIDDEN_CHANNELS, HIDDEN_CHANNELS, neural_engine_tokens)
        .expect("valid shape");
    CoreMlProjection::load_ml_program_mlp(
        shape,
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &fixture.gate_and_up_fp16_weights,
            down: &fixture.down_fp16_weights,
        },
        CONTRACTION_PART_WIDTH,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        directory,
    )
    .expect("the MLP program loads")
}

fn run_token_split_call(
    runtime: &MlxRuntime,
    compiled_swiglu: &MlxCompiledSwiGlu,
    fixture: &MlpFixture,
    projection: &mut CoreMlProjection,
) -> (StageDurations, MlxArray) {
    let neural_engine_tokens = projection.token_count();
    let call_started_at = Instant::now();
    let neural_engine_input = token_range(
        runtime,
        &fixture.token_major_bf16,
        0,
        neural_engine_tokens,
        HIDDEN_CHANNELS,
    );
    let prepared = channel_major_fp16(runtime, &neural_engine_input).expect("the reorder builds");
    runtime
        .evaluate_arrays(&[&prepared])
        .expect("the activations prepare");
    let input_preparation = call_started_at.elapsed();
    let gpu_input = token_range(
        runtime,
        &fixture.token_major_bf16,
        neural_engine_tokens,
        PREFILL_CHUNK_TOKEN_COUNT,
        HIDDEN_CHANNELS,
    );
    let gpu_slice = gpu_mlp(runtime, compiled_swiglu, fixture, &gpu_input);
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

fn assert_merged_matches_reference(runtime: &MlxRuntime, fixture: &MlpFixture, merged: &MlxArray) {
    let merged_values = channel_major_values(runtime, merged, HIDDEN_CHANNELS);
    let comparison =
        compare_values(&fixture.reference_channel_major, &merged_values).expect("same length");
    eprintln!(
        "    merged vs fp32 reference: max abs {:.3e} ({:.2e} of RMS), cosine {:.8}, non-finite {}",
        comparison.maximum_absolute_error,
        comparison.maximum_error_relative_to_reference_rms,
        comparison.cosine_similarity,
        comparison.non_finite_candidate_count,
    );
    assert_eq!(comparison.non_finite_candidate_count, 0);
    assert!(comparison.cosine_similarity >= MINIMUM_MERGED_COSINE);
}

/// One token share; returns its median call time.
fn measure_token_share(
    runtime: &MlxRuntime,
    compiled_swiglu: &MlxCompiledSwiGlu,
    fixture: &MlpFixture,
    neural_engine_tokens: usize,
    baseline_median: Duration,
) -> Duration {
    let directory = tempfile::tempdir().expect("a scratch directory for the model");
    let mut projection = load_mlp_program(fixture, neural_engine_tokens, directory.path());
    let placement = projection.placement();
    assert_eq!(placement.preferred, ComputePlacement::NeuralEngine);
    let warm_up_started_at = Instant::now();
    let mut warm_up_call_count = 0;
    while warm_up_call_count < MINIMUM_WARM_UP_CALL_COUNT
        || warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION
    {
        run_token_split_call(runtime, compiled_swiglu, fixture, &mut projection);
        warm_up_call_count += 1;
    }
    let mut stage_samples = Vec::with_capacity(MEASURED_CALL_COUNT);
    let mut last_merged = MlxArray::from_bool(false);
    let measuring_started_at = Instant::now();
    let mut last_progress_at = measuring_started_at;
    for call_index in 0..MEASURED_CALL_COUNT {
        let (stages, merged) =
            run_token_split_call(runtime, compiled_swiglu, fixture, &mut projection);
        stage_samples.push(stages);
        last_merged = merged;
        if last_progress_at.elapsed() >= Duration::from_secs(5) {
            report_progress(
                "token split",
                call_index + 1,
                measuring_started_at.elapsed(),
            );
            last_progress_at = Instant::now();
        }
    }
    let summaries = summarize_stages(&stage_samples);
    eprintln!(
        "[token-split] {neural_engine_tokens} NE tokens ({:.0}%), {} ops off the NE | prepare {:.3} \
         + write {:.3} + engines {:.3} + merge {:.3} = {:.3} ms (p90 {:.3}) | baseline {:.3} ms → {:.3}×",
        100.0 * neural_engine_tokens as f64 / PREFILL_CHUNK_TOKEN_COUNT as f64,
        placement.operations_off_neural_engine,
        milliseconds(summaries.input_preparation.median),
        milliseconds(summaries.surface_write.median),
        milliseconds(summaries.engines_after_write.median),
        milliseconds(summaries.merge.median),
        milliseconds(summaries.total.median),
        milliseconds(summaries.total.ninetieth_percentile),
        milliseconds(baseline_median),
        baseline_median.as_secs_f64() / summaries.total.median.as_secs_f64(),
    );
    assert_merged_matches_reference(runtime, fixture, &last_merged);
    assert_eq!(placement.operations_off_neural_engine, 0);
    summaries.total.median
}

#[test]
#[serial(neural_engine)]
#[ignore = "runs Ornith-1.5-9B-shaped MLPs on the real Neural Engine and GPU, split by token"]
fn should_measure_the_whole_mlp_split_by_token() {
    let runtime = initialize_runtime();
    let compiled_swiglu = MlxCompiledSwiGlu::new().expect("the activation compiles");
    let fixture = mlp_fixture(&runtime);
    let baseline = time_gpu_baseline(&runtime, &compiled_swiglu, &fixture);
    eprintln!(
        "[token-split] GPU baseline, whole MLP in one graph: median {:.3} ms (p90 {:.3})",
        milliseconds(baseline.median),
        milliseconds(baseline.ninetieth_percentile),
    );
    let mut best_median = Duration::MAX;
    for neural_engine_tokens in NEURAL_ENGINE_TOKEN_COUNTS {
        let share_median = measure_token_share(
            &runtime,
            &compiled_swiglu,
            &fixture,
            neural_engine_tokens,
            baseline.median,
        );
        best_median = best_median.min(share_median);
    }
    eprintln!(
        "[token-split] SUMMARY per chunk ×{MLP_LAYER_COUNT} layers: {:.1} → {:.1} ms | {:.3}×",
        milliseconds(baseline.median) * MLP_LAYER_COUNT as f64,
        milliseconds(best_median) * MLP_LAYER_COUNT as f64,
        baseline.median.as_secs_f64() / best_median.as_secs_f64(),
    );
}

fn print_footprint(program_count: usize, growth: MemoryGrowth, load_duration: Duration) {
    eprintln!(
        "[token-split] {program_count} MLP program(s) loaded: footprint {:+.0} MiB, wired {:+.0} MiB, \
         this load {:.1} s",
        mebibytes(growth.process_physical_footprint_bytes),
        mebibytes(growth.system_wired_bytes),
        load_duration.as_secs_f64(),
    );
}

/// Half of this Mac's memory; loading past it would push other applications
/// out, and the evidence is in the per-program growth anyway.
fn memory_growth_budget_bytes() -> i64 {
    let installed_bytes = physical_memory_bytes();
    assert!(installed_bytes > 0, "the installed memory is readable");
    i64::try_from(installed_bytes / 2).expect("half of the installed memory fits")
}

fn print_settled_growth(program_count: usize, growth: MemoryGrowth) {
    eprintln!(
        "[token-split] {program_count} programs after cycling: footprint {:+.0} MiB, wired {:+.0} MiB",
        mebibytes(growth.process_physical_footprint_bytes),
        mebibytes(growth.system_wired_bytes),
    );
}

/// One real layer's weights, made distinct from every other layer's by rotating
/// the rows, so Core ML cannot share identical programs between layers.
fn layer_weights(weights: &[u16], row_width: usize, layer_index: usize) -> Vec<u16> {
    let mut rotated = weights.to_vec();
    rotated.rotate_left(row_width * layer_index);
    rotated
}

fn load_distinct_layer(
    gate_and_up_weights: &[u16],
    down_weights: &[u16],
    layer_index: usize,
    directory: &std::path::Path,
) -> CoreMlProjection {
    let shape = ProjectionShape::new(HIDDEN_CHANNELS, HIDDEN_CHANNELS, LAYER_PROGRAM_TOKEN_COUNT)
        .expect("valid shape");
    CoreMlProjection::load_ml_program_mlp(
        shape,
        INTERMEDIATE_CHANNELS,
        MlpFp16Weights {
            gate_and_up: &layer_weights(gate_and_up_weights, HIDDEN_CHANNELS, layer_index),
            down: &layer_weights(down_weights, INTERMEDIATE_CHANNELS, layer_index),
        },
        CONTRACTION_PART_WIDTH,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        directory,
    )
    .expect("the layer program loads")
}

/// Prefill visits every layer once per chunk, so each prediction follows a
/// different program. Timing one program in a loop could hide a switching cost
/// that this pattern pays on every call.
fn time_layer_cycles(layers: &mut [(CoreMlProjection, tempfile::TempDir)]) -> TimingSummary {
    // fp16 1.0 everywhere: the cycle times the engine, not the numbers, which
    // the speed journey checks against the reference.
    let activations = vec![0x3c00_u16; HIDDEN_CHANNELS * LAYER_PROGRAM_TOKEN_COUNT];
    for (projection, _directory) in layers.iter_mut() {
        projection
            .write_activations(&activations)
            .expect("the activations fit");
    }
    for _ in 0..WARM_UP_LAYER_CYCLES {
        for (projection, _directory) in layers.iter_mut() {
            projection.predict().expect("the warm-up prediction runs");
        }
    }
    let mut prediction_durations = Vec::with_capacity(MEASURED_LAYER_CYCLES * layers.len());
    for _ in 0..MEASURED_LAYER_CYCLES {
        for (projection, _directory) in layers.iter_mut() {
            prediction_durations.push(projection.predict().expect("the prediction runs"));
        }
    }
    TimingSummary::from_samples(&prediction_durations).expect("predictions were timed")
}

/// Run alone, because earlier loads in the process blur these numbers.
#[test]
#[serial(neural_engine)]
#[ignore = "loads one distinct 151-million-weight program per layer and cycles through them"]
fn should_measure_one_mlp_program_per_layer() {
    let runtime = initialize_runtime();
    let fixture = mlp_fixture(&runtime);
    let MlpFixture {
        gate_and_up_fp16_weights,
        down_fp16_weights,
        ..
    } = fixture;
    // The fixture's GPU arrays are gone before the first reading, so MLX freeing
    // them cannot appear as negative growth, as it did with a shared fixture.
    drop(runtime);
    let growth_budget_bytes = memory_growth_budget_bytes();
    let before_loads = MemorySnapshot::capture();
    let mut layers = Vec::with_capacity(MLP_LAYER_COUNT);
    for layer_index in 0..MLP_LAYER_COUNT {
        let directory = tempfile::tempdir().expect("a scratch directory for the model");
        let load_started_at = Instant::now();
        let mut projection = load_distinct_layer(
            &gate_and_up_fp16_weights,
            &down_fp16_weights,
            layer_index,
            directory.path(),
        );
        projection.predict().expect("the program runs");
        let growth = MemoryGrowth::between(before_loads, MemorySnapshot::capture());
        print_footprint(layer_index + 1, growth, load_started_at.elapsed());
        layers.push((projection, directory));
        if growth.process_physical_footprint_bytes + growth.system_wired_bytes > growth_budget_bytes
        {
            eprintln!("[token-split] stopped loading at half of this Mac's memory");
            break;
        }
    }
    let cycling = time_layer_cycles(&mut layers);
    eprintln!(
        "[token-split] cycling {} programs of {LAYER_PROGRAM_TOKEN_COUNT} tokens: median {:.2} ms \
         per prediction (p90 {:.2}, max {:.2}), {:.1} ms per chunk",
        layers.len(),
        milliseconds(cycling.median),
        milliseconds(cycling.ninetieth_percentile),
        milliseconds(cycling.maximum),
        milliseconds(cycling.median) * layers.len() as f64,
    );
    print_settled_growth(
        layers.len(),
        MemoryGrowth::between(before_loads, MemorySnapshot::capture()),
    );
}

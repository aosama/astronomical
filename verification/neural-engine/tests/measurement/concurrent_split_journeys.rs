//! Rung 3 of issue #161: do the Neural Engine and the GPU together beat the
//! GPU alone?
//!
//! The mixer output projection's 2,048 output channels are independent, so
//! they are split: the first N go to the Neural Engine through Core ML and
//! the remaining 2,048 − N to MLX on the GPU. Both start at the same instant
//! (the Neural Engine prediction runs on its own dispatch queue while this
//! thread drives MLX), and the combined call ends when the slower finishes.
//! Each engine is also timed alone at its slice, so any slowdown from
//! running side by side is visible, and both slices are checked against the
//! matching rows of the fp32 reference.
//!
//! Not yet included: merging the two slices into one tensor (rung 4).

use std::time::{Duration, Instant};

use astronomical_neural_engine::weights::half_precision::f16_bits_slice_to_f32;
use astronomical_neural_engine_research::timing_summary::TimingSummary;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

use crate::support::engine_split::{
    MeasuredEngines, assert_slice_matches_reference, evaluate_gpu_slice, load_neural_engine_slice,
    milliseconds, summarize, time_split,
};
use crate::support::journey_constants::PREFILL_CHUNK_TOKEN_COUNT;
use crate::support::measured_projection::{MEASURED_CALL_COUNT, MINIMUM_WARM_UP_DURATION};
use crate::support::mlx_projection::{
    MlxProjectionForm, build_projection, initialize_runtime, upload,
};
use crate::support::projection_fixture::ProjectionFixture;

// Neural Engine shares of the mixer output's 2,048 output channels. The
// alone-timings (3.60 ms Neural Engine, 1.53 ms MLX for all 2,048) put the
// balance point near 610 channels, so the sweep brackets it.
const MIXER_OUTPUT_NEURAL_ENGINE_SHARES: [usize; 6] = [256, 384, 512, 640, 768, 1024];
// Neural Engine shares of the DeltaNet fused input's 12,352 output channels.
// The shape sweep (5.58 ms Neural Engine, 4.27 ms MLX) put the balance point
// near 43 percent, about 5,350 channels.
const DELTANET_INPUT_NEURAL_ENGINE_SHARES: [usize; 5] = [3072, 4096, 4608, 5120, 5632];
const DELTANET_INPUT_CHANNELS: usize = 2048;
const DELTANET_FUSED_OUTPUT_CHANNELS: usize = 12_352;

struct SplitTimings {
    gpu_alone: TimingSummary,
    neural_engine_alone: TimingSummary,
    together_total: TimingSummary,
    together_gpu: TimingSummary,
    together_neural_engine: TimingSummary,
}

fn read_gpu_slice(runtime: &MlxRuntime, gpu_form: &MlxProjectionForm<'_>) -> Vec<f32> {
    let projected = build_projection(runtime, gpu_form);
    let single_precision = runtime
        .astype(&projected, MlxDtype::Float32)
        .expect("the GPU slice converts");
    runtime
        .array_to_vec_f32(&single_precision)
        .expect("the GPU slice reads back")
}

struct GpuSlice {
    weights: MlxArray,
    activations: MlxArray,
}

fn upload_gpu_slice(runtime: &MlxRuntime, gpu_fixture: &ProjectionFixture) -> GpuSlice {
    let shape = gpu_fixture.shape;
    GpuSlice {
        weights: upload(
            runtime,
            &f16_bits_slice_to_f32(&gpu_fixture.fp16_weights),
            &[
                shape.output_channels() as i32,
                shape.input_channels() as i32,
            ],
            MlxDtype::Float16,
        ),
        activations: upload(
            runtime,
            &f16_bits_slice_to_f32(&gpu_fixture.fp16_activations),
            &[shape.input_channels() as i32, shape.token_count() as i32],
            MlxDtype::Float16,
        ),
    }
}

fn measure_split(
    runtime: &MlxRuntime,
    full_fixture: &ProjectionFixture,
    neural_engine_channels: usize,
) -> SplitTimings {
    let output_channels = full_fixture.shape.output_channels();
    let neural_engine_fixture = full_fixture.output_channel_slice(0, neural_engine_channels);
    let gpu_fixture = full_fixture.output_channel_slice(
        neural_engine_channels,
        output_channels - neural_engine_channels,
    );
    let model_directory = tempfile::tempdir().expect("a scratch directory for the model");
    let mut neural_engine_projection =
        load_neural_engine_slice(&neural_engine_fixture, 1, model_directory.path());
    let gpu_slice = upload_gpu_slice(runtime, &gpu_fixture);
    let gpu_form = MlxProjectionForm::Dense {
        weights: &gpu_slice.weights,
        activations: &gpu_slice.activations,
    };
    let label = format!(
        "split {neural_engine_channels} NE / {} GPU",
        output_channels - neural_engine_channels
    );
    let [gpu_alone, _, _] = time_split(
        runtime,
        &gpu_form,
        &mut neural_engine_projection,
        &MeasuredEngines::GpuAlone,
        &label,
    );
    let [neural_engine_alone, _, _] = time_split(
        runtime,
        &gpu_form,
        &mut neural_engine_projection,
        &MeasuredEngines::NeuralEngineAlone,
        &label,
    );
    let [together_total, together_gpu, together_neural_engine] = time_split(
        runtime,
        &gpu_form,
        &mut neural_engine_projection,
        &MeasuredEngines::Together,
        &label,
    );
    let neural_engine_projected =
        f16_bits_slice_to_f32(&neural_engine_projection.read_projected().expect("readable"));
    assert_slice_matches_reference(
        &neural_engine_fixture,
        &neural_engine_projected,
        "Neural Engine",
    );
    assert_slice_matches_reference(&gpu_fixture, &read_gpu_slice(runtime, &gpu_form), "GPU");
    SplitTimings {
        gpu_alone,
        neural_engine_alone,
        together_total,
        together_gpu,
        together_neural_engine,
    }
}

fn print_split(
    neural_engine_channels: usize,
    timings: &SplitTimings,
    gpu_only_full: &TimingSummary,
) {
    let speedup = gpu_only_full.median.as_secs_f64() / timings.together_total.median.as_secs_f64();
    eprintln!(
        "[neural-engine-journey] split {neural_engine_channels} NE: together {:.3} ms (p90 {:.3}) \
         vs GPU-only {:.3} ms → {speedup:.3}×\n    alone: GPU {:.3} ms, NE {:.3} ms | side by side: \
         GPU {:.3} ms ({:+.0}%), NE {:.3} ms ({:+.0}%)",
        milliseconds(timings.together_total.median),
        milliseconds(timings.together_total.ninetieth_percentile),
        milliseconds(gpu_only_full.median),
        milliseconds(timings.gpu_alone.median),
        milliseconds(timings.neural_engine_alone.median),
        milliseconds(timings.together_gpu.median),
        percent_change(timings.gpu_alone.median, timings.together_gpu.median),
        milliseconds(timings.together_neural_engine.median),
        percent_change(
            timings.neural_engine_alone.median,
            timings.together_neural_engine.median
        ),
    );
}

fn percent_change(alone: Duration, side_by_side: Duration) -> f64 {
    100.0 * (side_by_side.as_secs_f64() / alone.as_secs_f64() - 1.0)
}

fn measure_gpu_only_full(runtime: &MlxRuntime, full_fixture: &ProjectionFixture) -> TimingSummary {
    let gpu_slice = upload_gpu_slice(runtime, full_fixture);
    let gpu_form = MlxProjectionForm::Dense {
        weights: &gpu_slice.weights,
        activations: &gpu_slice.activations,
    };
    let warm_up_started_at = Instant::now();
    while warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION {
        evaluate_gpu_slice(runtime, &gpu_form);
    }
    let mut call_durations = Vec::with_capacity(MEASURED_CALL_COUNT);
    for _ in 0..MEASURED_CALL_COUNT {
        let call_started_at = Instant::now();
        evaluate_gpu_slice(runtime, &gpu_form);
        call_durations.push(call_started_at.elapsed());
    }
    summarize(&call_durations)
}

#[test]
#[serial(neural_engine)]
#[ignore = "splits the 34-GFLOP projection across the real Neural Engine and GPU"]
fn should_measure_the_projection_split_across_the_neural_engine_and_the_gpu() {
    sweep_splits(
        &ProjectionFixture::mixer_output_projection(),
        &MIXER_OUTPUT_NEURAL_ENGINE_SHARES,
    );
}

#[test]
#[serial(neural_engine)]
#[ignore = "splits the 104-GFLOP DeltaNet fused input across the real Neural Engine and GPU"]
fn should_measure_the_deltanet_fused_input_split_across_the_neural_engine_and_the_gpu() {
    sweep_splits(
        &ProjectionFixture::with_shape(
            DELTANET_INPUT_CHANNELS,
            DELTANET_FUSED_OUTPUT_CHANNELS,
            PREFILL_CHUNK_TOKEN_COUNT,
        ),
        &DELTANET_INPUT_NEURAL_ENGINE_SHARES,
    );
}

fn sweep_splits(full_fixture: &ProjectionFixture, neural_engine_shares: &[usize]) {
    let runtime = initialize_runtime();
    let gpu_only_full = measure_gpu_only_full(&runtime, full_fixture);
    eprintln!(
        "[neural-engine-journey] GPU-only baseline ({} → {}, MLX fp16): median {:.3} ms",
        full_fixture.shape.input_channels(),
        full_fixture.shape.output_channels(),
        milliseconds(gpu_only_full.median)
    );
    for neural_engine_channels in neural_engine_shares {
        let timings = measure_split(&runtime, full_fixture, *neural_engine_channels);
        print_split(*neural_engine_channels, &timings, &gpu_only_full);
    }
}

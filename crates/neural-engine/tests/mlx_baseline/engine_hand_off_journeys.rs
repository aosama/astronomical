//! Rung 4 of issue #161: what does passing data between MLX and the Neural
//! Engine cost, and how much of the two-engine gain survives it?
//!
//! Earlier journeys wrote activations outside the timed section and never
//! merged the two output slices, so their speedups were upper bounds. One
//! call here goes from a bf16 token-major activation, the form MLX holds
//! during prefill, to one bf16 token-major result:
//!
//! 1. the GPU converts the activation to channel-major fp16;
//! 2. the GPU slice starts asynchronously, so the host copy into the input
//!    surface does not leave the GPU idle;
//! 3. the Neural Engine runs its slice next to the GPU;
//! 4. the Neural Engine's surface becomes an MLX array and the slices merge.
//!
//! The hand-off steps themselves live in `engine_hand_off`, shared with the
//! token-split journeys. The baseline is the same activation through MLX's
//! 4-bit matrix multiply. Both arms evaluate once per operation. Serving batches operations into
//! larger graphs, so these figures do not yet include the synchronization
//! points a hand-off would force on it.

use std::time::{Duration, Instant};

use astronomical_neural_engine::core_ml_projection::{CoreMlComputeUnits, CoreMlProjection};
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxDtype, raw};
use serial_test::serial;

use super::concurrent_split_journeys::{
    assert_slice_matches_reference, load_neural_engine_slice, milliseconds,
};
use super::dense_model_ab_journeys::{
    DenseOperation, QuantizedOperation, channel_major_values, prepare_quantized_operation,
    quantized_form, trailing_rows,
};
use super::engine_hand_off::{
    OutputHandOff, StageDurations, StageSummaries, channel_major_fp16, summarize_stages,
    surface_as_array, write_prepared_input,
};
use super::mlx_projection_journeys::{build_projection, initialize_runtime, time_projection};
use crate::hello::measured_projection::{
    MEASURED_CALL_COUNT, MINIMUM_WARM_UP_CALL_COUNT, MINIMUM_WARM_UP_DURATION, report_progress,
};
use crate::hello::projection_fixture::ProjectionFixture;

const NEURAL_ENGINE_FINISH_TIMEOUT: Duration = Duration::from_secs(5);
const CONTRACTION_PART_WIDTH: usize = 2048;
// Shares in sixteenths of the output channels, bracketing the 44 percent the
// dense A/B found and the shift toward the Neural Engine or away from it that
// bf16 arithmetic and the later Neural Engine start could each cause.
const NEURAL_ENGINE_SHARE_SIXTEENTHS: [usize; 5] = [5, 6, 7, 8, 9];

struct HandOffArrays<'arrays> {
    token_major_bf16: &'arrays MlxArray,
    gpu_packed_weights: MlxArray,
    gpu_scales: MlxArray,
    gpu_biases: MlxArray,
}

/// Whether MLX kept the surface's own memory rather than silently copying,
/// which it does when Metal refuses to map the pointer. Timing alone could
/// not tell these apart, and the zero-copy arm would report a copy's cost.
fn surface_stays_mapped(runtime: &MlxRuntime, projection: &CoreMlProjection) -> bool {
    let token_count = projection.token_count() as i32;
    let surface = projection.lock_output_surface().expect("the surface locks");
    let fp16_values = surface
        .contiguous_fp16_values()
        .expect("a 2,048-token surface has no row padding");
    let surface_array = surface_as_array(
        &surface,
        [projection.output_channel_count() as i32, token_count],
        OutputHandOff::ZeroCopySurface,
    );
    runtime
        .evaluate_arrays(&[&surface_array])
        .expect("the surface array evaluates");
    // SAFETY: the array is evaluated and live; only its address is compared.
    let array_values = unsafe { raw::mlx_array_data_float16(surface_array.raw()) };
    std::ptr::eq(array_values.cast::<u16>(), fp16_values)
}

/// Joins the slices into `[1, tokens, channels]` bf16, the baseline's form.
fn merge_slices(
    runtime: &MlxRuntime,
    surface_array: &MlxArray,
    gpu_slice: &MlxArray,
    token_count: i32,
    neural_engine_channels: i32,
) -> MlxArray {
    let tokens_first = runtime
        .transpose_axes(surface_array, &[1, 0])
        .expect("the surface transposes");
    let batched = runtime
        .reshape(&tokens_first, &[1, token_count, neural_engine_channels])
        .expect("the surface reshapes");
    let bfloat = runtime
        .astype(&batched, MlxDtype::BFloat16)
        .expect("the surface converts");
    runtime
        .concatenate_axis(&[&bfloat, gpu_slice], 2)
        .expect("the slices merge")
}

fn merge_from_surface(
    runtime: &MlxRuntime,
    projection: &CoreMlProjection,
    gpu_slice: &MlxArray,
    output_hand_off: OutputHandOff,
) -> MlxArray {
    let token_count = gpu_slice.shape()[1];
    let neural_engine_channels = projection.output_channel_count() as i32;
    let surface = projection.lock_output_surface().expect("the surface locks");
    let surface_array = surface_as_array(
        &surface,
        [neural_engine_channels, token_count],
        output_hand_off,
    );
    let merged = merge_slices(
        runtime,
        &surface_array,
        gpu_slice,
        token_count,
        neural_engine_channels,
    );
    runtime
        .evaluate_arrays(&[&merged])
        .expect("the merge evaluates");
    merged
}

fn run_hand_off_call(
    runtime: &MlxRuntime,
    arrays: &HandOffArrays<'_>,
    projection: &mut CoreMlProjection,
    output_hand_off: OutputHandOff,
) -> (StageDurations, MlxArray) {
    let call_started_at = Instant::now();
    let prepared = channel_major_fp16(runtime, arrays.token_major_bf16);
    runtime
        .evaluate_arrays(&[&prepared])
        .expect("the activations prepare");
    let input_preparation = call_started_at.elapsed();
    let gpu_form = quantized_form(
        arrays.token_major_bf16,
        &arrays.gpu_packed_weights,
        &arrays.gpu_scales,
        &arrays.gpu_biases,
    );
    let gpu_slice = build_projection(runtime, &gpu_form);
    runtime
        .async_eval_arrays(&[&gpu_slice])
        .expect("the GPU slice starts");
    let write_started_at = Instant::now();
    write_prepared_input(&prepared, projection);
    let surface_write = write_started_at.elapsed();
    let engines_started_at = Instant::now();
    let pending = projection.begin_prediction().expect("the slice starts");
    runtime
        .evaluate_arrays(&[&gpu_slice])
        .expect("the GPU slice evaluates");
    pending
        .finish(NEURAL_ENGINE_FINISH_TIMEOUT)
        .expect("the Neural Engine slice finishes");
    let engines_after_write = engines_started_at.elapsed();
    let merge_started_at = Instant::now();
    let merged = merge_from_surface(runtime, projection, &gpu_slice, output_hand_off);
    let stages = StageDurations {
        input_preparation,
        surface_write,
        engines_after_write,
        merge: merge_started_at.elapsed(),
        total: call_started_at.elapsed(),
    };
    (stages, merged)
}

/// Warms up, measures, and returns the stage summaries plus the last result.
fn time_hand_off(
    runtime: &MlxRuntime,
    arrays: &HandOffArrays<'_>,
    projection: &mut CoreMlProjection,
    output_hand_off: OutputHandOff,
    label: &str,
) -> (StageSummaries, MlxArray) {
    let warm_up_started_at = Instant::now();
    let mut warm_up_call_count = 0;
    while warm_up_call_count < MINIMUM_WARM_UP_CALL_COUNT
        || warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION
    {
        run_hand_off_call(runtime, arrays, projection, output_hand_off);
        warm_up_call_count += 1;
    }
    let mut stage_samples = Vec::with_capacity(MEASURED_CALL_COUNT);
    let mut last_merged = MlxArray::from_bool(false);
    let measuring_started_at = Instant::now();
    let mut last_progress_at = measuring_started_at;
    for call_index in 0..MEASURED_CALL_COUNT {
        let (stages, merged) = run_hand_off_call(runtime, arrays, projection, output_hand_off);
        stage_samples.push(stages);
        last_merged = merged;
        if last_progress_at.elapsed() >= Duration::from_secs(5) {
            report_progress(label, call_index + 1, measuring_started_at.elapsed());
            last_progress_at = Instant::now();
        }
    }
    (summarize_stages(&stage_samples), last_merged)
}

/// The operation in the form real artifacts serve: bf16 activations with bf16
/// affine scales and biases. Pairing bf16 activations with the fixture's fp16
/// parameters would be a third form no artifact uses: those two dtypes promote
/// to fp32, which made a first version of this journey look far slower.
struct Bf16Operation {
    token_major_activations: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

fn bf16_operation(runtime: &MlxRuntime, quantized: &QuantizedOperation) -> Bf16Operation {
    let token_major_activations = runtime
        .astype(&quantized.token_major_activations, MlxDtype::BFloat16)
        .expect("activations convert");
    let scales = runtime
        .astype(&quantized.scales, MlxDtype::BFloat16)
        .expect("scales convert");
    let biases = runtime
        .astype(&quantized.biases, MlxDtype::BFloat16)
        .expect("biases convert");
    runtime
        .evaluate_arrays(&[&token_major_activations, &scales, &biases])
        .expect("the bf16 operation evaluates");
    Bf16Operation {
        token_major_activations,
        scales,
        biases,
    }
}

fn print_stages(
    operation: &DenseOperation,
    neural_engine_channels: usize,
    output_hand_off: OutputHandOff,
    summaries: &StageSummaries,
    baseline_median: Duration,
) {
    eprintln!(
        "[hand-off] {} | {neural_engine_channels} NE channels, {output_hand_off:?} | prepare {:.3} \
         + write {:.3} + engines {:.3} + merge {:.3} = {:.3} ms (p90 {:.3}) | baseline {:.3} ms → {:.3}×",
        operation.label,
        milliseconds(summaries.input_preparation.median),
        milliseconds(summaries.surface_write.median),
        milliseconds(summaries.engines_after_write.median),
        milliseconds(summaries.merge.median),
        milliseconds(summaries.total.median),
        milliseconds(summaries.total.ninetieth_percentile),
        milliseconds(baseline_median),
        baseline_median.as_secs_f64() / summaries.total.median.as_secs_f64(),
    );
}

/// One share with one output hand-off; returns the median total.
fn measure_share(
    runtime: &MlxRuntime,
    operation: &DenseOperation,
    quantized: &QuantizedOperation,
    bf16: &Bf16Operation,
    neural_engine_channels: usize,
    output_hand_offs: &[OutputHandOff],
    baseline_median: Duration,
) -> Duration {
    let neural_engine_fixture = quantized
        .fixture
        .output_channel_slice(0, neural_engine_channels);
    let contraction_part_count = (operation.input_channels / CONTRACTION_PART_WIDTH).max(1);
    let model_directory = tempfile::tempdir().expect("a scratch directory for the model");
    let mut projection = load_neural_engine_slice(
        &neural_engine_fixture,
        contraction_part_count,
        model_directory.path(),
    );
    let arrays = HandOffArrays {
        token_major_bf16: &bf16.token_major_activations,
        gpu_packed_weights: trailing_rows(
            runtime,
            &quantized.packed_weights,
            neural_engine_channels,
        ),
        gpu_scales: trailing_rows(runtime, &bf16.scales, neural_engine_channels),
        gpu_biases: trailing_rows(runtime, &bf16.biases, neural_engine_channels),
    };
    eprintln!(
        "[hand-off] {} {neural_engine_channels} NE: surface mapped without a copy = {}",
        operation.label,
        surface_stays_mapped(runtime, &projection)
    );
    let mut best_total = Duration::MAX;
    for output_hand_off in output_hand_offs {
        let label = format!("hand-off {} {neural_engine_channels} NE", operation.label);
        let (summaries, merged) =
            time_hand_off(runtime, &arrays, &mut projection, *output_hand_off, &label);
        print_stages(
            operation,
            neural_engine_channels,
            *output_hand_off,
            &summaries,
            baseline_median,
        );
        let merged_values = channel_major_values(runtime, &merged, operation.output_channels);
        assert_slice_matches_reference(&quantized.fixture, &merged_values, "merged");
        best_total = best_total.min(summaries.total.median);
    }
    best_total
}

/// The GPU alone in pure fp16 (activations and affine parameters). A control,
/// so that a two-engine gain is also judged against this other GPU-only form.
fn print_fp16_control(
    runtime: &MlxRuntime,
    operation: &DenseOperation,
    quantized: &QuantizedOperation,
    bf16_median: Duration,
) {
    let fp16_form = quantized_form(
        &quantized.token_major_activations,
        &quantized.packed_weights,
        &quantized.scales,
        &quantized.biases,
    );
    let fp16_label = format!("fp16 control {}", operation.label);
    let fp16_median = time_projection(runtime, &fp16_form, &fp16_label).median;
    eprintln!(
        "[hand-off] {} | GPU alone: pure bf16 {:.3} ms, pure fp16 {:.3} ms → {:.3}×",
        operation.label,
        milliseconds(bf16_median),
        milliseconds(fp16_median),
        bf16_median.as_secs_f64() / fp16_median.as_secs_f64(),
    );
}

/// Runs every share of one operation; returns (baseline, best) per chunk.
fn measure_operation(runtime: &MlxRuntime, operation: &DenseOperation) -> (f64, f64) {
    let quantized = prepare_quantized_operation(runtime, operation);
    let bf16 = bf16_operation(runtime, &quantized);
    let baseline_form = quantized_form(
        &bf16.token_major_activations,
        &quantized.packed_weights,
        &bf16.scales,
        &bf16.biases,
    );
    let baseline_label = format!("baseline {}", operation.label);
    let baseline_median = time_projection(runtime, &baseline_form, &baseline_label).median;
    print_fp16_control(runtime, operation, &quantized, baseline_median);
    let mut best_total = Duration::MAX;
    for share_sixteenths in NEURAL_ENGINE_SHARE_SIXTEENTHS {
        let neural_engine_channels = operation.output_channels * share_sixteenths / 16;
        let share_total = measure_share(
            runtime,
            operation,
            &quantized,
            &bf16,
            neural_engine_channels,
            &[OutputHandOff::ZeroCopySurface, OutputHandOff::CopiedSurface],
            baseline_median,
        );
        best_total = best_total.min(share_total);
    }
    let layers = operation.layer_count as f64;
    (
        milliseconds(baseline_median) * layers,
        milliseconds(best_total) * layers,
    )
}

fn measure_model(model_label: &str, operations: &[DenseOperation]) {
    let runtime = initialize_runtime();
    let mut baseline_chunk_milliseconds = 0.0;
    let mut best_chunk_milliseconds = 0.0;
    for operation in operations {
        let (baseline, best) = measure_operation(&runtime, operation);
        eprintln!(
            "[hand-off] SUMMARY {} ×{} | per chunk {baseline:.1} → {best:.1} ms | {:.3}×",
            operation.label,
            operation.layer_count,
            baseline / best
        );
        baseline_chunk_milliseconds += baseline;
        best_chunk_milliseconds += best;
    }
    eprintln!(
        "[hand-off] MODEL {model_label}: fixed-shape projections per chunk \
         {baseline_chunk_milliseconds:.1} → {best_chunk_milliseconds:.1} ms | {:.3}×",
        baseline_chunk_milliseconds / best_chunk_milliseconds
    );
}

#[test]
#[serial(neural_engine)]
#[ignore = "times every hand-off stage of Ornith-1.5-9B's projections on the real engines"]
fn should_measure_the_dense_model_hand_off() {
    measure_model(
        "Ornith-1.5-9B dense",
        &[
            DenseOperation {
                label: "DeltaNet fused input 4096→12352",
                input_channels: 4096,
                output_channels: 12_352,
                layer_count: 24,
            },
            DenseOperation {
                label: "attention fused input 4096→10240",
                input_channels: 4096,
                output_channels: 10_240,
                layer_count: 8,
            },
            DenseOperation {
                label: "mixer output 4096→4096",
                input_channels: 4096,
                output_channels: 4096,
                layer_count: 32,
            },
            DenseOperation {
                label: "MLP gate+up 4096→24576",
                input_channels: 4096,
                output_channels: 24_576,
                layer_count: 32,
            },
            DenseOperation {
                label: "MLP down 12288→4096",
                input_channels: 12_288,
                output_channels: 4096,
                layer_count: 32,
            },
        ],
    );
}

#[test]
#[serial(neural_engine)]
#[ignore = "times every hand-off stage of the 35B-A3B MoE's dense projections on the real engines"]
fn should_measure_the_mixture_of_experts_model_hand_off() {
    measure_model(
        "Ornith-1.5-35B-A3B MoE",
        &[
            DenseOperation {
                label: "DeltaNet fused input 2048→12352",
                input_channels: 2048,
                output_channels: 12_352,
                layer_count: 30,
            },
            DenseOperation {
                label: "attention fused input 2048→9216",
                input_channels: 2048,
                output_channels: 9216,
                layer_count: 10,
            },
            DenseOperation {
                label: "mixer output 4096→2048",
                input_channels: 4096,
                output_channels: 2048,
                layer_count: 40,
            },
            DenseOperation {
                label: "post-mixer fused input 2048→1281",
                input_channels: 2048,
                output_channels: 1281,
                layer_count: 40,
            },
            DenseOperation {
                label: "shared-expert down 512→2048",
                input_channels: 512,
                output_channels: 2048,
                layer_count: 40,
            },
        ],
    );
}

fn cpu_only_projection(
    fixture: &ProjectionFixture,
    directory: &std::path::Path,
) -> CoreMlProjection {
    let mut projection = CoreMlProjection::load(
        fixture.shape,
        &fixture.fp16_weights,
        CoreMlComputeUnits::CpuOnly,
        1,
        directory,
    )
    .expect("the projection loads");
    projection
        .write_activations(&fixture.fp16_activations)
        .expect("the activations fit");
    projection.predict().expect("the projection runs");
    projection
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads a small Core ML projection to inspect its output surface"]
fn should_read_the_same_values_through_the_locked_surface_as_by_copy() {
    let fixture = ProjectionFixture::with_shape(8, 4, 2048);
    let directory = tempfile::tempdir().expect("a scratch directory");
    let projection = cpu_only_projection(&fixture, directory.path());
    let copied_values = projection.read_projected().expect("the output copies");

    let surface = projection.lock_output_surface().expect("the surface locks");
    let fp16_values = surface
        .contiguous_fp16_values()
        .expect("2,048 fp16 tokens fill a row exactly");
    // SAFETY: the guard is alive and the surface holds this many values.
    let surface_values =
        unsafe { std::slice::from_raw_parts(fp16_values, fixture.shape.projected_element_count()) };

    assert_eq!(surface_values, copied_values.as_slice());
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads a small Core ML projection to inspect its output surface"]
fn should_refuse_a_contiguous_view_of_a_surface_with_padded_rows() {
    // 17 fp16 tokens are 34 bytes, which no surface row alignment leaves unpadded.
    let fixture = ProjectionFixture::with_shape(8, 4, 17);
    let directory = tempfile::tempdir().expect("a scratch directory");
    let projection = cpu_only_projection(&fixture, directory.path());

    let surface = projection.lock_output_surface().expect("the surface locks");

    assert!(surface.contiguous_fp16_values().is_none());
}

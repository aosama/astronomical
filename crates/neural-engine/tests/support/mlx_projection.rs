//! The projection on the GPU through MLX, shared by every journey that compares
//! the Neural Engine with the engine serving actually uses.

use std::time::{Duration, Instant};

use astronomical_neural_engine::half_precision::f16_bits_slice_to_f32;
use astronomical_neural_engine::numerical_comparison::compare_values;
use astronomical_neural_engine::timing_summary::TimingSummary;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};

use super::measured_projection::{
    MEASURED_CALL_COUNT, MINIMUM_WARM_UP_CALL_COUNT, MINIMUM_WARM_UP_DURATION, report_progress,
};
use super::projection_fixture::ProjectionFixture;

const ACTIVE_MEMORY_LIMIT_BYTES: usize = 8 * 1024 * 1024 * 1024;
const ALLOCATOR_CACHE_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub(crate) const PRODUCTION_QUANTIZATION_GROUP_SIZE: i32 = 64;

pub(crate) enum MlxProjectionForm<'arrays> {
    Dense {
        weights: &'arrays MlxArray,
        activations: &'arrays MlxArray,
    },
    QuantizedAffine {
        token_major_activations: &'arrays MlxArray,
        packed_weights: &'arrays MlxArray,
        scales: &'arrays MlxArray,
        biases: &'arrays MlxArray,
        bits: i32,
    },
}

pub(crate) fn build_projection(runtime: &MlxRuntime, form: &MlxProjectionForm<'_>) -> MlxArray {
    match form {
        MlxProjectionForm::Dense {
            weights,
            activations,
        } => runtime
            .matmul(weights, activations)
            .expect("dense matmul builds"),
        MlxProjectionForm::QuantizedAffine {
            token_major_activations,
            packed_weights,
            scales,
            biases,
            bits,
        } => runtime
            .quantized_matmul_affine(
                token_major_activations,
                packed_weights,
                scales,
                biases,
                true,
                PRODUCTION_QUANTIZATION_GROUP_SIZE,
                *bits,
            )
            .expect("quantized matmul builds"),
    }
}

/// Times graph construction plus a blocking evaluation, the MLX analogue of
/// one Core ML prediction call.
pub(crate) fn time_projection(
    runtime: &MlxRuntime,
    form: &MlxProjectionForm<'_>,
    engine_label: &str,
) -> TimingSummary {
    let warm_up_started_at = Instant::now();
    let mut warm_up_call_count = 0;
    while warm_up_call_count < MINIMUM_WARM_UP_CALL_COUNT
        || warm_up_started_at.elapsed() < MINIMUM_WARM_UP_DURATION
    {
        let projected = build_projection(runtime, form);
        runtime
            .evaluate_arrays(&[&projected])
            .expect("warm-up evaluates");
        warm_up_call_count += 1;
    }
    let mut call_durations = Vec::with_capacity(MEASURED_CALL_COUNT);
    let measuring_started_at = Instant::now();
    let mut last_progress_at = measuring_started_at;
    for call_index in 0..MEASURED_CALL_COUNT {
        let call_started_at = Instant::now();
        let projected = build_projection(runtime, form);
        runtime
            .evaluate_arrays(&[&projected])
            .expect("measured call evaluates");
        call_durations.push(call_started_at.elapsed());
        if last_progress_at.elapsed() >= Duration::from_secs(5) {
            report_progress(engine_label, call_index + 1, measuring_started_at.elapsed());
            last_progress_at = Instant::now();
        }
    }
    TimingSummary::from_samples(&call_durations).expect("calls were timed")
}

pub(crate) fn upload(
    runtime: &MlxRuntime,
    values: &[f32],
    shape: &[i32],
    dtype: MlxDtype,
) -> MlxArray {
    let single_precision = runtime
        .array_from_f32(values, shape)
        .expect("values upload");
    let converted = runtime
        .astype(&single_precision, dtype)
        .expect("values convert");
    runtime
        .evaluate_arrays(&[&converted])
        .expect("upload evaluates");
    converted
}

pub(crate) fn initialize_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(ACTIVE_MEMORY_LIMIT_BYTES, ALLOCATOR_CACHE_LIMIT_BYTES)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes")
}

pub(crate) fn measure_dense(
    runtime: &MlxRuntime,
    fixture: &ProjectionFixture,
    dtype: MlxDtype,
) -> (TimingSummary, f64) {
    let shape = fixture.shape;
    let weights = upload(
        runtime,
        &f16_bits_slice_to_f32(&fixture.fp16_weights),
        &[
            shape.output_channels() as i32,
            shape.input_channels() as i32,
        ],
        dtype,
    );
    let activations = upload(
        runtime,
        &f16_bits_slice_to_f32(&fixture.fp16_activations),
        &[shape.input_channels() as i32, shape.token_count() as i32],
        dtype,
    );
    let form = MlxProjectionForm::Dense {
        weights: &weights,
        activations: &activations,
    };
    let engine_label = format!("MLX GPU dense {dtype:?}");
    let timing = time_projection(runtime, &form, &engine_label);
    print_timing(&engine_label, fixture, &timing);
    let projected = build_projection(runtime, &form);
    let single_precision = runtime
        .astype(&projected, MlxDtype::Float32)
        .expect("result converts");
    let candidate = runtime
        .array_to_vec_f32(&single_precision)
        .expect("result reads back");
    let comparison = compare_values(&fixture.reference_projected, &candidate).expect("same length");
    eprintln!(
        "  vs fp32 reference: max abs {:.3e}, mean abs {:.3e}, cosine {:.8}, non-finite {}",
        comparison.maximum_absolute_error,
        comparison.mean_absolute_error,
        comparison.cosine_similarity,
        comparison.non_finite_candidate_count
    );
    (timing, comparison.cosine_similarity)
}

pub(crate) fn token_major(fixture: &ProjectionFixture) -> Vec<f32> {
    let shape = fixture.shape;
    let channel_major = f16_bits_slice_to_f32(&fixture.fp16_activations);
    let mut token_major_activations = vec![0.0_f32; channel_major.len()];
    for channel_index in 0..shape.input_channels() {
        for token_index in 0..shape.token_count() {
            token_major_activations[token_index * shape.input_channels() + channel_index] =
                channel_major[channel_index * shape.token_count() + token_index];
        }
    }
    token_major_activations
}

pub(crate) fn print_timing(
    engine_label: &str,
    fixture: &ProjectionFixture,
    timing: &TimingSummary,
) {
    eprintln!(
        "[neural-engine-hello] {engine_label}\n  call: median {:.2} ms, p90 {:.2} ms, min {:.2} ms, \
         max {:.2} ms over {} → {:.2} TFLOP/s",
        timing.median.as_secs_f64() * 1000.0,
        timing.ninetieth_percentile.as_secs_f64() * 1000.0,
        timing.minimum.as_secs_f64() * 1000.0,
        timing.maximum.as_secs_f64() * 1000.0,
        timing.sample_count,
        timing.median_teraflops(fixture.shape.floating_point_operations()),
    );
}

pub(crate) fn token_range(
    runtime: &MlxRuntime,
    token_major_activations: &MlxArray,
    first_token: usize,
    end_token: usize,
    channel_count: usize,
) -> MlxArray {
    runtime
        .slice(
            token_major_activations,
            &[0, first_token as i32, 0],
            &[1, end_token as i32, channel_count as i32],
            &[1, 1, 1],
        )
        .expect("the tokens slice")
}

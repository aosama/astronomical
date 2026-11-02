//! The same mixer output projection on the GPU through MLX, Astronomical's
//! serving engine, so the Neural Engine is compared with the real rival and
//! not only with Core ML's own GPU path.

use std::time::{Duration, Instant};

use astronomical_neural_engine::half_precision::f16_bits_slice_to_f32;
use astronomical_neural_engine::numerical_comparison::compare_values;
use astronomical_neural_engine::timing_summary::TimingSummary;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};
use serial_test::serial;

use crate::hello::measured_projection::{
    MEASURED_CALL_COUNT, MINIMUM_WARM_UP_CALL_COUNT, MINIMUM_WARM_UP_DURATION, report_progress,
};
use crate::hello::projection_fixture::ProjectionFixture;

const ACTIVE_MEMORY_LIMIT_BYTES: usize = 8 * 1024 * 1024 * 1024;
const ALLOCATOR_CACHE_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
// The MoE artifact stores these dense projections as 8-bit affine, group 64.
const PRODUCTION_QUANTIZATION_BITS: i32 = 8;
pub(super) const PRODUCTION_QUANTIZATION_GROUP_SIZE: i32 = 64;

pub(super) enum MlxProjectionForm<'arrays> {
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

pub(super) fn build_projection(runtime: &MlxRuntime, form: &MlxProjectionForm<'_>) -> MlxArray {
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
pub(super) fn time_projection(
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

pub(super) fn upload(
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

fn print_timing(engine_label: &str, fixture: &ProjectionFixture, timing: &TimingSummary) {
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

pub(super) fn initialize_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(ACTIVE_MEMORY_LIMIT_BYTES, ALLOCATOR_CACHE_LIMIT_BYTES)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes")
}

pub(super) fn measure_dense(
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

pub(super) fn token_major(fixture: &ProjectionFixture) -> Vec<f32> {
    let shape = fixture.shape;
    let channel_major = f16_bits_slice_to_f32(&fixture.fp16_activations);
    let mut token_major_values = vec![0.0_f32; channel_major.len()];
    for channel_index in 0..shape.input_channels() {
        for token_index in 0..shape.token_count() {
            token_major_values[token_index * shape.input_channels() + channel_index] =
                channel_major[channel_index * shape.token_count() + token_index];
        }
    }
    token_major_values
}

fn measure_quantized_affine() -> TimingSummary {
    let fixture = ProjectionFixture::mixer_output_projection();
    let runtime = initialize_runtime();
    let shape = fixture.shape;
    let weights = upload(
        &runtime,
        &f16_bits_slice_to_f32(&fixture.fp16_weights),
        &[
            shape.output_channels() as i32,
            shape.input_channels() as i32,
        ],
        MlxDtype::BFloat16,
    );
    let (packed_weights, scales, biases) = runtime
        .quantize_affine(
            &weights,
            PRODUCTION_QUANTIZATION_GROUP_SIZE,
            PRODUCTION_QUANTIZATION_BITS,
        )
        .expect("weights quantize");
    runtime
        .evaluate_arrays(&[&packed_weights, &scales, &biases])
        .expect("quantized weights evaluate");
    let token_major_activations = upload(
        &runtime,
        &token_major(&fixture),
        &[1, shape.token_count() as i32, shape.input_channels() as i32],
        MlxDtype::BFloat16,
    );
    let form = MlxProjectionForm::QuantizedAffine {
        token_major_activations: &token_major_activations,
        packed_weights: &packed_weights,
        scales: &scales,
        biases: &biases,
        bits: PRODUCTION_QUANTIZATION_BITS,
    };
    let engine_label = "MLX GPU 8-bit affine (production format)";
    let timing = time_projection(&runtime, &form, engine_label);
    print_timing(engine_label, &fixture, &timing);
    timing
}

#[test]
#[serial(neural_engine)]
#[ignore = "runs the 34-GFLOP projection on the real GPU through MLX"]
fn should_measure_the_mixer_output_projection_on_the_gpu_through_mlx() {
    let fixture = ProjectionFixture::mixer_output_projection();
    let runtime = initialize_runtime();
    let (_float16_timing, float16_cosine) = measure_dense(&runtime, &fixture, MlxDtype::Float16);
    let (_bfloat16_timing, bfloat16_cosine) = measure_dense(&runtime, &fixture, MlxDtype::BFloat16);
    let _quantized_timing = measure_quantized_affine();

    assert!(float16_cosine >= 0.9999, "fp16 cosine {float16_cosine}");
    // bf16 keeps three fewer mantissa bits than fp16, so its bound is looser.
    assert!(bfloat16_cosine >= 0.999, "bf16 cosine {bfloat16_cosine}");
}

//! The same mixer output projection on the GPU through MLX, Astronomical's
//! serving engine, so the Neural Engine is compared with the real rival and
//! not only with Core ML's own GPU path.

use astronomical_neural_engine::weights::half_precision::f16_bits_slice_to_f32;
use astronomical_neural_engine_research::timing_summary::TimingSummary;
use mlx_c_rust::MlxDtype;
use serial_test::serial;

use crate::support::mlx_projection::{
    MlxProjectionForm, PRODUCTION_QUANTIZATION_GROUP_SIZE, initialize_runtime, measure_dense,
    print_timing, time_projection, token_major, upload,
};
use crate::support::projection_fixture::ProjectionFixture;

// The MoE artifact stores these dense projections as 8-bit affine, group 64.
const PRODUCTION_QUANTIZATION_BITS: i32 = 8;

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

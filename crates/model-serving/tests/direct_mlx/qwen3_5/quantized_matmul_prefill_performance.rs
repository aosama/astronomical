//! Affine quantized-matmul kernel parity at Ornith-1.5-35B-A3B shapes.
//!
//! Prefill and decode both run roughly 11-13% below the 30k-token REST
//! baseline, and the fused attention kernels match one-for-one in isolation.
//! The remaining large shared component is the affine quantized matmul that
//! backs every projection and every routed/shared expert. This bench measures
//! the dense quantized matmul and the gather form at production shapes under
//! Astronomical's MLX build for isolated kernel-level comparison.

use std::time::{Duration, Instant};

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
const BENCH_ITERATIONS: usize = 8;

#[tokio::test]
#[ignore = "measures affine quantized-matmul graphics-processor timing at production shapes"]
async fn should_measure_affine_quantized_matmul_costs_at_production_shapes() {
    timeout(
        Duration::from_secs(115),
        measure_affine_quantized_matmul_costs(),
    )
    .await
    .expect("the quantized-matmul bench must finish within 115 seconds");
}

async fn measure_affine_quantized_matmul_costs() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the quantized-matmul bench memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");

    // (shape name, activation token count, output rows, input columns)
    let shapes = [
        ("prefill-qkv", 2048, 8192, 2048),
        ("prefill-down", 2048, 2048, 4096),
        ("decode-qkv", 1, 8192, 2048),
        ("decode-down", 1, 2048, 4096),
    ];
    for (shape_name, token_count, output_rows, input_columns) in shapes {
        let weights = sample_tensor(&runtime, &[output_rows, input_columns]);
        let (packed_weight, quantization_scales, quantization_biases) = runtime
            .quantize_affine(&weights, GROUP_SIZE, BITS)
            .expect("the weights should quantize");
        runtime
            .evaluate_arrays(&[&packed_weight, &quantization_scales, &quantization_biases])
            .expect("the quantized weights should evaluate");
        let activations = sample_tensor(&runtime, &[1, token_count, input_columns]);
        let millis = bench(&runtime, || {
            let output = runtime
                .quantized_matmul_affine(
                    &activations,
                    &packed_weight,
                    &quantization_scales,
                    &quantization_biases,
                    true,
                    GROUP_SIZE,
                    BITS,
                )
                .expect("the quantized matmul should build");
            runtime
                .evaluate_arrays(&[&output])
                .expect("the quantized matmul should evaluate");
        });
        eprintln!("[qmm-bench] dense {shape_name} tokens={token_count} {millis:.3} ms");
    }
}

fn sample_tensor(runtime: &MlxRuntime, shape: &[i32]) -> MlxArray {
    let element_count: usize = shape.iter().map(|dimension| *dimension as usize).product();
    let sample_values: Vec<f32> = (0..element_count)
        .map(|element_index| (element_index % 251) as f32 * 0.001 - 0.125)
        .collect();
    runtime
        .astype(
            &runtime
                .array_from_f32(&sample_values, shape)
                .expect("the sampled tensor should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the sampled tensor should cast to bfloat16")
}

fn bench(runtime: &MlxRuntime, mut measured: impl FnMut()) -> f64 {
    let ramp_started_at = Instant::now();
    while ramp_started_at.elapsed() < Duration::from_millis(1200) {
        measured();
    }
    runtime
        .synchronize_gpu_stream()
        .expect("the clock-ramp chain should drain");
    let started_at = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        measured();
    }
    started_at.elapsed().as_secs_f64() * 1000.0 / BENCH_ITERATIONS as f64
}

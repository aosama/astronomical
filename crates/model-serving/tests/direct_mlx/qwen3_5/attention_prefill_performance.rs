//! Attention and matmul kernel parity at Ornith-1.5-35B-A3B prefill/decode shapes.
//!
//! The 30k-token REST baseline shows Astronomical slower not only on prefill
//! (~11%) but on decode (~13%) and first-token latency (~11%) too — a broad,
//! roughly uniform gap rather than a single op family. A uniform gap points at
//! the kernels/build rather than model composition. This bench measures the
//! dominant shared kernels under Astronomical's MLX build for isolated
//! kernel-level comparison at the same shapes.

use std::time::{Duration, Instant};

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

const HEAD_COUNT: i32 = 16;
const KEY_VALUE_HEAD_COUNT: i32 = 2;
const HEAD_DIMENSION: i32 = 256;
const PREFILL_QUERY_TOKENS: i32 = 2_048;
const BENCH_ITERATIONS: usize = 4;

#[tokio::test]
#[ignore = "measures attention graphics-processor timing at production shapes"]
async fn should_measure_attention_kernel_costs_at_production_shapes() {
    timeout(Duration::from_secs(115), measure_attention_kernel_costs())
        .await
        .expect("the attention bench must finish within 115 seconds");
}

async fn measure_attention_kernel_costs() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the attention bench memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");

    let attention_scale = (HEAD_DIMENSION as f32).powf(-0.5);
    for key_value_token_count in [2048, 7719, 16384, 30720] {
        let queries = sample_tensor(
            &runtime,
            &[1, HEAD_COUNT, PREFILL_QUERY_TOKENS, HEAD_DIMENSION],
        );
        let keys = sample_tensor(
            &runtime,
            &[
                1,
                KEY_VALUE_HEAD_COUNT,
                key_value_token_count,
                HEAD_DIMENSION,
            ],
        );
        let values = sample_tensor(
            &runtime,
            &[
                1,
                KEY_VALUE_HEAD_COUNT,
                key_value_token_count,
                HEAD_DIMENSION,
            ],
        );
        let causal_prefill_millis = bench(&runtime, || {
            let output = runtime
                .causal_scaled_dot_product_attention(&queries, &keys, &values, attention_scale)
                .expect("causal attention should build");
            runtime
                .evaluate_arrays(&[&output])
                .expect("causal attention should evaluate");
        });
        eprintln!(
            "[attn-bench] causal q={PREFILL_QUERY_TOKENS} kv={key_value_token_count} {causal_prefill_millis:.3} ms"
        );

        let decode_queries = sample_tensor(&runtime, &[1, HEAD_COUNT, 1, HEAD_DIMENSION]);
        let decode_millis = bench(&runtime, || {
            let output = runtime
                .scaled_dot_product_attention(&decode_queries, &keys, &values, attention_scale)
                .expect("decode attention should build");
            runtime
                .evaluate_arrays(&[&output])
                .expect("decode attention should evaluate");
        });
        eprintln!("[attn-bench] decode q=1 kv={key_value_token_count} {decode_millis:.3} ms");
    }
}

fn sample_tensor(runtime: &MlxRuntime, shape: &[i32]) -> MlxArray {
    let element_count: usize = shape.iter().map(|dimension| *dimension as usize).product();
    let sample_values: Vec<f32> = (0..element_count)
        .map(|element_index| (element_index % 251) as f32 * 0.004 - 0.5)
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

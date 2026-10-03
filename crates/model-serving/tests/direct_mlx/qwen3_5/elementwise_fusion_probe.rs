//! Elementwise-chain fusion probe at scaled prefill shapes.
//!
//! The oMLX parity gap concentrates in small elementwise chains (the GDN
//! convolution section). Sub-0.3 millisecond probes at 2,048-token shapes sit
//! inside run-to-run noise, so this probe scales the workload four-fold and
//! reports the median of interleaved per-iteration timings. Interleaving makes
//! clock or machine drift affect every probe equally; the median kills outliers.
//!
//! Probes, smallest traffic first:
//! - `x * 1.0` (multiply scalar, 2 arrays touched)
//! - `sigmoid(x)`
//! - `silu(x)` = `x * sigmoid(x)` — one kernel when MLX embeds the unary into
//!   the binary, two kernels when it does not
//! - `silu(silu(x))` — doubles the node count; the within-rig ratio is the
//!   fusion verdict
//! - `concat+silu` and `concat+conv1d+silu` — the production convolution
//!   section chains

use std::time::{Duration, Instant};

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

const TOKEN_COUNT: i32 = 8_192;
const CONVOLUTION_DIMENSION: i32 = 8_192;
const CONVOLUTION_KERNEL_DIMENSION: i32 = 4;
const PROBE_ITERATIONS: usize = 31;

#[tokio::test]
#[ignore = "measures elementwise-chain graphics-processor timing with interleaved medians"]
async fn should_probe_elementwise_chain_costs_at_scaled_prefill_shapes() {
    timeout(Duration::from_secs(115), probe_elementwise_chains())
        .await
        .expect("the elementwise probe must finish within 115 seconds");
}

async fn probe_elementwise_chains() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the probe memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");

    let element_count = (TOKEN_COUNT * CONVOLUTION_DIMENSION) as usize;
    let sample_values: Vec<f32> = (0..element_count)
        .map(|element_index| (element_index % 251) as f32 * 0.004 - 0.5)
        .collect();
    let input = runtime
        .astype(
            &runtime
                .array_from_f32(&sample_values, &[1, TOKEN_COUNT, CONVOLUTION_DIMENSION])
                .expect("the probe input should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the probe input should cast to bfloat16");
    let rolling_state = runtime
        .zeros(
            &[1, CONVOLUTION_KERNEL_DIMENSION - 1, CONVOLUTION_DIMENSION],
            MlxDtype::BFloat16,
        )
        .expect("the rolling state should be valid");
    let weight_values: Vec<f32> = (0..(CONVOLUTION_DIMENSION * CONVOLUTION_KERNEL_DIMENSION)
        as usize)
        .map(|element_index| ((element_index % 17) as f32 - 8.0) * 0.1)
        .collect();
    let weight = runtime
        .astype(
            &runtime
                .array_from_f32(
                    &weight_values,
                    &[CONVOLUTION_DIMENSION, CONVOLUTION_KERNEL_DIMENSION, 1],
                )
                .expect("the conv weight should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the conv weight should cast to bfloat16");
    runtime
        .evaluate_arrays(&[&input, &rolling_state, &weight])
        .expect("the probe operands should evaluate");
    let compiled_elementwise_graphs =
        astronomical_runtime_integration::MlxCompiledElementwiseGraphs::new()
            .expect("the compiled elementwise graphs should build");

    let probes: Vec<(&str, Box<dyn Fn() -> MlxArray + '_>)> = vec![
        (
            "x*1.0",
            Box::new(|| {
                runtime
                    .multiply_scalar(&input, 1.0)
                    .expect("multiply scalar should build")
            }),
        ),
        (
            "sigmoid(x)",
            Box::new(|| runtime.sigmoid(&input).expect("sigmoid should build")),
        ),
        (
            "silu(x)",
            Box::new(|| runtime.silu(&input).expect("silu should build")),
        ),
        (
            "silu(silu(x))",
            Box::new(|| {
                let inner = runtime.silu(&input).expect("inner silu should build");
                runtime.silu(&inner).expect("outer silu should build")
            }),
        ),
        (
            "concat+silu",
            Box::new(|| {
                let concatenated = runtime
                    .concatenate_axis(&[&rolling_state, &input], 1)
                    .expect("concat should build");
                runtime.silu(&concatenated).expect("silu should build")
            }),
        ),
        (
            "concat+conv1d+silu",
            Box::new(|| {
                let concatenated = runtime
                    .concatenate_axis(&[&rolling_state, &input], 1)
                    .expect("concat should build");
                let convolved = runtime
                    .conv1d(&concatenated, &weight, 1, 0, 1, CONVOLUTION_DIMENSION)
                    .expect("conv1d should build");
                runtime.silu(&convolved).expect("silu should build")
            }),
        ),
        (
            "compiled-silu(x)",
            Box::new(|| {
                runtime
                    .apply_compiled_silu(&compiled_elementwise_graphs, &input)
                    .expect("compiled silu should build")
            }),
        ),
    ];

    // Warm every probe once, then interleave: one iteration of each probe per
    // round, so machine drift lands on all probes equally.
    for (_, probe) in &probes {
        let output = probe();
        runtime
            .evaluate_arrays(&[&output])
            .expect("the warmup probe should evaluate");
    }
    runtime
        .synchronize_gpu_stream()
        .expect("the warmup probes should drain");

    let probe_count = probes.len();
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); probe_count];
    let ramp_started_at = Instant::now();
    let mut round_index = 0;
    while ramp_started_at.elapsed() < Duration::from_millis(1500) || round_index < PROBE_ITERATIONS
    {
        for (probe_index, (_, probe)) in probes.iter().enumerate() {
            let started_at = Instant::now();
            let output = probe();
            runtime
                .evaluate_arrays(&[&output])
                .expect("the probe should evaluate");
            samples[probe_index].push(started_at.elapsed().as_secs_f64() * 1000.0);
        }
        round_index += 1;
        if round_index >= PROBE_ITERATIONS * 4 {
            break;
        }
    }
    runtime
        .synchronize_gpu_stream()
        .expect("the probes should drain");

    let mut medians = Vec::with_capacity(probe_count);
    for (probe_name, _) in &probes {
        let probe_samples = &mut samples[medians.len()];
        probe_samples.sort_by(|left, right| left.partial_cmp(right).expect("finite times"));
        let median = probe_samples[probe_samples.len() / 2];
        medians.push(median);
        eprintln!(
            "[elementwise-probe] {probe_name} median {median:.3} ms ({:.2} us/token) samples={}",
            median * 1000.0 / f64::from(TOKEN_COUNT),
            probe_samples.len()
        );
    }
    let silu_median = medians[2];
    let double_silu_median = medians[3];
    eprintln!(
        "[elementwise-probe] fusion ratio silu(silu)/silu = {:.2} (1.0=fused, 2.0=unfused)",
        double_silu_median / silu_median
    );
    let compiled_silu_median = medians[6];
    eprintln!(
        "[elementwise-probe] compiled-silu/sigmoid ratio = {:.2} (1.0 means the compiled graph fused to one kernel)",
        compiled_silu_median / medians[1]
    );

    // Numerics: the compiled graph must agree with the raw two-op composition
    // within bf16 rounding of the same op sequence.
    let numerics_element_count = 64 * 64;
    let numerics_values: Vec<f32> = (0..numerics_element_count)
        .map(|element_index| (element_index % 61) as f32 * 0.02 - 0.6)
        .collect();
    let numerics_input = runtime
        .astype(
            &runtime
                .array_from_f32(&numerics_values, &[1, 64, 64])
                .expect("the numerics input should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the numerics input should cast");
    let raw_output = runtime
        .silu(&numerics_input)
        .expect("raw silu should build");
    let compiled_output = runtime
        .apply_compiled_silu(&compiled_elementwise_graphs, &numerics_input)
        .expect("compiled silu should build");
    let raw_float32_output = runtime
        .astype(&raw_output, MlxDtype::Float32)
        .expect("the raw silu values should cast");
    let compiled_float32_output = runtime
        .astype(&compiled_output, MlxDtype::Float32)
        .expect("the compiled silu values should cast");
    runtime
        .evaluate_arrays(&[&raw_float32_output, &compiled_float32_output])
        .expect("the numerics outputs should evaluate");
    let raw_values = raw_float32_output.to_vec_f32().expect("raw silu values");
    let compiled_values = compiled_float32_output
        .to_vec_f32()
        .expect("compiled silu values");
    let maximum_difference = raw_values
        .iter()
        .zip(compiled_values.iter())
        .map(|(raw_value, compiled_value)| (raw_value - compiled_value).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("[elementwise-probe] raw-vs-compiled max abs diff = {maximum_difference:.6}");
    assert!(
        maximum_difference <= 0.004,
        "compiled silu must match the raw composition within one bf16 ulp: {maximum_difference}"
    );
}

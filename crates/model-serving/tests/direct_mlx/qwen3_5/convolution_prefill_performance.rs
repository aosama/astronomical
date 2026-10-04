//! Isolated convolution-chain costs at gated-delta prefill shapes.
//!
//! The prefill attribution journey located Astronomical's linear-attention
//! prefill excess in the convolution section (rolling-state concat, depthwise
//! conv1d, SiLU). This bench runs that exact chain in isolation at the
//! production shapes so its op-level cost can be compared against Python
//! microbenches without attribution synchronization or model-serving
//! context in the way.

use std::time::{Duration, Instant};

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_model_serving::ConvolutionState;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

const TOKEN_COUNT: i32 = 2048;
const CONVOLUTION_DIMENSION: i32 = 8_192;
const CONVOLUTION_KERNEL_DIMENSION: i32 = 4;
const BENCH_ITERATIONS: usize = 8;

#[tokio::test]
#[ignore = "measures convolution-chain graphics-processor timing at production prefill shapes"]
async fn should_measure_isolated_convolution_chain_costs_at_prefill_shapes() {
    timeout(Duration::from_secs(115), measure_convolution_chain_costs())
        .await
        .expect("the isolated convolution bench must finish within 115 seconds");
}

async fn measure_convolution_chain_costs() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    // Production-shaped memory limits: the tiny direct-MLX test defaults (8 MB
    // allocator cache) force fresh Metal buffer allocations for every op and
    // would measure allocator churn instead of kernel cost.
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the convolution bench memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");

    let token_values: Vec<f32> = (0..TOKEN_COUNT * CONVOLUTION_DIMENSION)
        .map(|element_index| (element_index % 251) as f32 * 0.004 - 0.5)
        .collect();
    let mixed_queries_keys_values = runtime
        .astype(
            &runtime
                .array_from_f32(&token_values, &[1, TOKEN_COUNT, CONVOLUTION_DIMENSION])
                .expect("the mixed qkv activations should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the mixed qkv activations should cast to bfloat16");
    let weight_values: Vec<f32> = (0..CONVOLUTION_DIMENSION * CONVOLUTION_KERNEL_DIMENSION)
        .map(|element_index| ((element_index % 17) as f32 - 8.0) * 0.1)
        .collect();
    let convolution_weight = runtime
        .astype(
            &runtime
                .reshape(
                    &runtime
                        .array_from_f32(
                            &weight_values,
                            &[CONVOLUTION_DIMENSION, CONVOLUTION_KERNEL_DIMENSION, 1],
                        )
                        .expect("the flat conv weight should be valid"),
                    &[CONVOLUTION_DIMENSION, CONVOLUTION_KERNEL_DIMENSION, 1],
                )
                .expect("the conv weight should reshape"),
            MlxDtype::BFloat16,
        )
        .expect("the conv weight should cast to bfloat16");

    // Two state-priming updates so the bench measures the steady state (a
    // populated rolling buffer), not the first-chunk zero initialization.
    for _ in 0..2 {
        let mut convolution_state =
            ConvolutionState::empty_with_shape(CONVOLUTION_KERNEL_DIMENSION, CONVOLUTION_DIMENSION);
        let convolution_input = convolution_state
            .update(&runtime, &mixed_queries_keys_values, TOKEN_COUNT)
            .expect("the priming convolution update should be valid");
        let convolution_output = runtime
            .conv1d(
                &convolution_input,
                &convolution_weight,
                1,
                0,
                1,
                CONVOLUTION_DIMENSION,
            )
            .expect("the priming conv1d should be valid");
        runtime
            .evaluate_arrays(&[&convolution_output])
            .expect("the priming conv1d should evaluate");
    }

    let mut convolution_state =
        ConvolutionState::empty_with_shape(CONVOLUTION_KERNEL_DIMENSION, CONVOLUTION_DIMENSION);
    let convolution_input = convolution_state
        .update(&runtime, &mixed_queries_keys_values, TOKEN_COUNT)
        .expect("the steady-state convolution input should be valid");
    runtime
        .evaluate_arrays(&[&convolution_input])
        .expect("the steady-state convolution input should evaluate");

    let section_millis = bench_chain(&runtime, || {
        let convolution_output = runtime
            .conv1d(
                &convolution_input,
                &convolution_weight,
                1,
                0,
                1,
                CONVOLUTION_DIMENSION,
            )
            .expect("the conv1d should be valid");
        let convolution_output = runtime
            .silu(&convolution_output)
            .expect("the silu should be valid");
        runtime
            .evaluate_arrays(&[&convolution_output])
            .expect("the conv chain should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] conv1d+silu {section_millis:.3} ms/call ({:.2} us/token)",
        section_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );

    // The exact conv-section composite: fresh zero rolling buffer, lazy
    // concat, depthwise conv1d, SiLU, one evaluation. The Python microbench
    // measures precisely this chain, so this is the matched apples-to-apples
    // comparison for the convolution section.
    let zero_rolling_buffer = runtime
        .zeros(
            &[1, CONVOLUTION_KERNEL_DIMENSION - 1, CONVOLUTION_DIMENSION],
            MlxDtype::BFloat16,
        )
        .expect("the zero rolling buffer should be valid");
    runtime
        .evaluate_arrays(&[&zero_rolling_buffer])
        .expect("the zero rolling buffer should evaluate");
    let composite_millis = bench_chain(&runtime, || {
        let conv_input = runtime
            .concatenate_axis(&[&zero_rolling_buffer, &mixed_queries_keys_values], 1)
            .expect("the composite concat should be valid");
        let convolution_output = runtime
            .conv1d(
                &conv_input,
                &convolution_weight,
                1,
                0,
                1,
                CONVOLUTION_DIMENSION,
            )
            .expect("the composite conv1d should be valid");
        let convolution_output = runtime
            .silu(&convolution_output)
            .expect("the composite silu should be valid");
        runtime
            .evaluate_arrays(&[&convolution_output])
            .expect("the composite conv chain should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] full chain concat+conv1d+silu {composite_millis:.3} ms/call ({:.2} us/token)",
        composite_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );

    // Elementwise-fusion probe: silu is sigmoid+multiply (two graph nodes).
    // Fused, doubling the node count barely moves time; unfused it roughly
    // doubles. The within-rig ratio is self-normalizing across rigs.
    let fusion_probe_input = runtime
        .astype(&convolution_input, MlxDtype::BFloat16)
        .expect("the fusion probe input should cast");
    runtime
        .evaluate_arrays(&[&fusion_probe_input])
        .expect("the fusion probe input should evaluate");
    let silu_once_millis = bench_chain(&runtime, || {
        let activated = runtime
            .silu(&fusion_probe_input)
            .expect("single silu should be valid");
        runtime
            .evaluate_arrays(&[&activated])
            .expect("single silu should evaluate");
    });
    eprintln!("[gdn-conv-bench] fusion probe silu(x) {silu_once_millis:.3} ms/call");
    let silu_twice_millis = bench_chain(&runtime, || {
        let activated = runtime
            .silu(&fusion_probe_input)
            .expect("first silu should be valid");
        let activated = runtime
            .silu(&activated)
            .expect("second silu should be valid");
        runtime
            .evaluate_arrays(&[&activated])
            .expect("double silu should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] fusion probe silu(silu(x)) {silu_twice_millis:.3} ms/call ratio={:.2}",
        silu_twice_millis / silu_once_millis
    );

    let sigmoid_alone_millis = bench_chain(&runtime, || {
        let activated = runtime
            .sigmoid(&fusion_probe_input)
            .expect("sigmoid probe should be valid");
        runtime
            .evaluate_arrays(&[&activated])
            .expect("sigmoid probe should evaluate");
    });
    eprintln!("[gdn-conv-bench] fusion probe sigmoid(x) alone {sigmoid_alone_millis:.3} ms/call");

    // Chain B: concat+silu without conv1d, so subtracting it from the full
    // chain isolates the depthwise conv1d cost in both rigs.
    let concat_silu_millis = bench_chain(&runtime, || {
        let conv_input = runtime
            .concatenate_axis(&[&zero_rolling_buffer, &mixed_queries_keys_values], 1)
            .expect("the concat-silu concat should be valid");
        let activated = runtime
            .silu(&conv_input)
            .expect("the concat-silu activation should be valid");
        runtime
            .evaluate_arrays(&[&activated])
            .expect("the concat-silu chain should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] chain concat+silu {concat_silu_millis:.3} ms/call ({:.2} us/token)",
        concat_silu_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );

    let conv1d_only_millis = bench_chain(&runtime, || {
        let convolution_output = runtime
            .conv1d(
                &convolution_input,
                &convolution_weight,
                1,
                0,
                1,
                CONVOLUTION_DIMENSION,
            )
            .expect("the conv1d should be valid");
        runtime
            .evaluate_arrays(&[&convolution_output])
            .expect("the conv1d should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] conv1d only {conv1d_only_millis:.3} ms/call ({:.2} us/token)",
        conv1d_only_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );

    let conv1d_output = runtime
        .conv1d(
            &convolution_input,
            &convolution_weight,
            1,
            0,
            1,
            CONVOLUTION_DIMENSION,
        )
        .expect("the conv1d output for the silu bench should be valid");
    runtime
        .evaluate_arrays(&[&conv1d_output])
        .expect("the conv1d output should evaluate");
    let silu_only_millis = bench_chain(&runtime, || {
        let silu_output = runtime
            .silu(&conv1d_output)
            .expect("the silu should be valid");
        runtime
            .evaluate_arrays(&[&silu_output])
            .expect("the silu should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] silu only {silu_only_millis:.3} ms/call ({:.2} us/token)",
        silu_only_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );

    let concat_millis = bench_chain(&runtime, || {
        let concatenated = runtime
            .concatenate_axis(&[&convolution_input, &mixed_queries_keys_values], 1)
            .expect("the concat should be valid");
        runtime
            .evaluate_arrays(&[&concatenated])
            .expect("the concat should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] concat [1,3,8192]+[1,2048,8192] {concat_millis:.3} ms/call ({:.2} us/token)",
        concat_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );

    // Dtype sensitivity: the same conv1d with float32 input and weight, in
    // case the production path silently upcasts.
    let float32_convolution_input = runtime
        .astype(&convolution_input, MlxDtype::Float32)
        .expect("the float32 conv input should cast");
    let float32_convolution_weight = runtime
        .astype(&convolution_weight, MlxDtype::Float32)
        .expect("the float32 conv weight should cast");
    runtime
        .evaluate_arrays(&[&float32_convolution_input, &float32_convolution_weight])
        .expect("the float32 conv operands should evaluate");
    let float32_millis = bench_chain(&runtime, || {
        let convolution_output = runtime
            .conv1d(
                &float32_convolution_input,
                &float32_convolution_weight,
                1,
                0,
                1,
                CONVOLUTION_DIMENSION,
            )
            .expect("the float32 conv1d should be valid");
        runtime
            .evaluate_arrays(&[&convolution_output])
            .expect("the float32 conv1d should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] conv1d only float32 {float32_millis:.3} ms/call ({:.2} us/token)",
        float32_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );

    // The eval+synchronize floor: one blocking evaluation of a tiny array.
    // Any attribution section carries at least this cost, so section costs
    // below roughly twice the floor are not separable GPU time.
    let floor_token = runtime
        .array_from_f32(&[1.0], &[1])
        .expect("the floor token should be valid");
    let floor_millis = bench_chain(&runtime, || {
        runtime
            .evaluate_arrays(&[&floor_token])
            .expect("the floor eval should pass");
    });
    eprintln!("[gdn-conv-bench] eval floor {floor_millis:.3} ms/call");

    // Production rolling-state shape: the state operand is a strided slice of
    // the previous chunk's wider buffer, not a contiguous array. A reference
    // cache API materializes its window instead; if concat on the strided view
    // is slower, that difference is the convolution-section excess.
    let production_state = runtime
        .slice(
            &convolution_input,
            &[0, TOKEN_COUNT, 0],
            &[
                1,
                TOKEN_COUNT + CONVOLUTION_KERNEL_DIMENSION - 1,
                CONVOLUTION_DIMENSION,
            ],
            &[1, 1, 1],
        )
        .expect("the strided production state slice should be valid");
    runtime
        .evaluate_arrays(&[&production_state])
        .expect("the strided production state should evaluate");
    let strided_state_concat_millis = bench_chain(&runtime, || {
        let concatenated = runtime
            .concatenate_axis(&[&production_state, &mixed_queries_keys_values], 1)
            .expect("the strided-state concat should be valid");
        runtime
            .evaluate_arrays(&[&concatenated])
            .expect("the strided-state concat should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] concat strided-state {strided_state_concat_millis:.3} ms/call ({:.2} us/token)",
        strided_state_concat_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );

    // Server-context simulation: the production worker keeps the model's
    // wired weights resident while prefill runs. Hold a comparable amount of
    // live memory and re-bench the chain; if the chain degrades, the
    // convolution-section excess is allocator/context pressure rather than a
    // kernel difference.
    let pressure_blocks: Vec<MlxArray> = (0..64)
        .map(|_| {
            runtime
                .zeros(&[1, 8192, 8192], MlxDtype::BFloat16)
                .expect("the pressure block should allocate")
        })
        .collect();
    let pressure_references: Vec<&MlxArray> = pressure_blocks.iter().collect();
    runtime
        .evaluate_arrays(&pressure_references)
        .expect("the pressure blocks should evaluate");
    eprintln!(
        "[gdn-conv-bench] pressure held: {:.1} GB",
        64.0 * f64::from(8192 * 8192) * 2.0 / 1e9
    );

    let pressured_chain_millis = bench_chain(&runtime, || {
        let convolution_output = runtime
            .conv1d(
                &convolution_input,
                &convolution_weight,
                1,
                0,
                1,
                CONVOLUTION_DIMENSION,
            )
            .expect("the pressured conv1d should be valid");
        let convolution_output = runtime
            .silu(&convolution_output)
            .expect("the pressured silu should be valid");
        runtime
            .evaluate_arrays(&[&convolution_output])
            .expect("the pressured conv chain should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] conv1d+silu under pressure {pressured_chain_millis:.3} ms/call ({:.2} us/token)",
        pressured_chain_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );
    let pressured_concat_millis = bench_chain(&runtime, || {
        let concatenated = runtime
            .concatenate_axis(&[&production_state, &mixed_queries_keys_values], 1)
            .expect("the pressured concat should be valid");
        runtime
            .evaluate_arrays(&[&concatenated])
            .expect("the pressured concat should evaluate");
    });
    eprintln!(
        "[gdn-conv-bench] concat under pressure {pressured_concat_millis:.3} ms/call ({:.2} us/token)",
        pressured_concat_millis * 1000.0 / f64::from(TOKEN_COUNT)
    );
}

fn bench_chain(runtime: &MlxRuntime, mut chain: impl FnMut()) -> f64 {
    let _ = runtime;
    // Ramp graphics-processor clocks to the sustained state before timing:
    // a sub-second bench otherwise measures idle-clocked hardware and reports
    // several times the steady-state cost.
    let ramp_started_at = Instant::now();
    while ramp_started_at.elapsed() < Duration::from_millis(1500) {
        chain();
    }
    runtime
        .synchronize_gpu_stream()
        .expect("the clock-ramp chain should drain");
    let started_at = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        chain();
    }
    started_at.elapsed().as_secs_f64() * 1000.0 / BENCH_ITERATIONS as f64
}

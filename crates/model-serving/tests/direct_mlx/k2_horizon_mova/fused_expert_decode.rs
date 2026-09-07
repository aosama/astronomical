//! Fused expert decode parity gates against the gathered reference paths.
//!
//! The fused single-token kernels must reproduce the production gathered
//! quantized matmul chain (projection, SwiGLU, router-weighted reduction)
//! within a bounded tolerance on identical inputs. These tests use realistic
//! K2 geometry so the gate covers the exact production numeric path.

use astronomical_model_serving::PerformanceAttribution;
use astronomical_model_serving::{
    FusedExpertDecodeKernels, K2HorizonMoVAAffineLinear, gathered_fused_swiglu,
    gathered_value_experts,
};
use astronomical_runtime_integration::{
    MlxArray, MlxCompiledSwiGlu, MlxDtype, MlxMemoryLimits, MlxRuntime,
};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const PROBE_EXPERT_COUNT: i32 = 8;
const PROBE_FF_INTERMEDIATE: i32 = 768;
const PROBE_VALUE_OUTPUT: i32 = 1024;
const PROBE_HIDDEN_SIZE: i32 = 2560;
const PROBE_GROUP_SIZE: u32 = 64;
const PROBE_BITS: u32 = 4;

#[tokio::test]
async fn should_match_gathered_value_experts_within_tolerance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut attribution = PerformanceAttribution::disabled();

    let value_experts = quantized_stack(
        &runtime,
        PROBE_EXPERT_COUNT,
        PROBE_VALUE_OUTPUT,
        PROBE_HIDDEN_SIZE,
    );
    let flat_hidden = activation_vector(&runtime, 1.0);
    let indices = index_vector(&runtime, &[2, 6, 1, 7]);
    let scores = score_vector(&runtime, &[0.7, 0.2, 0.1, 0.9]);

    let reference = gathered_value_experts(
        &runtime,
        &flat_hidden,
        &value_experts,
        &indices,
        &scores,
        &mut attribution,
    )
    .expect("gathered reference");
    let kernels = FusedExpertDecodeKernels::new().expect("fused kernels compile");
    let fused = kernels
        .fused_value_expert_decode(
            &runtime,
            &flat_hidden,
            &indices,
            &scores,
            &value_experts,
            &mut attribution,
        )
        .expect("fused decode");
    assert_close(&runtime, &reference, &fused, "fused value experts");
}

#[tokio::test]
async fn should_match_gathered_routed_ffn_within_tolerance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut attribution = PerformanceAttribution::disabled();

    let switch_gate_up = quantized_stack(
        &runtime,
        PROBE_EXPERT_COUNT,
        PROBE_FF_INTERMEDIATE * 2,
        PROBE_HIDDEN_SIZE,
    );
    let switch_down = quantized_stack(
        &runtime,
        PROBE_EXPERT_COUNT,
        PROBE_HIDDEN_SIZE,
        PROBE_FF_INTERMEDIATE,
    );
    let flat_hidden = activation_vector(&runtime, 2.0);
    let indices = index_vector(&runtime, &[0, 5, 3, 4]);
    let scores = score_vector(&runtime, &[0.6, 0.15, 0.05, 0.2]);

    let reference = gathered_fused_swiglu_reference(
        &runtime,
        &flat_hidden,
        &switch_gate_up,
        &switch_down,
        &indices,
        &scores,
    )
    .expect("gathered reference");
    let kernels = FusedExpertDecodeKernels::new().expect("fused kernels compile");
    let fused = kernels
        .fused_routed_ffn_decode(
            &runtime,
            &flat_hidden,
            &indices,
            &scores,
            &switch_gate_up,
            &switch_down,
            &mut attribution,
        )
        .expect("fused decode");
    assert_close(&runtime, &reference, &fused, "fused routed FFN");
}

/// Builds one stacked 4-bit group-64 affine expert projection from a
/// deterministic small-amplitude wave so quantization stays representative.
fn quantized_stack(
    runtime: &MlxRuntime,
    expert_count: i32,
    output_rows: i32,
    input_dimension: i32,
) -> K2HorizonMoVAAffineLinear {
    let weight_element_count = (expert_count * output_rows * input_dimension) as usize;
    let weights: Vec<f32> = (0..weight_element_count)
        .map(|element_index| ((element_index as f32 + 1.0) * 0.013).sin() * 0.35)
        .collect();
    let dense = runtime
        .array_from_f32(&weights, &[expert_count, output_rows, input_dimension])
        .expect("expert weights");
    let (packed, scales, biases) = runtime
        .quantize_affine(&dense, PROBE_GROUP_SIZE as i32, PROBE_BITS as i32)
        .expect("quantize expert weights");
    K2HorizonMoVAAffineLinear::new(packed, scales, biases, PROBE_BITS, PROBE_GROUP_SIZE, None)
}

/// One bf16 activation row whose entries are small deterministic waves.
fn activation_vector(runtime: &MlxRuntime, seed_offset: f32) -> MlxArray {
    let values: Vec<f32> = (0..PROBE_HIDDEN_SIZE)
        .map(|element_index| {
            let position = element_index as f32 + seed_offset;
            (position * 0.017).sin() * 0.5
        })
        .collect();
    runtime
        .array_from_f32(&values, &[1, PROBE_HIDDEN_SIZE])
        .and_then(|array| runtime.astype(&array, MlxDtype::BFloat16))
        .expect("flat hidden activation")
}

fn index_vector(runtime: &MlxRuntime, expert_indices: &[u32]) -> MlxArray {
    runtime
        .array_from_u32(expert_indices, &[1, expert_indices.len() as i32])
        .expect("routed indices")
}

fn score_vector(runtime: &MlxRuntime, score_values: &[f32]) -> MlxArray {
    runtime
        .array_from_f32(score_values, &[1, score_values.len() as i32])
        .and_then(|array| runtime.astype(&array, MlxDtype::BFloat16))
        .expect("routed scores")
}

/// The production gathered SwiGLU chain, which is also the decoder's prefill
/// and fallback path, used here as the numeric oracle.
fn gathered_fused_swiglu_reference(
    runtime: &MlxRuntime,
    flat_hidden: &MlxArray,
    gate_up: &K2HorizonMoVAAffineLinear,
    down: &K2HorizonMoVAAffineLinear,
    indices: &MlxArray,
    scores: &MlxArray,
) -> Result<MlxArray, Box<dyn std::error::Error>> {
    let compiled_swiglu = MlxCompiledSwiGlu::new()?;
    let mut attribution = PerformanceAttribution::disabled();
    Ok(gathered_fused_swiglu(
        runtime,
        flat_hidden,
        gate_up,
        down,
        indices,
        scores,
        &compiled_swiglu,
        None,
        &mut attribution,
    )?)
}

fn assert_close(runtime: &MlxRuntime, reference: &MlxArray, fused: &MlxArray, label: &str) {
    let reference_values = runtime
        .astype(reference, MlxDtype::Float32)
        .expect("cast reference")
        .to_vec_f32()
        .expect("read reference");
    let fused_values = runtime
        .astype(fused, MlxDtype::Float32)
        .expect("cast fused")
        .to_vec_f32()
        .expect("read fused");
    let maximum_error = reference_values
        .iter()
        .zip(fused_values.iter())
        .map(|(reference, fused)| (reference - fused).abs())
        .fold(0.0_f32, f32::max);
    // The gathered reference rounds intermediates through bfloat16 while the
    // fused kernels accumulate in float32, so agreement within a couple of
    // bf16 units in the last place (relative ~2^-7) is the correct gate.
    let reference_magnitude = reference_values
        .iter()
        .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
    let relative_error = maximum_error / reference_magnitude.max(1.0);
    eprintln!("{label} max error {maximum_error:.5} relative {relative_error:.5}");
    assert!(
        relative_error < 0.01,
        "the fused expert decode must match the gathered reference within a relative tolerance, got {relative_error}"
    );
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("test memory limits"),
    )
    .expect("direct MLX runtime")
}

#[tokio::test]
async fn should_probe_fused_quantized_expert_decode_capability() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut attribution = PerformanceAttribution::disabled();
    let capabilities =
        astronomical_model_serving::worker_process_kernel_capabilities(&runtime, &mut attribution);
    let verdict = capabilities
        .verdict(astronomical_model_serving::CustomMetalKernelFamily::FusedQuantizedExpertDecode);
    eprintln!("fused capability verdict: {verdict:?}");
    assert!(matches!(
        verdict,
        astronomical_model_serving::CustomKernelVerdict::Supported
    ));
}

#[tokio::test]
#[ignore = "per-kernel decode micro-benchmark: fused kernels against the gathered chain"]
async fn should_profile_fused_kernels_against_gathered_chain() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut attribution = PerformanceAttribution::disabled();

    let switch_gate_up = quantized_stack(
        &runtime,
        PROBE_EXPERT_COUNT,
        PROBE_FF_INTERMEDIATE * 2,
        PROBE_HIDDEN_SIZE,
    );
    let switch_down = quantized_stack(
        &runtime,
        PROBE_EXPERT_COUNT,
        PROBE_HIDDEN_SIZE,
        PROBE_FF_INTERMEDIATE,
    );
    let flat_hidden = activation_vector(&runtime, 2.0);
    let indices = index_vector(&runtime, &[0, 5, 3, 4]);
    let scores = score_vector(&runtime, &[0.6, 0.15, 0.05, 0.2]);
    let kernels = FusedExpertDecodeKernels::new().expect("fused kernels compile");

    let value_experts = quantized_stack(
        &runtime,
        PROBE_EXPERT_COUNT,
        PROBE_VALUE_OUTPUT,
        PROBE_HIDDEN_SIZE,
    );
    const ITERATIONS: u32 = 200;
    let value_gathered_start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        let value_gathered = gathered_value_experts(
            &runtime,
            &flat_hidden,
            &value_experts,
            &indices,
            &scores,
            &mut attribution,
        )
        .expect("gathered values");
        runtime
            .evaluate_arrays(&[&value_gathered])
            .expect("evaluate gathered values");
    }
    let value_gathered_microseconds =
        value_gathered_start.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(ITERATIONS);
    let value_fused_start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        let value_fused = kernels
            .fused_value_expert_decode(
                &runtime,
                &flat_hidden,
                &indices,
                &scores,
                &value_experts,
                &mut attribution,
            )
            .expect("fused values");
        runtime
            .evaluate_arrays(&[&value_fused])
            .expect("evaluate fused values");
    }
    let value_fused_microseconds =
        value_fused_start.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(ITERATIONS);
    eprintln!(
        "per-layer MoVA values: gathered {value_gathered_microseconds:.1} us, fused {value_fused_microseconds:.1} us"
    );
    let gathered_start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        let gathered_output = gathered_fused_swiglu_reference(
            &runtime,
            &flat_hidden,
            &switch_gate_up,
            &switch_down,
            &indices,
            &scores,
        )
        .expect("gathered");
        runtime
            .evaluate_arrays(&[&gathered_output])
            .expect("evaluate gathered");
    }
    let gathered_microseconds =
        gathered_start.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(ITERATIONS);
    let fused_start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        let fused_output = kernels
            .fused_routed_ffn_decode(
                &runtime,
                &flat_hidden,
                &indices,
                &scores,
                &switch_gate_up,
                &switch_down,
                &mut attribution,
            )
            .expect("fused decode");
        runtime
            .evaluate_arrays(&[&fused_output])
            .expect("evaluate fused");
    }
    let fused_microseconds =
        fused_start.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(ITERATIONS);
    eprintln!(
        "per-layer routed FFN: gathered {gathered_microseconds:.1} us, fused {fused_microseconds:.1} us"
    );
    let _ = (switch_gate_up, switch_down, indices, scores);
}

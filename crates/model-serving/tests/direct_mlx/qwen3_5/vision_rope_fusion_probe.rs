//! Vision rotate-half RoPE fusion probe at scaled vision shapes.
//!
//! The vision tower applies rotate-half RoPE to Q and K in every one of its 27
//! blocks. The composed path dispatches eight small elementwise kernels per
//! call; the compiled graph fuses the elementwise tail into one kernel (the two
//! head-half slices and the rotate-half concat remain separate). This probe
//! scales the patch count to real image workloads and reports the median of
//! interleaved per-iteration timings so machine drift lands on both paths
//! equally.

use std::time::{Duration, Instant};

use astronomical_runtime_integration::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxDtype, MlxMemoryLimits, MlxRuntime,
};
use tokio::time::timeout;

const HEAD_COUNT: i32 = 16;
const HEAD_DIMENSION: i32 = 72;
// Qwen3.5 vision processes thousands of patches per image; 8192 keeps the
// elementwise traffic large enough to clear run-to-run noise.
const PATCH_COUNT: i32 = 8_192;
const PROBE_ITERATIONS: usize = 31;

#[tokio::test]
#[ignore = "measures composed-vs-compiled vision RoPE graphics-processor timing with interleaved medians"]
async fn should_probe_vision_rope_fusion_costs_at_scaled_vision_shapes() {
    timeout(Duration::from_secs(115), probe_vision_rope_fusion())
        .await
        .expect("the vision RoPE probe must finish within 115 seconds");
}

async fn probe_vision_rope_fusion() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the probe memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let compiled_elementwise_graphs =
        MlxCompiledElementwiseGraphs::new().expect("the compiled elementwise graphs should build");

    let attention_states = bf16_array(
        &runtime,
        &patterned_values(
            (PATCH_COUNT * HEAD_COUNT * HEAD_DIMENSION) as usize,
            0.01,
            -0.5,
        ),
        &[PATCH_COUNT, HEAD_COUNT, HEAD_DIMENSION],
    );
    let rotary_cosines = f32_array(
        &runtime,
        &patterned_values((PATCH_COUNT * HEAD_DIMENSION) as usize, 0.02, -1.0),
        &[PATCH_COUNT, 1, HEAD_DIMENSION],
    );
    let rotary_sines = f32_array(
        &runtime,
        &patterned_values((PATCH_COUNT * HEAD_DIMENSION) as usize, 0.03, 0.25),
        &[PATCH_COUNT, 1, HEAD_DIMENSION],
    );
    runtime
        .evaluate_arrays(&[&attention_states, &rotary_cosines, &rotary_sines])
        .expect("the probe operands should evaluate");

    let half_head_dimension = HEAD_DIMENSION / 2;
    let first_half = runtime
        .slice(
            &attention_states,
            &[0, 0, 0],
            &[PATCH_COUNT, HEAD_COUNT, half_head_dimension],
            &[1, 1, 1],
        )
        .expect("the first head half should slice");
    let second_half = runtime
        .slice(
            &attention_states,
            &[0, 0, half_head_dimension],
            &[PATCH_COUNT, HEAD_COUNT, HEAD_DIMENSION],
            &[1, 1, 1],
        )
        .expect("the second head half should slice");

    let probes: Vec<(&str, Box<dyn Fn() -> MlxArray + '_>)> = vec![
        (
            "composed-rope",
            Box::new(|| {
                composed_vision_rope(
                    &runtime,
                    &attention_states,
                    &rotary_cosines,
                    &rotary_sines,
                    &first_half,
                    &second_half,
                )
            }),
        ),
        (
            "compiled-rope",
            Box::new(|| {
                runtime
                    .apply_compiled_vision_rope(
                        &compiled_elementwise_graphs,
                        &attention_states,
                        &rotary_cosines,
                        &rotary_sines,
                        &first_half,
                        &second_half,
                    )
                    .expect("the compiled vision rope should build")
            }),
        ),
    ];

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
            "[vision-rope-probe] {probe_name} median {median:.3} ms samples={}",
            probe_samples.len()
        );
    }
    eprintln!(
        "[vision-rope-probe] compiled/composed ratio = {:.2} (< 1.0 means the compiled graph is faster)",
        medians[1] / medians[0]
    );

    let composed_values = runtime
        .astype(&probes[0].1(), MlxDtype::Float32)
        .expect("the composed output should cast")
        .to_vec_f32()
        .expect("the composed output should evaluate");
    let compiled_values = runtime
        .astype(&probes[1].1(), MlxDtype::Float32)
        .expect("the compiled output should cast")
        .to_vec_f32()
        .expect("the compiled output should evaluate");
    let maximum_difference = composed_values
        .iter()
        .zip(compiled_values.iter())
        .map(|(composed_value, compiled_value)| (composed_value - compiled_value).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("[vision-rope-probe] composed-vs-compiled max abs diff = {maximum_difference:.6}");
    assert!(
        maximum_difference == 0.0,
        "the compiled vision rope must be bit-exact with the composed path: {maximum_difference}"
    );
}

// Mirrors the production composed RoPE op sequence (the path the compiled
// graph replaces) so the probe measures old-vs-new dispatch cost directly.
fn composed_vision_rope(
    runtime: &MlxRuntime,
    attention_states: &MlxArray,
    rotary_cosines: &MlxArray,
    rotary_sines: &MlxArray,
    first_half: &MlxArray,
    second_half: &MlxArray,
) -> MlxArray {
    let negative_second_half = runtime
        .negative(second_half)
        .expect("the second half should negate");
    let rotated_states = runtime
        .concatenate_axis(&[&negative_second_half, first_half], 2)
        .expect("the rotated halves should concatenate");
    let cosine_component = runtime
        .multiply(attention_states, rotary_cosines)
        .expect("the cosine component should build");
    let sine_component = runtime
        .multiply(&rotated_states, rotary_sines)
        .expect("the sine component should build");
    let summed_components = runtime
        .add(&cosine_component, &sine_component)
        .expect("the components should add");
    runtime
        .astype(&summed_components, attention_states.dtype())
        .expect("the composed rope should restore the input dtype")
}

fn bf16_array(
    runtime: &MlxRuntime,
    values: &[f32],
    shape: &[i32],
) -> astronomical_runtime_integration::MlxArray {
    runtime
        .astype(
            &runtime
                .array_from_f32(values, shape)
                .expect("the bfloat16 source should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the array should cast to bfloat16")
}

fn f32_array(
    runtime: &MlxRuntime,
    values: &[f32],
    shape: &[i32],
) -> astronomical_runtime_integration::MlxArray {
    runtime
        .array_from_f32(values, shape)
        .expect("the float32 array should be valid")
}

fn patterned_values(element_count: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..element_count)
        .map(|value_index| ((value_index % 23) as f32 - 11.0) * scale + offset)
        .collect()
}

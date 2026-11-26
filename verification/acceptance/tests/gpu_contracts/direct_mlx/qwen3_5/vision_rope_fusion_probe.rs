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

use mlx_bridge::{MlxArray, MlxCompiledElementwiseGraphs, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

use super::vision_rope_operands::VisionRopeOperands;

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

#[expect(clippy::too_many_lines)]
async fn probe_vision_rope_fusion() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the probe memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let compiled_elementwise_graphs =
        MlxCompiledElementwiseGraphs::new().expect("the compiled elementwise graphs should build");

    let operands = VisionRopeOperands::patterned(&runtime, PATCH_COUNT);
    runtime
        .evaluate_arrays(&[
            &operands.attention_states,
            &operands.rotary_cosines,
            &operands.rotary_sines,
        ])
        .expect("the probe operands should evaluate");

    let probes: Vec<(&str, Box<dyn Fn() -> MlxArray + '_>)> = vec![
        (
            "composed-rope",
            Box::new(|| operands.composed_rope(&runtime)),
        ),
        (
            "compiled-rope",
            Box::new(|| {
                runtime
                    .apply_compiled_vision_rope(
                        &compiled_elementwise_graphs,
                        &operands.attention_states,
                        &operands.rotary_cosines,
                        &operands.rotary_sines,
                        &operands.first_half,
                        &operands.second_half,
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

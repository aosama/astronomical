//! Vision rotate-half RoPE fusion probe at scaled vision shapes.
//!
//! The vision tower applies rotate-half RoPE to Q and K in every one of its 27
//! blocks. The composed path dispatches eight small elementwise kernels per
//! call; the compiled graph fuses the elementwise tail into one kernel (the two
//! head-half slices and the rotate-half concat remain separate). This probe
//! scales the patch count to real image workloads and reports the median of
//! interleaved per-iteration timings so machine drift lands on both paths
//! equally.

use std::time::Duration;

use mlx_bridge::{MlxCompiledElementwiseGraphs, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

use super::interleaved_probe_support::{
    NamedProbe, collect_interleaved_samples, sorted_median, warm_every_probe,
};
use super::vision_rope_operands::VisionRopeOperands;

// Qwen3.5 vision processes thousands of patches per image; 8192 keeps the
// elementwise traffic large enough to clear run-to-run noise.
const PATCH_COUNT: i32 = 8_192;

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

    let operands = VisionRopeOperands::patterned(&runtime, PATCH_COUNT);
    runtime
        .evaluate_arrays(&[
            &operands.attention_states,
            &operands.rotary_cosines,
            &operands.rotary_sines,
        ])
        .expect("the probe operands should evaluate");

    let probes = build_probes(&runtime, &operands, &compiled_elementwise_graphs);

    warm_every_probe(&runtime, &probes);
    let mut samples = collect_interleaved_samples(&runtime, &probes);

    report_probe_medians(&probes, &mut samples);
    assert_compiled_rope_is_bit_exact_with_composed(&runtime, &probes);
}

/// The composed path is probe 0 and the compiled path probe 1; the ratio and
/// the bit-exactness check read them by that position.
fn build_probes<'a>(
    runtime: &'a MlxRuntime,
    operands: &'a VisionRopeOperands,
    compiled_elementwise_graphs: &'a MlxCompiledElementwiseGraphs,
) -> Vec<NamedProbe<'a>> {
    vec![
        (
            "composed-rope",
            Box::new(|| operands.composed_rope(runtime)),
        ),
        (
            "compiled-rope",
            Box::new(|| {
                runtime
                    .apply_compiled_vision_rope(
                        compiled_elementwise_graphs,
                        &operands.attention_states,
                        &operands.rotary_cosines,
                        &operands.rotary_sines,
                        &operands.first_half,
                        &operands.second_half,
                    )
                    .expect("the compiled vision rope should build")
            }),
        ),
    ]
}

/// Prints each probe's median and the compiled/composed ratio.
fn report_probe_medians(probes: &[NamedProbe<'_>], samples: &mut [Vec<f64>]) {
    let mut medians = Vec::with_capacity(probes.len());
    for (probe_name, _) in probes {
        let probe_samples = &mut samples[medians.len()];
        let median = sorted_median(probe_samples);
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
}

fn assert_compiled_rope_is_bit_exact_with_composed(
    runtime: &MlxRuntime,
    probes: &[NamedProbe<'_>],
) {
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

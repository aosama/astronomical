//! Vision attention padding-zero cache probe at scaled vision shapes.
//!
//! The vision tower pads Q/K/V head dimensions from 72 to the fused SDPA width
//! in every one of its 27 blocks. The composed path allocates an identical
//! zeros tensor per pad call — 81 zero-tensor allocations and writes per image
//! forward; the retained cache allocates it once per request. This probe times
//! one image forward's padding section both ways with interleaved medians so
//! machine drift lands on both paths equally, and reports the ratio.

use std::time::{Duration, Instant};

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_model_serving::Qwen3_5VisionPaddingZeroCache;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

const HEAD_COUNT: i32 = 16;
const HEAD_DIMENSION: i32 = 72;
const FUSED_HEAD_DIMENSION: i32 = 80;
// Qwen3.5 vision processes thousands of patches per image; 8192 keeps the
// padding traffic large enough to clear run-to-run noise.
const PATCH_COUNT: i32 = 8_192;
const VISION_BLOCK_COUNT: usize = 27;
const PAD_COMPONENT_COUNT: usize = 3;
const PROBE_ITERATIONS: usize = 31;
const PROBE_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
#[ignore = "measures per-call-versus-cached vision padding graphics-processor timing with interleaved medians"]
async fn should_probe_vision_padding_cache_costs_at_scaled_vision_shapes() {
    timeout(PROBE_TIMEOUT, probe_vision_padding_cache())
        .await
        .expect("the vision padding probe must finish within 115 seconds");
}

async fn probe_vision_padding_cache() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the probe memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let padding_zero_cache = Qwen3_5VisionPaddingZeroCache::new();
    // One distinct attention-state tensor per block, as the production tower
    // produces, so the concat inputs cannot alias one another.
    let block_attention_states: Vec<MlxArray> = (0..VISION_BLOCK_COUNT)
        .map(|block_index| patterned_head_array(&runtime, 0.004, -0.37 + block_index as f32 * 0.01))
        .collect();
    runtime
        .evaluate_arrays(&block_attention_states.iter().collect::<Vec<_>>())
        .expect("the probe operands should evaluate");

    let uncached_padding = |block_index: usize| -> MlxArray {
        let attention_states = &block_attention_states[block_index];
        let padding_states = runtime
            .zeros(
                &[
                    1,
                    HEAD_COUNT,
                    PATCH_COUNT,
                    FUSED_HEAD_DIMENSION - HEAD_DIMENSION,
                ],
                MlxDtype::BFloat16,
            )
            .expect("the per-call padding zeros should be valid");
        runtime
            .concatenate_axis(&[attention_states, &padding_states], 3)
            .expect("the per-call padding should concatenate")
    };
    // The cached arm drives the production cache type, whose first call
    // retains the zeros and whose later calls reuse them.
    let cached_padding =
        |block_index: usize, padding_zero_cache: &Qwen3_5VisionPaddingZeroCache| -> MlxArray {
            padding_zero_cache
                .pad_attention_head_dimension(
                    &runtime,
                    &block_attention_states[block_index],
                    HEAD_COUNT,
                    PATCH_COUNT,
                    FUSED_HEAD_DIMENSION,
                )
                .expect("the cached padding should build")
        };

    let probes: Vec<(&str, Box<dyn Fn() -> Vec<MlxArray> + '_>)> = vec![
        (
            "per-call-zeros-padding",
            Box::new(|| {
                let mut outputs = Vec::new();
                for block_index in 0..VISION_BLOCK_COUNT {
                    for _component_index in 0..PAD_COMPONENT_COUNT {
                        outputs.push(uncached_padding(block_index));
                    }
                }
                outputs
            }),
        ),
        (
            "cached-padding",
            Box::new(|| {
                let mut outputs = Vec::new();
                for block_index in 0..VISION_BLOCK_COUNT {
                    for _component_index in 0..PAD_COMPONENT_COUNT {
                        outputs.push(cached_padding(block_index, &padding_zero_cache));
                    }
                }
                outputs
            }),
        ),
    ];

    for (_, probe) in &probes {
        let outputs = probe();
        let output_references = outputs.iter().collect::<Vec<_>>();
        runtime
            .evaluate_arrays(&output_references)
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
            let outputs = probe();
            let output_references = outputs.iter().collect::<Vec<_>>();
            runtime
                .evaluate_arrays(&output_references)
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
            "[vision-padding-probe] {probe_name} median {median:.3} ms samples={}",
            probe_samples.len()
        );
    }
    eprintln!(
        "[vision-padding-probe] cached/per-call ratio = {:.2} (< 1.0 means the retained cache is faster)",
        medians[1] / medians[0]
    );

    // The cached padding must stay bitwise identical to the per-call path.
    let per_call_outputs = probes[0].1();
    let cached_outputs = probes[1].1();
    for (component_index, (per_call_output, cached_output)) in per_call_outputs
        .iter()
        .zip(cached_outputs.iter())
        .enumerate()
    {
        let per_call_values = runtime
            .astype(per_call_output, MlxDtype::Float32)
            .expect("the per-call output should cast")
            .to_vec_f32()
            .expect("the per-call output should evaluate");
        let cached_values = runtime
            .astype(cached_output, MlxDtype::Float32)
            .expect("the cached output should cast")
            .to_vec_f32()
            .expect("the cached output should evaluate");
        let maximum_difference = per_call_values
            .iter()
            .zip(cached_values.iter())
            .map(|(per_call_value, cached_value)| (per_call_value - cached_value).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            maximum_difference == 0.0,
            "the cached padding must be bit-exact with the per-call path at component {component_index}: {maximum_difference}"
        );
    }
}

fn patterned_head_array(runtime: &MlxRuntime, base_step: f32, base_offset: f32) -> MlxArray {
    let shape = [1, HEAD_COUNT, PATCH_COUNT, HEAD_DIMENSION];
    let element_count: usize = shape.iter().map(|dimension| *dimension as usize).product();
    let sample_values: Vec<f32> = (0..element_count)
        .map(|element_index| {
            let phase = (element_index % 97) as f32;
            let magnitude = base_offset + phase * base_step;
            let signed = if element_index % 3 == 0 {
                -magnitude
            } else {
                magnitude
            };
            if element_index % 11 == 0 {
                signed * 41.0
            } else {
                signed
            }
        })
        .collect();
    runtime
        .astype(
            &runtime
                .array_from_f32(&sample_values, &shape)
                .expect("the patterned head array should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the patterned head array should cast to bfloat16")
}

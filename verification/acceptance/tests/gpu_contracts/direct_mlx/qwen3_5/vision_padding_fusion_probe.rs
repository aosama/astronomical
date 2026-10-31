//! Vision attention padding-zero cache probe at scaled vision shapes.
//!
//! The vision tower pads Q/K/V head dimensions from 72 to the fused SDPA width
//! in every one of its 27 blocks. The composed path allocates an identical
//! zeros tensor per pad call — 81 zero-tensor allocations and writes per image
//! forward; the retained cache allocates it once per request. This probe times
//! one image forward's padding section both ways with interleaved medians so
//! machine drift lands on both paths equally, and reports the ratio.

use std::time::{Duration, Instant};

use astronomical_model_serving::Qwen3_5VisionPaddingZeroCache;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::{MlxArray, MlxDtype};
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

/// The two padding strategies under measurement, so the probe arms dispatch
/// through one named type instead of boxed closures.
#[derive(Clone, Copy)]
enum PaddingProbe {
    /// Fresh zeros and a concatenate on every call, as the pre-cache path did.
    PerCallZeros,
    /// The production cache type, whose first call retains the zeros and whose
    /// later calls reuse them.
    Cached,
}

impl PaddingProbe {
    fn name(&self) -> &'static str {
        match self {
            PaddingProbe::PerCallZeros => "per-call-zeros-padding",
            PaddingProbe::Cached => "cached-padding",
        }
    }

    /// Runs one full sweep: every vision block, every padding component.
    fn run(
        self,
        runtime: &MlxRuntime,
        block_attention_states: &[MlxArray],
        padding_zero_cache: &Qwen3_5VisionPaddingZeroCache,
    ) -> Vec<MlxArray> {
        let mut outputs = Vec::new();
        for block_index in 0..VISION_BLOCK_COUNT {
            for _component_index in 0..PAD_COMPONENT_COUNT {
                match self {
                    PaddingProbe::PerCallZeros => outputs.push(uncached_padding(
                        runtime,
                        &block_attention_states[block_index],
                    )),
                    PaddingProbe::Cached => outputs.push(cached_padding(
                        runtime,
                        padding_zero_cache,
                        &block_attention_states[block_index],
                    )),
                }
            }
        }
        outputs
    }
}

/// The measurement order: the per-call arm first, the cached arm second, so
/// the interleaved rounds alternate exactly as the recorded analysis did.
const PROBE_ORDER: [PaddingProbe; 2] = [PaddingProbe::PerCallZeros, PaddingProbe::Cached];

/// Ordering for probe sample times; NaN would mean a broken clock, not a
/// measurement, so it refuses rather than ordering.
fn finite_time_ordering(left: &f64, right: &f64) -> std::cmp::Ordering {
    left.partial_cmp(right)
        .expect("probe sample times should be finite")
}

/// Fresh per-call padding: zeros allocated and concatenated on every call.
fn uncached_padding(runtime: &MlxRuntime, attention_states: &MlxArray) -> MlxArray {
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
}

/// Cached padding through the production cache type.
fn cached_padding(
    runtime: &MlxRuntime,
    padding_zero_cache: &Qwen3_5VisionPaddingZeroCache,
    attention_states: &MlxArray,
) -> MlxArray {
    padding_zero_cache
        .pad_attention_head_dimension(
            runtime,
            attention_states,
            HEAD_COUNT,
            PATCH_COUNT,
            FUSED_HEAD_DIMENSION,
        )
        .expect("the cached padding should build")
}

/// One distinct attention-state tensor per block, as the production tower
/// produces, so the concat inputs cannot alias one another.
fn build_block_attention_states(runtime: &MlxRuntime) -> Vec<MlxArray> {
    let mut block_attention_states = Vec::with_capacity(VISION_BLOCK_COUNT);
    for block_index in 0..VISION_BLOCK_COUNT {
        let pattern_offset = -0.37 + block_index as f32 * 0.01;
        block_attention_states.push(patterned_head_array(runtime, 0.004, pattern_offset));
    }
    runtime
        .evaluate_arrays(&block_attention_states.iter().collect::<Vec<_>>())
        .expect("the probe operands should evaluate");
    block_attention_states
}

/// Runs both probes once and drains the stream, so the measured rounds start
/// from a warm, synchronized state.
fn warm_up_probes(
    runtime: &MlxRuntime,
    block_attention_states: &[MlxArray],
    padding_zero_cache: &Qwen3_5VisionPaddingZeroCache,
) {
    for probe in PROBE_ORDER {
        let outputs = probe.run(runtime, block_attention_states, padding_zero_cache);
        let output_references = outputs.iter().collect::<Vec<_>>();
        runtime
            .evaluate_arrays(&output_references)
            .expect("the warmup probe should evaluate");
    }
    runtime
        .synchronize_gpu_stream()
        .expect("the warmup probes should drain");
}

/// Interleaves the two probes until both a 1.5-second ramp and the minimum
/// iteration count are satisfied (with a hard four-times ceiling), timing each
/// round to a fully evaluated output.
fn collect_probe_samples(
    runtime: &MlxRuntime,
    block_attention_states: &[MlxArray],
    padding_zero_cache: &Qwen3_5VisionPaddingZeroCache,
) -> Vec<Vec<f64>> {
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); PROBE_ORDER.len()];
    let ramp_started_at = Instant::now();
    let mut round_index = 0;
    while ramp_started_at.elapsed() < Duration::from_millis(1500) || round_index < PROBE_ITERATIONS
    {
        for (probe_index, probe) in PROBE_ORDER.iter().enumerate() {
            let started_at = Instant::now();
            let outputs = probe.run(runtime, block_attention_states, padding_zero_cache);
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
    samples
}

/// Reduces the samples to per-probe medians and prints them plus the
/// cached/per-call ratio.
fn report_probe_medians(samples: &[Vec<f64>]) -> Vec<f64> {
    let mut medians = Vec::with_capacity(PROBE_ORDER.len());
    for (probe_index, probe) in PROBE_ORDER.iter().enumerate() {
        let probe_samples = &mut samples[probe_index].clone();
        probe_samples.sort_by(finite_time_ordering);
        let median = probe_samples[probe_samples.len() / 2];
        medians.push(median);
        eprintln!(
            "[vision-padding-probe] {} median {median:.3} ms samples={}",
            probe.name(),
            probe_samples.len()
        );
    }
    eprintln!(
        "[vision-padding-probe] cached/per-call ratio = {:.2} (< 1.0 means the retained cache is faster)",
        medians[1] / medians[0]
    );
    medians
}

/// The cached padding must stay bitwise identical to the per-call path.
fn assert_cached_padding_bit_exact(
    runtime: &MlxRuntime,
    per_call_outputs: &[MlxArray],
    cached_outputs: &[MlxArray],
) {
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
        let mut maximum_difference = 0.0_f32;
        for (per_call_value, cached_value) in per_call_values.iter().zip(cached_values.iter()) {
            maximum_difference = maximum_difference.max((per_call_value - cached_value).abs());
        }
        assert!(
            maximum_difference == 0.0,
            "the cached padding must be bit-exact with the per-call path at component {component_index}: {maximum_difference}"
        );
    }
}

async fn probe_vision_padding_cache() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(16 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .expect("the probe memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let padding_zero_cache = Qwen3_5VisionPaddingZeroCache::new();
    let block_attention_states = build_block_attention_states(&runtime);
    warm_up_probes(&runtime, &block_attention_states, &padding_zero_cache);

    let samples = collect_probe_samples(&runtime, &block_attention_states, &padding_zero_cache);
    report_probe_medians(&samples);

    let per_call_outputs =
        PaddingProbe::PerCallZeros.run(&runtime, &block_attention_states, &padding_zero_cache);
    let cached_outputs =
        PaddingProbe::Cached.run(&runtime, &block_attention_states, &padding_zero_cache);
    assert_cached_padding_bit_exact(&runtime, &per_call_outputs, &cached_outputs);
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

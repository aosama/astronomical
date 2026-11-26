//! The interleaved-median timing loop shared by the elementwise and vision RoPE
//! fusion probes, so both measure with the very same warmup, round structure
//! and median rule. The restore reconstruction probe shares the probe type, the
//! warmup and the median rule, and keeps its own fixed-round timing loop.

use std::time::{Duration, Instant};

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;

const PROBE_ITERATIONS: usize = 31;

/// One graph built and dispatched per call, so every timed round rebuilds it.
pub(super) type ProbeGraphBuilder<'a> = Box<dyn Fn() -> MlxArray + 'a>;

/// A probe's report name paired with the graph it builds.
pub(super) type NamedProbe<'a> = (&'static str, ProbeGraphBuilder<'a>);

/// Warms every probe once, outside any timed region, then drains the stream so
/// the measured rounds start from a synchronized state.
pub(super) fn warm_every_probe(runtime: &MlxRuntime, probes: &[NamedProbe<'_>]) {
    for (_, probe) in probes {
        let output = probe();
        runtime
            .evaluate_arrays(&[&output])
            .expect("the warmup probe should evaluate");
    }
    runtime
        .synchronize_gpu_stream()
        .expect("the warmup probes should drain");
}

/// Interleaves the probes (one iteration of each per round, so machine drift
/// lands on all probes equally) until both a 1.5-second ramp and the minimum
/// iteration count are satisfied, with a hard four-times ceiling. Each sample
/// is the millisecond time to a fully evaluated output.
pub(super) fn collect_interleaved_samples(
    runtime: &MlxRuntime,
    probes: &[NamedProbe<'_>],
) -> Vec<Vec<f64>> {
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); probes.len()];
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
    samples
}

/// Sorts one probe's samples in place and returns the median, which kills
/// outliers that a mean would carry.
pub(super) fn sorted_median(probe_samples: &mut [f64]) -> f64 {
    probe_samples.sort_by(|left, right| left.partial_cmp(right).expect("finite times"));
    probe_samples[probe_samples.len() / 2]
}

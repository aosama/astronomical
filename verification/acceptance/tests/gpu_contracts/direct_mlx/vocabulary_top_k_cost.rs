//! What finding the best tokens of a vocabulary-wide row costs, beside the
//! time reading the row once takes.
//!
//! A draft round finds a few candidate tokens in every drafted row of the
//! drafter's logits (248,320 wide on the Qwen 3.5 family), and a sampled
//! verify finds the target's top-k in every tree row. On a 35B-A3B at a
//! 16 GB limit the drafter's head and candidates measured about 2.6 ms of a
//! 7.6 ms draft, where the head's weights need about 0.9 ms. This prices
//! `vocabulary_top_k` alone at the row counts a round uses.
//!
//! Random bfloat16 logits; no model on disk. Run with
//! `cargo measure-vocabulary-top-k-cost`. Seconds.

use std::time::Instant;

use mlx_bridge::{MlxArray, MlxDtype};
use model_serving::gpu_token_sampling::vocabulary_top_k;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

const VOCABULARY_WIDTH: i32 = 248_320;
const KEPT_TOKEN_COUNT: i32 = 16;
const CALLS_A_PASS: usize = 50;
const TIMED_PASS_COUNT: usize = 11;
/// Bytes a second the GPU reads in a large one-row multiply on this
/// machine (`cargo measure-primitive-dispatch-cost`).
const MEASURED_BYTES_A_SECOND: f64 = 290.0e9;

fn top_k_runtime() -> MlxRuntime {
    let two_gigabytes = 2 * 1024 * 1024 * 1024;
    let memory_limits = MlxMemoryLimits::new(two_gigabytes, two_gigabytes / 4);
    MlxRuntime::initialize(memory_limits.expect("the limits are valid")).expect("MLX starts")
}

fn random_logits(runtime: &MlxRuntime, row_count: i32) -> MlxArray {
    let random_key = runtime.random_key(2_569).expect("a random key");
    let logits = runtime
        .random_normal(
            &[1, row_count, VOCABULARY_WIDTH],
            MlxDtype::BFloat16,
            0.0,
            3.0,
            &random_key,
        )
        .expect("the logits are drawn");
    runtime
        .evaluate_arrays(&[&logits])
        .expect("the logits evaluate");
    logits
}

/// Seconds for `call_count` independent top-k calls in one evaluation.
fn top_k_pass_seconds(runtime: &MlxRuntime, logits: &MlxArray, call_count: usize) -> f64 {
    let mut kept_ids = Vec::with_capacity(call_count);
    for _ in 0..call_count {
        kept_ids.push(vocabulary_top_k(runtime, logits, KEPT_TOKEN_COUNT).expect("top-k builds"));
    }
    let kept_id_references: Vec<&MlxArray> = kept_ids.iter().collect();
    let started_at = Instant::now();
    runtime
        .evaluate_arrays(&kept_id_references)
        .expect("top-k evaluates");
    started_at.elapsed().as_secs_f64()
}

/// The median of the timed passes, in microseconds a call, after one
/// untimed pass that compiles the kernel.
fn median_microseconds_a_call(runtime: &MlxRuntime, logits: &MlxArray, call_count: usize) -> f64 {
    top_k_pass_seconds(runtime, logits, call_count);
    let mut pass_seconds = Vec::with_capacity(TIMED_PASS_COUNT);
    for _ in 0..TIMED_PASS_COUNT {
        pass_seconds.push(top_k_pass_seconds(runtime, logits, call_count));
    }
    pass_seconds.sort_by(f64::total_cmp);
    pass_seconds[TIMED_PASS_COUNT / 2] * 1e6 / call_count as f64
}

#[test]
#[ignore = "prices the vocabulary top-k kernel on the local GPU"]
#[serial]
fn should_price_the_vocabulary_top_k_at_the_row_counts_a_round_uses() {
    let runtime = top_k_runtime();
    for row_count in [1, 3, 4, 16] {
        let logits = random_logits(&runtime, row_count);
        let overlapped_microseconds = median_microseconds_a_call(&runtime, &logits, CALLS_A_PASS);
        // A round makes one call and waits for it: its latency, not its
        // throughput, is what a round pays.
        let alone_microseconds = median_microseconds_a_call(&runtime, &logits, 1);
        let floor_microseconds = logits.byte_count() as f64 / MEASURED_BYTES_A_SECOND * 1e6;
        eprintln!(
            "[vocabulary-top-k-cost] rows={row_count} kept={KEPT_TOKEN_COUNT} microseconds_one_call_alone={alone_microseconds:.1} microseconds_a_call_overlapped={overlapped_microseconds:.1} read_once_floor_microseconds={floor_microseconds:.1}"
        );
        assert!(alone_microseconds > 0.0, "a top-k call must take time");
    }
}

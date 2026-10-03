//! Measures the serving throughput of the resident 35B sparse-MoE model over the
//! supervisor IPC boundary: prompt-processing (prefill) tokens per second and
//! decode tokens per second, both server-attributed from the worker's generation
//! performance log (never client wall clock).
//!
//! The model is looked up by the stable `resident_sparse_moe` role from
//! registry/e2e_test_model_names.json instead of a hardcoded name, so this keeps
//! measuring the correct leaf when the retargeted artifact changes.
//!
//! The test case follows AGENTS.md "Instructions for Performance Throughput
//! Tests":
//! - Send at least 10,000 input tokens. The input is the continuation
//!   instruction plus the ~10,000-token Romeo and Juliet source fixture, which
//!   together tokenize to ~10,500 tokens.
//! - Acquire ~1,000 output tokens (±10% is acceptable). Each measured
//!   completion is capped at 1,000 output tokens.
//! - The SSD (persistent prompt) cache is disabled in the worker's resolved
//!   configuration, so every completion prefills the full prompt from scratch.
//!
//! Method: a short warmup completion (~1,000 input tokens, ~100 output tokens)
//! is discarded so first-use JIT kernel compilation never inflates the measured
//! prefill, then the single measured completion over the full prompt reports the
//! server-attributed rates persisted to the durable historical record. The
//! journey asserts nothing about the measured rates — it is a measurement, not
//! a threshold check.
//!
//! This is a laptop-only journey: it loads real weights into wired GPU memory,
//! so it is `#[ignore]`d and not wired into CI. Invoke it through
//! scripts/run-performance-throughput.sh (invocation only) or directly with
//! `cargo test --release -p astronomical-inference-worker --features
//! astronomical-inference-worker/performance_throughput --test
//! performance_throughput_tests -- --ignored --exact`.

use crate::performance_throughput::historical_record::ThroughputJourneyKind;
use crate::performance_throughput::support::{ThroughputJourney, run_journey_with_timeout};
use crate::support::resident_sparse_moe_model_id;

/// The short warmup: a ~1,000-token Romeo and Juliet opening, continued for a
/// short passage, that spins up first-use JIT kernels before the measured run.
const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";

/// The ~1,000-token Romeo and Juliet opening used as the warmup input.
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_warmup_romeo_and_juliet.txt");

/// The warmup output cap: ~100 output tokens.
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 100;

/// The continuation instruction prepended to the full Romeo and Juliet source.
const MEASURED_INPUT_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 1000 words.";

/// The ~10,000-token Romeo and Juliet source; with the instruction the full
/// input is ~10,500 tokens, clearing the >=10,000-token input requirement.
const MEASURED_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_10000_tokens_romeo_and_juliet.txt");

/// The measured output cap: ~1,000 output tokens (±10% acceptable).
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 1_000;

/// The sampling temperature in thousandths (1_000 = 1.0, unclamped).
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;

/// Measures the resident sparse-MoE model's serving prompt-processing and
/// decode throughput: after a short ~1,000-token warmup, a >=10,000-token Romeo
/// and Juliet input drives ~1,000 output tokens of generation with the SSD
/// (persistent prompt) cache disabled, and the measured completion's
/// server-attributed rates are persisted to the durable historical record.
#[test]
#[ignore = "loads the resident 35B sparse-MoE model and measures serving prompt-processing and decode throughput over IPC"]
fn should_measure_resident_sparse_moe_prompt_processing_and_decode_throughput() {
    let journey = ThroughputJourney {
        journey_kind: ThroughputJourneyKind::Text,
        warmup_input_prompt: format!(
            "{WARMUP_INPUT_INSTRUCTION}\n\n{WARMUP_ROMEO_AND_JULIET_SOURCE}"
        ),
        warmup_images: Vec::new(),
        warmup_output_tokens: WARMUP_MAXIMUM_OUTPUT_TOKENS,
        measured_input_prompt: format!(
            "{MEASURED_INPUT_INSTRUCTION}\n\n{MEASURED_ROMEO_AND_JULIET_SOURCE}"
        ),
        measured_images: Vec::new(),
        measured_output_tokens: MEASURED_MAXIMUM_OUTPUT_TOKENS,
        temperature_thousandths: TEMPERATURE_THOUSANDTHS,
    };
    run_journey_with_timeout(resident_sparse_moe_model_id(), journey);
}

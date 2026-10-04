//! Measures the serving throughput of the canonical `dense_mtp` model — the
//! 27B dense multi-token-prediction artifact — over the supervisor IPC
//! boundary: prompt-processing (prefill) tokens per second and decode tokens
//! per second, both server-attributed from the worker's generation
//! performance log (never client wall clock).
//!
//! This journey is the MTP baseline surface. The durable record carries an
//! `mtp_draft_depth` field: the baseline lines recorded here run with MTP
//! disabled (the resolved configuration's default), and the MTP-engaged
//! comparisons this surface exists for will record their draft depth in the
//! same field, so same-model lines separate cleanly in the shared history
//! log. MTP claims — the upstream reports of large decode speedups at draft
//! depths 1 through 3 — are what this surface must verify on this codebase,
//! never assume.
//!
//! The test case follows AGENTS.md "Instructions for Performance Throughput
//! Tests" with one documented exception: this is a 27B **dense** model, so
//! every token runs all 27B parameters (the sparse-MoE siblings run ~3B).
//! That makes both prefill and decode far slower, and the standard
//! ~10,000-input/~1,000-output case cannot finish inside the 115s journey
//! deadline. This surface therefore measures a ~5,000-token Romeo and Juliet
//! input driving ~500 output tokens (still a long-prefill / long-decode
//! shape, just at a budget that fits). A short warmup completion
//! (~1,000 input tokens, ~100 output tokens) is discarded so first-use JIT
//! kernel compilation never inflates the measured prefill, and the SSD
//! (persistent prompt) cache is disabled in the worker's resolved
//! configuration. The journey asserts nothing about the measured rates.
//!
//! This is a laptop-only journey: it loads real weights into wired GPU
//! memory, so it is `#[ignore]`d and not wired into CI. Invoke it through
//! scripts/run-performance-throughput.sh (invocation only) or directly with
//! `cargo test --release -p astronomical-inference-worker --features
//! astronomical-inference-worker/performance_throughput --test
//! performance_throughput_tests -- --ignored --exact`.

use crate::performance_throughput::historical_record::ThroughputJourneyKind;
use crate::performance_throughput::support::{ThroughputJourney, run_journey_with_timeout};
use crate::support::dense_mtp_model_id;

/// The short warmup: a ~1,000-token Romeo and Juliet opening, continued for a
/// short passage, that spins up first-use JIT kernels before the measured run.
const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";

/// The ~1,000-token Romeo and Juliet opening used as the warmup input.
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_warmup_romeo_and_juliet.txt");

/// The warmup output cap: ~100 output tokens.
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 100;

/// The continuation instruction prepended to the Romeo and Juliet source.
const MEASURED_INPUT_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 500 words.";

/// The ~5,000-token Romeo and Juliet source; with the instruction the full
/// input is ~5,050 tokens. This is the documented dense-model exception to the
/// standard ~10,000-token input, sized so the compute-heavy 27B dense model
/// finishes the measured completion inside the journey deadline.
const MEASURED_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_tokens_romeo_and_juliet.txt");

/// The measured output cap: ~500 output tokens (the dense-model exception to
/// the standard ~1,000-output case; ±10% acceptable).
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 500;

/// The sampling temperature in thousandths (1_000 = 1.0, unclamped).
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;

/// Builds the one dense-model journey case; only the MTP draft depth varies, so
/// every durable history line differs from the baseline in that field alone.
fn dense_mtp_journey(mtp_draft_depth: Option<u8>) -> ThroughputJourney {
    ThroughputJourney {
        journey_kind: ThroughputJourneyKind::Text,
        mtp_draft_depth,
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
    }
}

/// MTP-off baseline: after a short ~1,000-token warmup, a ~5,000-token Romeo and
/// Juliet input (the dense-model exception to the standard ~10,000-token case)
/// drives ~500 output tokens with the SSD (persistent prompt) cache disabled.
#[test]
#[ignore = "loads the canonical dense MTP model and measures its MTP-off serving throughput over IPC"]
fn should_measure_dense_mtp_prompt_processing_and_decode_throughput_with_mtp_disabled() {
    run_journey_with_timeout(dense_mtp_model_id(), dense_mtp_journey(None));
}

/// Same case with multi-token prediction engaged at draft depth 1, the regime
/// the published reference runner supports. Upstream claims of large decode
/// speedups are verified here, never assumed.
#[test]
#[ignore = "loads the canonical dense MTP model and measures its draft-depth-1 serving throughput over IPC"]
fn should_measure_dense_mtp_prompt_processing_and_decode_throughput_with_mtp_draft_depth_one() {
    run_journey_with_timeout(dense_mtp_model_id(), dense_mtp_journey(Some(1)));
}

/// Same case at draft depth 2.
#[test]
#[ignore = "loads the canonical dense MTP model and measures its draft-depth-2 serving throughput over IPC"]
fn should_measure_dense_mtp_prompt_processing_and_decode_throughput_with_mtp_draft_depth_two() {
    run_journey_with_timeout(dense_mtp_model_id(), dense_mtp_journey(Some(2)));
}

/// Same case at draft depth 3, the artifact's own default depth.
#[test]
#[ignore = "loads the canonical dense MTP model and measures its draft-depth-3 serving throughput over IPC"]
fn should_measure_dense_mtp_prompt_processing_and_decode_throughput_with_mtp_draft_depth_three() {
    run_journey_with_timeout(dense_mtp_model_id(), dense_mtp_journey(Some(3)));
}

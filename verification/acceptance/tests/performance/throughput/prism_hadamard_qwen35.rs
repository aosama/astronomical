//! Tier 5: serving throughput of the folded ternary pack over the daemon IPC
//! boundary, server-attributed from the worker's generation performance log.
//!
//! The test case follows AGENTS.md "Instructions for Performance Throughput
//! Tests": a warmup completion of roughly 500 prompt tokens and 50 output
//! tokens is discarded, then one measured completion sends at least 10,000
//! Romeo and Juliet prompt tokens and acquires 500 output tokens with the
//! astro SSD prompt cache disabled and the prefill chunk fixed at 2,048
//! tokens (both pinned by the worker configuration this driver builds).
//!
//! The pack is located through the explicit test-run `model_directory` (the Hugging
//! Face snapshot directory), never a hardcoded developer path, and the
//! worker discovers it through the snapshot's `models--` entry root. Discovery
//! names a Hugging Face hub cache entry by the leaf of its decoded `org/repo` identity, so the
//! requested model id is derived from that directory name rather than typed.

use std::path::PathBuf;
use std::time::Duration;

use serial_test::serial;

use crate::performance_throughput::historical_record::ThroughputJourneyKind;
use crate::performance_throughput::support::{self as throughput_support, ThroughputJourney};

const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";
const MEASURED_INPUT_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 500 tokens.";
const WARMUP_MINIMUM_INPUT_TOKENS: usize = 500;
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 50;
const MEASURED_MINIMUM_INPUT_TOKENS: usize = 10_000;
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 500;
/// The sampling temperature in thousandths (1_000 = 1.0), matching the other
/// serving throughput journeys.
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;
/// A dense 27B pack processes a 10,000-token prompt in tens of seconds and
/// decodes 500 tokens in tens more, after a multi-gigabyte load; the budget
/// bounds the whole journey (launch, warmup, measured completion).
const JOURNEY_TIMEOUT: Duration = Duration::from_secs(600);

fn pack_directory() -> PathBuf {
    astronomical_test_harness::test_run::TestRunContext::current().required_model_directory()
}

/// Measures the folded ternary pack's serving prompt-processing and decode
/// throughput over IPC: after a short warmup, a >=10,000-token Romeo and
/// Juliet input drives 500 output tokens with the astro SSD prompt cache disabled, and the
/// measured completion's server-attributed rates are persisted to the
/// durable historical record.
#[test]
#[ignore = "loads the real roughly 8 GB folded ternary pack and measures serving prompt-processing and decode throughput over IPC"]
#[serial]
fn should_measure_folded_ternary_prompt_processing_and_decode_throughput() {
    let mut journey = ThroughputJourney::production_default(
        ThroughputJourneyKind::Text,
        throughput_support::romeo_and_juliet_prompt_of_at_least(
            &pack_directory(),
            WARMUP_INPUT_INSTRUCTION,
            WARMUP_MINIMUM_INPUT_TOKENS,
        ),
        Vec::new(),
        WARMUP_MAXIMUM_OUTPUT_TOKENS,
        throughput_support::romeo_and_juliet_prompt_of_at_least(
            &pack_directory(),
            MEASURED_INPUT_INSTRUCTION,
            MEASURED_MINIMUM_INPUT_TOKENS,
        ),
        Vec::new(),
        MEASURED_MAXIMUM_OUTPUT_TOKENS,
        TEMPERATURE_THOUSANDTHS,
    );
    let pack_directory = pack_directory();
    let model_id: &'static str =
        Box::leak(throughput_support::discovered_model_id(&pack_directory).into_boxed_str());
    journey.model_directory = Some(pack_directory);
    journey.journey_timeout = JOURNEY_TIMEOUT;
    let measured_throughput = throughput_support::run_journey_with_timeout(model_id, journey);
    eprintln!(
        "[prism-hadamard-perf:tier=serving] prompt_tokens={} generated_tokens={} prefill_tokens_per_second={:.1} decode_tokens_per_second={:.2}",
        measured_throughput.total_input_tokens,
        measured_throughput.total_output_tokens,
        measured_throughput.prefill_tokens_per_second,
        measured_throughput.decode_tokens_per_second,
    );
    throughput_support::assert_measurement_shape(&measured_throughput);
}

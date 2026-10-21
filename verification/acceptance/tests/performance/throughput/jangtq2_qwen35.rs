//! Tier 5: serving throughput of the jangtq2 codebook weight family (the
//! 122B hybrid sparse-MoE bundle) over the daemon IPC boundary,
//! server-attributed from the worker's generation performance log.
//!
//! The test case follows AGENTS.md "Instructions for Performance Throughput
//! Tests": a warmup completion of roughly 500 prompt tokens and 50 output
//! tokens is discarded, then one measured completion sends at least 10,000
//! Romeo and Juliet prompt tokens and acquires 500 output tokens with the SSD
//! (persistent prompt) cache disabled and the prefill chunk fixed at 2,048
//! tokens (both pinned by the worker environment this driver builds).
//!
//! The bundle is located through `ASTRONOMICAL_JANGTQ2_MODEL_DIR` (the
//! installed bundle directory, named by its model id), never a hardcoded
//! developer path — the same variable the real-bundle plan contracts use.
//! The worker discovers it through that directory, and the requested model id
//! is the directory's leaf name.
//!
//! The memory ceiling is the throughput surface's shared real-model cap
//! (`MAXIMUM_MODEL_MEMORY_BYTES`), the same cap the resident 35B journey
//! records under, so rates in the historical record compare like for like.
//!
//! This is a laptop-only journey: it streams real expert weights from the
//! solid-state drive, so it is `#[ignore]`d and not wired into CI. Invoke it
//! with `cargo test-jangtq2-serving-throughput` (see `.cargo/config.toml`)
//! after building the release worker.

use std::path::PathBuf;
use std::time::Duration;

use serial_test::serial;

use crate::performance_throughput::historical_record::ThroughputJourneyKind;
use crate::performance_throughput::support::{self as throughput_support, ThroughputJourney};

/// The environment variable pointing at the installed bundle directory.
const MODEL_DIRECTORY_ENV: &str = "ASTRONOMICAL_JANGTQ2_MODEL_DIR";
const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";
/// The Romeo and Juliet source used for the bounded warmup prompt.
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_warmup_romeo_and_juliet.txt");
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 50;
const MEASURED_INPUT_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 500 tokens.";
/// The ~10,000-token Romeo and Juliet source under the Qwen tokenizer; with
/// the instruction the input clears the >=10,000-token requirement.
const MEASURED_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_10000_tokens_romeo_and_juliet.txt");
/// The measured output cap: 500 output tokens (±10% acceptable).
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 500;
/// The sampling temperature in thousandths (1_000 = 1.0), matching the other
/// serving throughput journeys.
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;
/// The bundle streams about 32 GB of packed experts from the solid-state
/// drive during the first prefill and decodes 500 tokens afterwards; the
/// budget bounds the whole journey (launch, warmup, measured completion) and
/// is deliberately wide so a slow run reports a rate instead of a timeout.
const JOURNEY_TIMEOUT: Duration = Duration::from_secs(3_600);

fn model_directory() -> PathBuf {
    match std::env::var_os(MODEL_DIRECTORY_ENV) {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => panic!(
            "set {MODEL_DIRECTORY_ENV} to the installed jangtq2 bundle directory \
             (containing config.json, jang_config.json, and the safetensors shards)"
        ),
    }
}

/// The model id discovery derives for the bundle: the directory's leaf name.
fn discovered_model_id(model_directory: &std::path::Path) -> String {
    model_directory
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| {
            panic!("{MODEL_DIRECTORY_ENV} must name the bundle directory by its model id")
        })
        .to_owned()
}

/// Measures the jangtq2 bundle's serving prompt-processing and decode
/// throughput over IPC: after a short warmup, a >=10,000-token Romeo and
/// Juliet input drives 500 output tokens with the SSD cache disabled, and the
/// measured completion's server-attributed rates are persisted to the
/// durable historical record.
#[test]
#[ignore = "streams the real 122B jangtq2 bundle from the solid-state drive and measures serving prompt-processing and decode throughput over IPC"]
#[serial]
fn should_measure_jangtq2_prompt_processing_and_decode_throughput() {
    let mut journey = ThroughputJourney::production_default(
        ThroughputJourneyKind::Text,
        throughput_support::romeo_and_juliet_warmup_prompt(
            WARMUP_INPUT_INSTRUCTION,
            WARMUP_ROMEO_AND_JULIET_SOURCE,
        ),
        Vec::new(),
        WARMUP_MAXIMUM_OUTPUT_TOKENS,
        format!("{MEASURED_INPUT_INSTRUCTION}\n\n{MEASURED_ROMEO_AND_JULIET_SOURCE}"),
        Vec::new(),
        MEASURED_MAXIMUM_OUTPUT_TOKENS,
        TEMPERATURE_THOUSANDTHS,
    );
    let model_directory = model_directory();
    let model_id: &'static str = Box::leak(discovered_model_id(&model_directory).into_boxed_str());
    journey.model_directory = Some(model_directory);
    journey.maximum_mlx_memory_bytes = Some(throughput_support::MAXIMUM_MODEL_MEMORY_BYTES);
    journey.journey_timeout = JOURNEY_TIMEOUT;
    let measured_throughput = throughput_support::run_journey_with_timeout(model_id, journey);
    eprintln!(
        "[jangtq2-perf:tier=serving] prompt_tokens={} generated_tokens={} prefill_tokens_per_second={:.1} decode_tokens_per_second={:.2}",
        measured_throughput.total_input_tokens,
        measured_throughput.total_output_tokens,
        measured_throughput.prefill_tokens_per_second,
        measured_throughput.decode_tokens_per_second,
    );
    throughput_support::assert_measurement_shape(&measured_throughput);
}

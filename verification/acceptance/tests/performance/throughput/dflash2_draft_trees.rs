//! Serving throughput of a target with its DFlash2 drafter, over the daemon
//! IPC boundary: the real worker finds the drafter, converts the target's
//! projections into register order and decodes admitted sessions in
//! draft-tree rounds.
//!
//! The cell follows AGENTS.md "Performance Instructions": a warmup of
//! roughly 500 prompt tokens and 50 output tokens is discarded, then one
//! measured completion sends at least 10,000 Romeo and Juliet prompt tokens
//! and acquires 500 output tokens at temperature 1.0 with the
//! `astro_ssd_prompt_cache` disabled.
//!
//! The target is located through the explicit test-run `model_directory`,
//! never a hardcoded developer path. Its drafter is the one the engine's
//! discovery pairs with it: the `drafter` folder inside the target's own
//! folder, or a sibling model folder in the same `huggingface_hub_cache` or
//! plain library. Without a paired drafter the same journey measures plain
//! decoding, so running it on a target and on the same model without its
//! drafter is the serving A/B. On the M5-family GPU these were measured on,
//! the dense 27B decoded about 17 tok/s plain against about 36 with draft
//! trees, and the Ornith 1.5 35B-A3B OptiQ 4-bit 67.7 against 81.3, so the
//! decode rate itself says which path ran.

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
/// Chat's real temperature, in thousandths.
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;
/// The load converts roughly 15 GB of projections once, the prompt takes
/// about half a minute and the reply a quarter; the budget bounds the whole
/// journey with room for a cold file read.
const JOURNEY_TIMEOUT: Duration = Duration::from_secs(600);

fn target_directory() -> PathBuf {
    test_harness::test_run::TestRunContext::current().required_model_directory()
}

/// The memory limit the worker serves under: the machine's default, or the
/// test run's `paging_experiment.maximum_mlx_memory_gigabytes` when the run
/// sets one other than that field's own default. A limit below the model's
/// size is how serving is measured while the model streams from storage.
fn configured_memory_limit_bytes() -> Option<u64> {
    let paging_experiment = test_harness::test_run::TestRunContext::current()
        .configuration
        .paging_experiment;
    if paging_experiment.maximum_mlx_memory_gigabytes
        == test_harness::test_run::PagingExperimentCase::default().maximum_mlx_memory_gigabytes
    {
        return None;
    }
    Some(
        paging_experiment
            .maximum_mlx_memory_bytes()
            .expect("the configured memory limit is valid"),
    )
}

#[test]
#[ignore = "loads a real target and its DFlash2 drafter and measures serving throughput over IPC"]
#[serial]
fn should_measure_dflash2_draft_tree_prompt_processing_and_decode_throughput() {
    let target_directory = target_directory();
    let mut journey = ThroughputJourney::production_default(
        ThroughputJourneyKind::Text,
        throughput_support::romeo_and_juliet_prompt_of_at_least(
            &target_directory,
            WARMUP_INPUT_INSTRUCTION,
            WARMUP_MINIMUM_INPUT_TOKENS,
        ),
        Vec::new(),
        WARMUP_MAXIMUM_OUTPUT_TOKENS,
        throughput_support::romeo_and_juliet_prompt_of_at_least(
            &target_directory,
            MEASURED_INPUT_INSTRUCTION,
            MEASURED_MINIMUM_INPUT_TOKENS,
        ),
        Vec::new(),
        MEASURED_MAXIMUM_OUTPUT_TOKENS,
        TEMPERATURE_THOUSANDTHS,
    );
    let model_id: &'static str =
        Box::leak(throughput_support::discovered_model_id(&target_directory).into_boxed_str());
    journey.model_directory = Some(target_directory);
    journey.maximum_mlx_memory_bytes = configured_memory_limit_bytes();
    journey.journey_timeout = JOURNEY_TIMEOUT;
    let measured_throughput = throughput_support::run_journey_with_timeout(model_id, journey);
    eprintln!(
        "[dflash2-serving] prompt_tokens={} generated_tokens={} prefill_tokens_per_second={:.1} decode_tokens_per_second={:.2}",
        measured_throughput.total_input_tokens,
        measured_throughput.total_output_tokens,
        measured_throughput.prefill_tokens_per_second,
        measured_throughput.decode_tokens_per_second,
    );
    throughput_support::assert_measurement_shape(&measured_throughput);
}

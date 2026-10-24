//! Shared pack, runtime, tokenizer, prompt, and timing helpers for the folded
//! ternary journeys and their performance tiers.
//!
//! Every tier loads the real installed pack through the explicit test-run configuration
//! (never a hardcoded developer path), feeds Romeo and Juliet source text
//! through the pack's own tokenizer, fixes the prompt-processing chunk at
//! 2,048 tokens, and reports medians over repeated evaluated runs.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use astronomical_model_serving::prism_hadamard_qwen35::artifacts::FoldedPackValidator;
use astronomical_model_serving::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::MlxArray;

/// Wired and allocator limits for one roughly 8 GB pack plus activations and
/// a 10,000-token key-value prefix, far under the wired ceiling of a
/// maximum-memory laptop.
const JOURNEY_WIRED_LIMIT_BYTES: usize = 24 * 1024 * 1024 * 1024;

/// The fixed prompt-processing chunk every tier uses, so prefill numbers are
/// reproducible across runs and comparable with the serving configuration.
pub(crate) const PREFILL_CHUNK_TOKENS: usize = 2_048;
/// The warmup completion: roughly 500 prompt tokens and 50 generated tokens
/// absorb first-use kernel compilation before anything is measured.
pub(crate) const WARMUP_INPUT_TOKENS: usize = 500;
pub(crate) const WARMUP_OUTPUT_TOKENS: usize = 50;
/// The measured completion: at least 10,000 prompt tokens and 500 generated
/// tokens, both within ten percent.
pub(crate) const MEASURED_INPUT_TOKENS: usize = 10_000;
pub(crate) const MEASURED_OUTPUT_TOKENS: usize = 500;
const MINIMUM_ACCEPTABLE_INPUT_TOKENS: usize = 9_000;
const MAXIMUM_ACCEPTABLE_INPUT_TOKENS: usize = 11_000;
const MINIMUM_ACCEPTABLE_OUTPUT_TOKENS: usize = 450;
const MAXIMUM_ACCEPTABLE_OUTPUT_TOKENS: usize = 550;

/// The Romeo and Juliet source every prompt is cut from, word by word.
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt");
/// The continuation instruction prepended to the source excerpt.
const CONTINUATION_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 500 tokens.";

/// Resolves the installed pack root, refusing to guess a developer path.
pub(crate) fn pack_directory() -> PathBuf {
    astronomical_test_harness::test_run::TestRunContext::current().required_model_directory()
}

/// Resolves the upstream reference dump for the parity journey.
pub(crate) fn parity_reference_path() -> PathBuf {
    let reference_path = astronomical_test_harness::test_run::TestRunContext::current()
        .required_reference_artifact_path();
    assert!(
        reference_path.is_file(),
        "the parity reference artifact must be a file"
    );
    reference_path
}

/// The process-wide journey runtime, sized for the whole-pack residency.
pub(crate) fn shared_journey_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(JOURNEY_WIRED_LIMIT_BYTES, JOURNEY_WIRED_LIMIT_BYTES)
            .expect("the journey memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Loads the real pack into a ready-to-run family text model and reports the
/// load time.
pub(crate) fn loaded_journey_model() -> PrismHadamardTextModel {
    let load_started_at = Instant::now();
    let pack = FoldedPackValidator::new()
        .validate(&pack_directory(), 8192)
        .expect("the installed folded pack should validate");
    let model = PrismHadamardTextModel::new(shared_journey_runtime(), pack)
        .expect("the installed folded pack should load into the runtime");
    eprintln!(
        "[prism-hadamard-perf] model_load_seconds={:.2}",
        load_started_at.elapsed().as_secs_f64()
    );
    model
}

/// The pack's own tokenizer.
pub(crate) fn pack_tokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(pack_directory().join("tokenizer.json"))
        .expect("the pack tokenizer should load")
}

/// Encodes one rendered prompt into model token ids.
pub(crate) fn encode_prompt(tokenizer: &tokenizers::Tokenizer, rendered_prompt: &str) -> Vec<i32> {
    tokenizer
        .encode(rendered_prompt, false)
        .expect("the rendered prompt should encode")
        .get_ids()
        .iter()
        .map(|token_id| *token_id as i32)
        .collect()
}

/// Renders the single-user-turn, thinking-disabled chat template of the pack
/// around a continuation instruction and a Romeo and Juliet excerpt.
fn rendered_continuation_prompt(word_count: usize) -> String {
    let excerpt = ROMEO_AND_JULIET_SOURCE
        .split_whitespace()
        .take(word_count)
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "<|im_start|>user\n{CONTINUATION_INSTRUCTION}\n\n{excerpt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

/// Cuts the Romeo and Juliet source at the fewest words whose rendered
/// prompt encodes to at least `minimum_token_count` tokens, so a tier sends
/// at least the required prompt length without overshooting by more than a
/// word's worth of tokens.
pub(crate) fn romeo_and_juliet_prompt_token_ids(
    tokenizer: &tokenizers::Tokenizer,
    minimum_token_count: usize,
) -> Vec<i32> {
    let total_word_count = ROMEO_AND_JULIET_SOURCE.split_whitespace().count();
    let encoded_length = |word_count: usize| {
        encode_prompt(tokenizer, &rendered_continuation_prompt(word_count)).len()
    };
    assert!(
        encoded_length(total_word_count) >= minimum_token_count,
        "the Romeo and Juliet source must encode to at least {minimum_token_count} tokens"
    );
    let (mut low, mut high) = (1_usize, total_word_count);
    while low < high {
        let middle = low + (high - low) / 2;
        if encoded_length(middle) >= minimum_token_count {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    encode_prompt(tokenizer, &rendered_continuation_prompt(low))
}

/// Checks the measured completion shape the performance rules require.
pub(crate) fn assert_measured_shape(input_token_count: usize, output_token_count: usize) {
    assert!(
        (MINIMUM_ACCEPTABLE_INPUT_TOKENS..=MAXIMUM_ACCEPTABLE_INPUT_TOKENS)
            .contains(&input_token_count),
        "the measured prompt must carry 9,000 to 11,000 tokens, got {input_token_count}"
    );
    assert!(
        (MINIMUM_ACCEPTABLE_OUTPUT_TOKENS..=MAXIMUM_ACCEPTABLE_OUTPUT_TOKENS)
            .contains(&output_token_count),
        "the measured completion must generate 450 to 550 tokens, got {output_token_count}"
    );
}

/// Builds one lazy result and evaluates it, returning the wall time of graph
/// construction plus graphics-processor completion.
pub(crate) fn time_evaluated(
    model: &PrismHadamardTextModel,
    build: impl FnOnce() -> MlxArray,
) -> Duration {
    let started_at = Instant::now();
    let output = build();
    model
        .evaluate_outputs(&[&output])
        .expect("the measured graph should evaluate");
    started_at.elapsed()
}

/// Runs `warmup_iterations` discarded runs, then `measured_iterations`
/// timed runs, returning the measured wall times in milliseconds.
pub(crate) fn measure_iterations(
    model: &PrismHadamardTextModel,
    warmup_iterations: usize,
    measured_iterations: usize,
    mut build: impl FnMut() -> MlxArray,
) -> Vec<f64> {
    for _ in 0..warmup_iterations {
        time_evaluated(model, &mut build);
    }
    (0..measured_iterations)
        .map(|_| time_evaluated(model, &mut build).as_secs_f64() * 1_000.0)
        .collect()
}

/// Builds `repetitions` independent copies of one lazy result and evaluates
/// them together, so the device streams the kernels back to back and the one
/// host synchronization is amortized: the per-repetition time is the
/// kernel's own cost, which the per-call `measure_iterations` overstates by
/// the synchronization floor on sub-millisecond operations.
pub(crate) fn measure_streamed(
    model: &PrismHadamardTextModel,
    repetitions: usize,
    warmup_iterations: usize,
    measured_iterations: usize,
    mut build: impl FnMut() -> MlxArray,
) -> Vec<f64> {
    let mut run_once = || {
        let started_at = Instant::now();
        let outputs: Vec<MlxArray> = (0..repetitions).map(|_| build()).collect();
        let output_references: Vec<&MlxArray> = outputs.iter().collect();
        model
            .evaluate_outputs(&output_references)
            .expect("the streamed graph should evaluate");
        started_at.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64
    };
    for _ in 0..warmup_iterations {
        run_once();
    }
    (0..measured_iterations).map(|_| run_once()).collect()
}

/// The median of a sample set.
pub(crate) fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

/// Materializes an array so later timings never include its construction.
pub(crate) fn materialized(model: &PrismHadamardTextModel, array: MlxArray) -> MlxArray {
    model
        .evaluate_outputs(&[&array])
        .expect("the prepared input should evaluate");
    array
}

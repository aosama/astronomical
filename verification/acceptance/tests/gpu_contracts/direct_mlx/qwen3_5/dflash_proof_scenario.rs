//! The DFlash proof journey's measurement scenario and content sources.
//!
//! The scenario is the committed contract for what the journey measures:
//! every key — context length, prompt format, content corpus, verification
//! temperature, draft block length, and measured output tokens — is read
//! at runtime from the committed YAML file, so a measurement can be
//! retargeted by editing the file and rerunning, with no rebuild and no
//! environment variables.

/// The proof journey's scenario, parsed from the committed YAML file in the
/// fixtures directory (`dflash_proof_scenario.yaml`). That file documents
/// every key and the measured findings behind the defaults; editing it
/// changes the next run without a rebuild and without any environment
/// variable. The struct-level `default` keeps every key optional, and
/// `deny_unknown_fields` turns a typo'd key into a hard parse failure
/// instead of a silently ignored measurement choice.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ProofScenario {
    /// Prompt tokens fed to the tap-capturing prefill before decode is
    /// measured. Published Apple Silicon measurements show speculation
    /// peaking near one-to-four thousand tokens and eroding as key-value
    /// cache traffic grows past the weight read.
    pub(crate) context_tokens: usize,
    /// Whether the prompt is tokenized through the worker's chat template
    /// (the exact serving tokenization) instead of raw continuation. The
    /// winning format: measured acceptance is far higher chat-formatted
    /// than raw on this drafter.
    pub(crate) chat_formatted: bool,
    /// Which content corpus backs the measured prompt.
    pub(crate) fixture: ProofFixtureKind,
    /// Verification temperature in thousandths. The default 1,000 is the
    /// realistic serving point — the artifact's own default that users
    /// run. It samples one tempered target token per position (the exact
    /// sampled output law). Zero would run the greedy argmax contract,
    /// which no shipped user exercises; it stays an explicit diagnostic
    /// opt-in via the scenario file, never the default. Top-k and top-p
    /// stay at the engine's serving defaults.
    pub(crate) temperature_thousandths: u16,
    /// Draft block length per round. Four dominates eight on this stack:
    /// the drafter is roughly ninety-five percent reliable across draft
    /// positions one through four but only about forty-five percent across
    /// positions five through eight, and verify cost grows near-linearly
    /// per token on this sparse mixture-of-experts target. The vendor's
    /// block-length-eight recipe is tuned for flatter verify cost on other
    /// hardware.
    pub(crate) speculation_tokens: usize,
    /// Timed output tokens per mode: each of speculative and baseline
    /// generates this many tokens after the untimed warmup. The default
    /// 500 follows the repository performance convention; larger values
    /// sharpen the steady-state throughput estimate at the cost of run
    /// time.
    pub(crate) measured_output_tokens: usize,
}

impl Default for ProofScenario {
    fn default() -> Self {
        Self {
            context_tokens: 10_000,
            chat_formatted: true,
            fixture: ProofFixtureKind::SourceCode,
            temperature_thousandths: 1_000,
            speculation_tokens: 4,
            measured_output_tokens: 500,
        }
    }
}

/// The journey's prompt source corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProofFixtureKind {
    RomeoAndJuliet,
    SourceCode,
}

/// Absolute path of the committed scenario file, resolved from the
/// acceptance package root at compile time so the journey finds it from
/// any working directory it is invoked from.
const PROOF_SCENARIO_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/dflash_proof_scenario.yaml"
);

/// Loads the scenario file. A missing or malformed file fails the journey
/// loudly rather than silently falling back to defaults: a measurement run
/// must never report numbers the operator did not choose.
pub(crate) fn load_proof_scenario() -> ProofScenario {
    let scenario_text = match std::fs::read_to_string(PROOF_SCENARIO_PATH) {
        Ok(scenario_text) => scenario_text,
        Err(read_error) => panic!(
            "the DFlash proof scenario file {PROOF_SCENARIO_PATH} must be readable: {read_error}"
        ),
    };
    match serde_yaml::from_str(&scenario_text) {
        Ok(scenario) => scenario,
        Err(parse_error) => {
            panic!("the DFlash proof scenario file {PROOF_SCENARIO_PATH} must parse: {parse_error}")
        }
    }
}

/// The shared Shakespeare prose fixture, truncated by word count exactly as
/// the serving support helper does; the fifty-thousand-word edition backs
/// context sweeps beyond ten thousand tokens. Provenance: public-domain
/// Project Gutenberg editions — Romeo and Juliet complete, extended past
/// one play's length with Hamlet (the fifty-thousand-word file) and further
/// with Othello and Macbeth (the one-hundred-thousand-word file), so the
/// corpus never repeats on any period.
pub(crate) fn romeo_and_juliet_excerpt(maximum_word_count: usize) -> String {
    include_str!("../../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt")
        .split_whitespace()
        .take(maximum_word_count)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The real-source-code fixture, truncated by word count.
///
/// This is the content-class counterpart to the Shakespeare prose fixtures:
/// real, varied Python from the MLX library (MIT License, Apple Inc.)
/// concatenated in source order into a long, PROGRESSIVE corpus. This is
/// exactly the structured, in-distribution content class the published
/// DFlash reference stacks measure high acceptance on. Real source never
/// repeats on any short period — the first-thousand-word block does not
/// reappear later in the file. Provenance and the file list are recorded in
/// the fixture header. Covered by the AGENTS.md coding-fixture exception to
/// the Shakespeare fixtures because fiction prose is not suitable for
/// measuring a drafter against the reference's code regime.
pub(crate) fn progressive_source_code_excerpt(maximum_word_count: usize) -> String {
    include_str!("../../../fixtures/model_metrics_progressive_source_code.txt")
        .split_whitespace()
        .take(maximum_word_count)
        .collect::<Vec<_>>()
        .join(" ")
}

//! Calibration sequences for the exact sensitivity sweep.
//!
//! The bundled mix is OptiQ's `optiq.jsonl`, vendored verbatim under
//! `crates/quantization/calibration/` with its MIT license beside it: 40
//! hand-curated samples across six domains (prose, thinking traces, code
//! reasoning, agent loops, tool calling, constraint-bearing instructions).
//! Diversity of activation patterns is the point; a prose-only corpus leaves
//! the sweep blind to the layers tool calling depends on.
//!
//! Each JSONL line is one of:
//! - `{"domain": "prose", "text": "..."}`, raw text encoded directly;
//! - `{"domain": "thought", "messages": [...]}`, a chat;
//! - `{"domain": "tool", "messages": [...], "tools": [...]}`, a chat with
//!   tool definitions.
//!
//! OptiQ renders chat samples through the tokenizer's Jinja chat template.
//! The Rust `tokenizers` crate has no template engine, so this port renders
//! every chat sample with OptiQ's own fallback, role-prefixed concatenation
//! (`[role]: content` joined by blank lines), the path OptiQ takes whenever a
//! template rejects a sample. The consequence is that calibration never sees
//! a model's exact chat-control tokens; the sweep still sees the domain mix.
//! A caller who needs template-exact sequences supplies pre-tokenized samples.
//!
//! Every sample is encoded, the token streams are concatenated, and the
//! stream is cut into fixed-length sequences whose starting offsets are
//! shuffled by a seeded generator and truncated to the requested count.

use std::path::Path;

use serde::Deserialize;

use crate::error::QuantizationError;
use crate::progress::ProgressReporter;
use crate::sensitivity::CalibrationForwardSample;

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod calibration_tests;

/// The bundled six-domain mix, embedded so the binary needs no data path.
pub const BUNDLED_OPTIQ_MIX_JSONL: &str = include_str!("../../calibration/optiq.jsonl");

/// OptiQ's sweep defaults: 512 tokens per sequence is the memory-safe sweet
/// spot for 27B-plus models during bf16-streaming probes (longer sequences
/// inflate per-probe activation memory enough to trigger kernel memory
/// kills), and the mix is sampled with seed 42.
pub const DEFAULT_SEQUENCE_LENGTH_TOKENS: usize = 512;
pub const DEFAULT_SHUFFLE_SEED: u64 = 42;

/// One chat turn in a chat-style sample.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct CalibrationMessage {
    pub role: String,
    pub content: String,
}

/// One raw line of the mix.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CalibrationSample {
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<CalibrationMessage>>,
    /// Tool definitions, carried as opaque JSON; the fallback renderer does
    /// not inline them.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
}

impl CalibrationSample {
    /// The text to encode: `text` as-is, or the role-prefixed rendering of
    /// `messages`. Samples with neither are skipped.
    #[must_use]
    pub fn render_text(&self) -> Option<String> {
        todo!("return text, else join messages as \"[role]: content\" separated by blank lines")
    }
}

/// What the loader needs from a tokenizer. The production implementation
/// wraps the `tokenizers` crate; tests use a whitespace fake.
pub trait CalibrationTokenizer {
    /// Encodes text to token ids without adding special tokens.
    fn encode(&self, text: &str) -> Result<Vec<u32>, QuantizationError>;
}

/// The `tokenizers` crate adapter over a `tokenizer.json`.
pub struct HuggingFaceCalibrationTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl HuggingFaceCalibrationTokenizer {
    pub fn from_tokenizer_file(tokenizer_path: &Path) -> Result<Self, QuantizationError> {
        let _ = tokenizer_path;
        todo!("tokenizers::Tokenizer::from_file, mapped to TokenizerFailure")
    }
}

impl CalibrationTokenizer for HuggingFaceCalibrationTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, QuantizationError> {
        let _ = (&self.tokenizer, text);
        todo!("self.tokenizer.encode(text, false) and collect get_ids()")
    }
}

/// Token counts per domain, reported after loading so an operator can see
/// what the sweep was calibrated on.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CalibrationMixSummary {
    pub token_count_by_domain: std::collections::BTreeMap<String, usize>,
    pub sequence_count: usize,
    pub sequence_length_tokens: usize,
}

/// A loaded, tokenized, chunked mix.
#[derive(Clone, Debug)]
pub struct CalibrationMix {
    samples: Vec<CalibrationForwardSample>,
    summary: CalibrationMixSummary,
}

impl CalibrationMix {
    /// Parses JSONL text into samples, skipping blank lines.
    pub fn parse_jsonl(jsonl_text: &str) -> Result<Vec<CalibrationSample>, QuantizationError> {
        let _ = jsonl_text;
        todo!(
            "serde_json::from_str per non-blank line; map a parse failure to CalibrationMixUnreadable with the line number"
        )
    }

    /// Loads the bundled mix.
    pub fn bundled(
        tokenizer: &dyn CalibrationTokenizer,
        sequence_count: usize,
        sequence_length_tokens: usize,
        shuffle_seed: u64,
        progress_reporter: &mut dyn ProgressReporter,
    ) -> Result<Self, QuantizationError> {
        let samples = Self::parse_jsonl(BUNDLED_OPTIQ_MIX_JSONL)?;
        Self::from_samples(
            samples,
            tokenizer,
            sequence_count,
            sequence_length_tokens,
            shuffle_seed,
            progress_reporter,
        )
    }

    /// Loads a caller-supplied JSONL file in the same schema.
    pub fn from_jsonl_file(
        mix_path: &Path,
        tokenizer: &dyn CalibrationTokenizer,
        sequence_count: usize,
        sequence_length_tokens: usize,
        shuffle_seed: u64,
        progress_reporter: &mut dyn ProgressReporter,
    ) -> Result<Self, QuantizationError> {
        let _ = (
            mix_path,
            tokenizer,
            sequence_count,
            sequence_length_tokens,
            shuffle_seed,
            progress_reporter,
        );
        todo!(
            "read the file (CalibrationMixUnreadable on failure), parse, and delegate to from_samples"
        )
    }

    /// Tokenizes, concatenates, and chunks.
    ///
    /// Work to port (OptiQ's `_load_optiq_calibration`):
    /// 1. Render and encode every sample, accumulating per-domain token
    ///    counts and advancing progress once per sample.
    /// 2. Fail with `CalibrationMixTooShort` when the concatenated stream is
    ///    shorter than one sequence.
    /// 3. Starting offsets are `0, L, 2L, ...` up to `total - L` (exclusive),
    ///    shuffled with a seeded generator, then the first `sequence_count`
    ///    are cut into sequences. The shuffle must be a documented
    ///    deterministic algorithm (a small xorshift or splitmix over the
    ///    offsets); matching numpy's `RandomState.shuffle` exactly is not a
    ///    goal, reproducibility across Rust runs is.
    pub fn from_samples(
        samples: Vec<CalibrationSample>,
        tokenizer: &dyn CalibrationTokenizer,
        sequence_count: usize,
        sequence_length_tokens: usize,
        shuffle_seed: u64,
        progress_reporter: &mut dyn ProgressReporter,
    ) -> Result<Self, QuantizationError> {
        let _ = (
            samples,
            tokenizer,
            sequence_count,
            sequence_length_tokens,
            shuffle_seed,
            progress_reporter,
        );
        todo!(
            "tokenize, concatenate, shuffle offsets deterministically, and chunk as documented above"
        )
    }

    /// Builds a mix from sequences the caller already tokenized (for example
    /// through a chat template this crate cannot render).
    #[must_use]
    pub fn from_pretokenized(samples: Vec<CalibrationForwardSample>) -> Self {
        let sequence_length_tokens = samples.first().map_or(0, |sample| sample.token_ids.len());
        let summary = CalibrationMixSummary {
            token_count_by_domain: std::collections::BTreeMap::new(),
            sequence_count: samples.len(),
            sequence_length_tokens,
        };
        Self { samples, summary }
    }

    #[must_use]
    pub fn samples(&self) -> &[CalibrationForwardSample] {
        &self.samples
    }

    #[must_use]
    pub const fn summary(&self) -> &CalibrationMixSummary {
        &self.summary
    }
}

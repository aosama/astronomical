//! Loading, tokenizing, and chunking a calibration mix into fixed-length
//! sequences for the sensitivity sweep.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::QuantizationError;
use crate::progress::{ProgressReporter, StageProgressTracker};
use crate::sensitivity::CalibrationForwardSample;

use super::sample::CalibrationSample;
use super::seeded_shuffle::shuffle_like_numpy_random_state;
use super::tokenizer::CalibrationTokenizer;

/// The bundled six-domain mix, embedded so the binary needs no data path.
pub const BUNDLED_OPTIQ_MIX_JSONL: &str = include_str!("../../calibration/optiq.jsonl");

/// OptiQ's sweep defaults: 512 tokens per sequence is the memory-safe sweet
/// spot for 27B-plus models during bf16-streaming probes (longer sequences
/// inflate per-probe activation memory enough to trigger kernel memory
/// kills), and the mix is sampled with seed 42.
pub const DEFAULT_SEQUENCE_LENGTH_TOKENS: usize = 512;
pub const DEFAULT_SHUFFLE_SEED: u64 = 42;

/// Token counts per domain, reported after loading so an operator can see
/// what the sweep was calibrated on.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CalibrationMixSummary {
    pub token_count_by_domain: BTreeMap<String, usize>,
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
        parse_jsonl_from(jsonl_text, Path::new(INLINE_MIX_SOURCE_NAME))
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
        let jsonl_text = match std::fs::read_to_string(mix_path) {
            Ok(jsonl_text) => jsonl_text,
            Err(read_error) => {
                return Err(QuantizationError::CalibrationMixUnreadable {
                    mix_path: mix_path.to_path_buf(),
                    description: read_error.to_string(),
                });
            }
        };
        let samples = parse_jsonl_from(&jsonl_text, mix_path)?;
        Self::from_samples(
            samples,
            tokenizer,
            sequence_count,
            sequence_length_tokens,
            shuffle_seed,
            progress_reporter,
        )
    }

    /// Tokenizes, concatenates, and chunks, as OptiQ's
    /// `_load_optiq_calibration` does:
    /// 1. Render and encode every sample, accumulating per-domain token
    ///    counts and advancing progress once per sample.
    /// 2. Fail with `CalibrationMixTooShort` when the stream does not hold
    ///    one full sequence plus at least one more token (OptiQ's offsets
    ///    stop short of `total - L`, so exactly `L` tokens yield nothing).
    /// 3. Starting offsets `0, L, 2L, ...` below `total - L` are shuffled
    ///    exactly as `np.random.RandomState(seed).shuffle` would, and the
    ///    first `sequence_count` are cut into sequences, so a Rust sweep sees
    ///    the same sequences as a Python sweep of the same mix and tokenizer.
    pub fn from_samples(
        samples: Vec<CalibrationSample>,
        tokenizer: &dyn CalibrationTokenizer,
        sequence_count: usize,
        sequence_length_tokens: usize,
        shuffle_seed: u64,
        progress_reporter: &mut dyn ProgressReporter,
    ) -> Result<Self, QuantizationError> {
        let (token_stream, token_count_by_domain) =
            tokenize_samples(&samples, tokenizer, progress_reporter)?;
        let mut sequence_offsets = sequence_offsets(token_stream.len(), sequence_length_tokens)?;
        shuffle_like_numpy_random_state(&mut sequence_offsets, shuffle_seed);
        sequence_offsets.truncate(sequence_count);
        let mut forward_samples = Vec::with_capacity(sequence_offsets.len());
        for sequence_offset in sequence_offsets {
            let sequence_tokens =
                &token_stream[sequence_offset..sequence_offset + sequence_length_tokens];
            forward_samples.push(CalibrationForwardSample {
                token_ids: sequence_tokens.to_vec(),
            });
        }
        let summary = CalibrationMixSummary {
            token_count_by_domain,
            sequence_count: forward_samples.len(),
            sequence_length_tokens,
        };
        Ok(Self {
            samples: forward_samples,
            summary,
        })
    }

    /// Builds a mix from sequences the caller already tokenized (for example
    /// through a chat template this crate cannot render).
    #[must_use]
    pub fn from_pretokenized(samples: Vec<CalibrationForwardSample>) -> Self {
        let sequence_length_tokens = match samples.first() {
            Some(first_sample) => first_sample.token_ids.len(),
            None => 0,
        };
        let summary = CalibrationMixSummary {
            token_count_by_domain: BTreeMap::new(),
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

/// The source name reported for JSONL parsed from memory (the bundled mix).
const INLINE_MIX_SOURCE_NAME: &str = "<inline calibration mix>";

/// The domain OptiQ records for a sample that names none.
const UNNAMED_DOMAIN: &str = "?";

/// Parses one sample per non-blank line, naming the source and the line on
/// failure.
fn parse_jsonl_from(
    jsonl_text: &str,
    mix_path: &Path,
) -> Result<Vec<CalibrationSample>, QuantizationError> {
    let mut samples = Vec::new();
    for (line_index, line) in jsonl_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(line) {
            Ok(sample) => samples.push(sample),
            Err(parse_error) => {
                return Err(QuantizationError::CalibrationMixUnreadable {
                    mix_path: mix_path.to_path_buf(),
                    description: format!("line {}: {parse_error}", line_index + 1),
                });
            }
        }
    }
    Ok(samples)
}

/// Renders and encodes every sample into one stream, reporting progress per
/// sample on the repository's five-second cadence.
fn tokenize_samples(
    samples: &[CalibrationSample],
    tokenizer: &dyn CalibrationTokenizer,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<(Vec<u32>, BTreeMap<String, usize>), QuantizationError> {
    let mut progress_tracker = StageProgressTracker::new(
        "calibration tokenization",
        samples.len() as u64,
        0,
        StageProgressTracker::DEFAULT_REPORT_INTERVAL,
        progress_reporter,
    );
    let mut token_stream = Vec::new();
    let mut token_count_by_domain = BTreeMap::new();
    for sample in samples {
        let domain = if sample.domain.is_empty() {
            UNNAMED_DOMAIN
        } else {
            sample.domain.as_str()
        };
        if let Some(rendered_text) = rendered_sample_text(tokenizer, sample) {
            let sample_tokens = tokenizer.encode(&rendered_text)?;
            *token_count_by_domain.entry(domain.to_owned()).or_insert(0) += sample_tokens.len();
            token_stream.extend(sample_tokens);
        }
        progress_tracker.advance(domain);
    }
    progress_tracker.report_now("tokenized");
    Ok((token_stream, token_count_by_domain))
}

/// `0, L, 2L, ...` strictly below `total - L`, OptiQ's chunk starts.
/// The sample through the model's chat template, else as flat text.
fn rendered_sample_text(
    tokenizer: &dyn CalibrationTokenizer,
    sample: &CalibrationSample,
) -> Option<String> {
    match tokenizer.render_chat(sample) {
        Some(templated_text) => Some(templated_text),
        None => sample.render_text(),
    }
}

fn sequence_offsets(
    available_token_count: usize,
    sequence_length_tokens: usize,
) -> Result<Vec<usize>, QuantizationError> {
    let too_short = QuantizationError::CalibrationMixTooShort {
        available_token_count,
        sequence_length_tokens,
    };
    if sequence_length_tokens == 0 || available_token_count <= sequence_length_tokens {
        return Err(too_short);
    }
    let last_exclusive_offset = available_token_count - sequence_length_tokens;
    Ok((0..last_exclusive_offset)
        .step_by(sequence_length_tokens)
        .collect())
}

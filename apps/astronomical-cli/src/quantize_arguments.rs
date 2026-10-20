//! Arguments for `astronomical quantize`, an ephemeral in-process verb with
//! three sub-verbs.
//!
//! - `quantize plan --sensitivity FILE --target-bits-per-weight N [...]`:
//!   allocate from an existing `optiq/sensitivity.json` and print the bit
//!   map. Pure; runs on any host; the way to re-target an already-measured
//!   base without a sweep.
//! - `quantize inspect ARTIFACT_DIR`: print an artifact's quantization
//!   provenance (method, reference, target and achieved bits per weight,
//!   per-width module counts). Pure.
//! - `quantize convert SOURCE_DIR --output DIR [...]`: the full pipeline.
//!   Needs the `quantize-convert` build feature.
//!
//! Flag names spell every word out, per the repository's naming rule:
//! `--target-bits-per-weight`, never `--target-bpw`.

use std::ffi::OsString;
use std::path::PathBuf;

use crate::errors::UsageError;

/// Which quantize sub-verb was requested.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantizeCommand {
    Plan(QuantizePlanArguments),
    Inspect(QuantizeInspectArguments),
    Convert(QuantizeConvertArguments),
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizePlanArguments {
    pub sensitivity_document_path: PathBuf,
    pub target_bits_per_weight: f64,
    /// `None` infers the ladder from the document.
    pub candidate_bits: Option<Vec<u32>>,
    pub floor_components_per_block: usize,
    pub maximum_low_bit_block_run: usize,
    pub render_json: bool,
}

impl Eq for QuantizePlanArguments {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizeInspectArguments {
    pub artifact_directory: PathBuf,
    pub render_json: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizeConvertArguments {
    pub source_directory: PathBuf,
    pub output_directory: PathBuf,
    pub base_model_name: Option<String>,
    /// `exact` (default) or `static`.
    pub method: QuantizeMethodArgument,
    /// `auto` (default), `bf16`, or `uniform-4-bit`.
    pub reference: QuantizeReferenceArgument,
    pub target_bits_per_weight: f64,
    pub candidate_bits: Vec<u32>,
    pub group_size: u32,
    pub calibration_sequence_count: usize,
    pub calibration_mix_path: Option<PathBuf>,
    pub floor_components_per_block: usize,
    pub range_search_enabled: bool,
    pub range_search_maximum_bits: u32,
    pub overwrite_output: bool,
}

impl Eq for QuantizeConvertArguments {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizeMethodArgument {
    Exact,
    Static,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizeReferenceArgument {
    Auto,
    Bfloat16,
    Uniform4Bit,
}

/// OptiQ's command-line defaults, kept so a Rust conversion with no flags
/// matches a Python conversion with no flags.
pub const DEFAULT_TARGET_BITS_PER_WEIGHT: f64 = 5.0;
pub const DEFAULT_CANDIDATE_BITS: [u32; 2] = [4, 8];
pub const DEFAULT_GROUP_SIZE: u32 = 64;
pub const DEFAULT_CALIBRATION_SEQUENCE_COUNT: usize = 24;
pub const DEFAULT_FLOOR_COMPONENTS_PER_BLOCK: usize = 2;
pub const DEFAULT_MAXIMUM_LOW_BIT_BLOCK_RUN: usize = 3;
pub const DEFAULT_RANGE_SEARCH_MAXIMUM_BITS: u32 = 3;

/// Parses everything after `astronomical quantize`.
///
/// Work to implement: dispatch on the first argument (`plan`, `inspect`,
/// `convert`; missing → `MissingCommand`; other →
/// `UnknownQuantizeTarget`), then parse the sub-verb's flags with the same
/// repeated-flag and missing-value discipline as the other verbs. A
/// `--candidate-bits` value is a comma-separated list of integers; a
/// non-integer or an unsupported width is `InvalidQuantizeValue`. Floats
/// for `--target-bits-per-weight` must be finite and positive.
pub fn parse_quantize_command(
    remaining_arguments: &[OsString],
) -> Result<QuantizeCommand, UsageError> {
    let _ = remaining_arguments;
    todo!("dispatch on the sub-verb and parse its flags")
}

/// Parses a comma-separated bit-width list such as `2,4,8`.
pub fn parse_candidate_bits(value: &str) -> Result<Vec<u32>, UsageError> {
    let _ = value;
    todo!(
        "split on commas, parse each as u32, validate through astronomical_quantization::BitWidth"
    )
}

//! Arguments for `astronomical quantize`, an ephemeral in-process verb with
//! four sub-verbs.
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
//! - `quantize evaluate ARTIFACT_DIR --reference DIR [...]`: how far the
//!   artifact's next-token distribution drifts from a reference model's.
//!   Needs the `quantize-convert` build feature.
//!
//! Flag names spell every word out, per the repository's naming rule:
//! `--target-bits-per-weight`, never `--target-bpw`.
//!
//! The defaults are OptiQ's command-line defaults (`optiq/cli.py`).

use std::ffi::OsString;
use std::path::PathBuf;

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::evaluation::{
    DEFAULT_EVALUATION_PROMPT_COUNT, DEFAULT_EVALUATION_SEQUENCE_LENGTH_TOKENS,
};

use crate::arguments::UsageError;

/// Which quantize sub-verb was requested.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantizeCommand {
    Plan(QuantizePlanArguments),
    Inspect(QuantizeInspectArguments),
    Convert(QuantizeConvertArguments),
    Evaluate(QuantizeEvaluateArguments),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizeEvaluateArguments {
    /// The quant being evaluated.
    pub artifact_directory: PathBuf,
    /// The model whose distribution is the truth, usually the bfloat16 base.
    pub reference_directory: PathBuf,
    pub prompt_count: usize,
    pub sequence_length_tokens: usize,
    pub render_json: bool,
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

/// Parses everything after `astronomical quantize`: the sub-verb, then its
/// flags with the same repeated-flag and missing-value discipline as the
/// other verbs.
pub fn parse_quantize_command(
    remaining_arguments: &[OsString],
) -> Result<QuantizeCommand, UsageError> {
    let Some(sub_verb) = remaining_arguments.first() else {
        return Err(UsageError::MissingCommand);
    };
    parse_quantize_sub_verb(&sub_verb.to_string_lossy(), &remaining_arguments[1..])
}

/// One sub-verb's arguments, by the sub-verb's name.
fn parse_quantize_sub_verb(
    sub_verb: &str,
    flag_arguments: &[OsString],
) -> Result<QuantizeCommand, UsageError> {
    match sub_verb {
        "plan" => parse_plan_arguments(flag_arguments).map(QuantizeCommand::Plan),
        "inspect" => parse_inspect_arguments(flag_arguments).map(QuantizeCommand::Inspect),
        "convert" => parse_convert_arguments(flag_arguments).map(QuantizeCommand::Convert),
        "evaluate" => parse_evaluate_arguments(flag_arguments).map(QuantizeCommand::Evaluate),
        unknown_target => Err(UsageError::UnknownQuantizeTarget(unknown_target.to_owned())),
    }
}

/// Parses a comma-separated bit-width list such as `2,4,8`; every entry
/// must be an MLX affine width or the keep-bfloat16 tier.
pub fn parse_candidate_bits(value: &str) -> Result<Vec<u32>, UsageError> {
    let mut candidate_bits = Vec::new();
    for bit_text in value.split(',') {
        let bits = parse_unsigned::<u32>("--candidate-bits", bit_text.trim())?;
        if let Err(width_error) = BitWidth::new(bits) {
            return Err(invalid_value("--candidate-bits", width_error.to_string()));
        }
        if candidate_bits.contains(&bits) {
            return Err(invalid_value(
                "--candidate-bits",
                format!("{bits} appears twice"),
            ));
        }
        candidate_bits.push(bits);
    }
    Ok(candidate_bits)
}

/// Walks a sub-verb's arguments one flag at a time.
struct QuantizeArgumentCursor<'arguments> {
    arguments: &'arguments [OsString],
    next_index: usize,
}

impl<'arguments> QuantizeArgumentCursor<'arguments> {
    const fn new(arguments: &'arguments [OsString]) -> Self {
        Self {
            arguments,
            next_index: 0,
        }
    }

    /// The next argument, lossily as text.
    fn next_argument(&mut self) -> Option<String> {
        let argument = self.arguments.get(self.next_index)?;
        self.next_index += 1;
        Some(argument.to_string_lossy().into_owned())
    }

    /// The value after a value-taking flag; a missing or flag-shaped value
    /// is the flag's missing value.
    fn flag_value(&mut self, flag_name: &'static str) -> Result<String, UsageError> {
        match self.next_argument() {
            Some(flag_value) if !flag_value.is_empty() && !flag_value.starts_with('-') => {
                Ok(flag_value)
            }
            _ => Err(UsageError::MissingValue(flag_name)),
        }
    }
}

/// The plan sub-verb's flags, accumulated before defaults are applied.
#[derive(Default)]
struct PlanFlagValues {
    sensitivity_document_path: Option<PathBuf>,
    target_bits_per_weight: Option<f64>,
    candidate_bits: Option<Vec<u32>>,
    floor_components_per_block: Option<usize>,
    maximum_low_bit_block_run: Option<usize>,
    render_json: bool,
}

fn parse_plan_arguments(arguments: &[OsString]) -> Result<QuantizePlanArguments, UsageError> {
    let mut cursor = QuantizeArgumentCursor::new(arguments);
    let mut flag_values = PlanFlagValues::default();
    while let Some(argument) = cursor.next_argument() {
        apply_plan_argument(&mut flag_values, &mut cursor, argument)?;
    }
    Ok(QuantizePlanArguments {
        sensitivity_document_path: required(
            flag_values.sensitivity_document_path,
            "--sensitivity",
        )?,
        target_bits_per_weight: required(
            flag_values.target_bits_per_weight,
            "--target-bits-per-weight",
        )?,
        candidate_bits: flag_values.candidate_bits,
        floor_components_per_block: flag_values
            .floor_components_per_block
            .unwrap_or(DEFAULT_FLOOR_COMPONENTS_PER_BLOCK),
        maximum_low_bit_block_run: flag_values
            .maximum_low_bit_block_run
            .unwrap_or(DEFAULT_MAXIMUM_LOW_BIT_BLOCK_RUN),
        render_json: flag_values.render_json,
    })
}

fn apply_plan_argument(
    flag_values: &mut PlanFlagValues,
    cursor: &mut QuantizeArgumentCursor<'_>,
    argument: String,
) -> Result<(), UsageError> {
    match argument.as_str() {
        "--sensitivity" => {
            let path = PathBuf::from(cursor.flag_value("--sensitivity")?);
            set_once(
                &mut flag_values.sensitivity_document_path,
                path,
                "--sensitivity",
            )
        }
        "--target-bits-per-weight" => {
            let target = parse_target(&cursor.flag_value("--target-bits-per-weight")?)?;
            set_once(
                &mut flag_values.target_bits_per_weight,
                target,
                "--target-bits-per-weight",
            )
        }
        "--candidate-bits" => {
            let ladder = parse_candidate_bits(&cursor.flag_value("--candidate-bits")?)?;
            set_once(&mut flag_values.candidate_bits, ladder, "--candidate-bits")
        }
        "--floor-components-per-block" => {
            let flag = "--floor-components-per-block";
            let floor = parse_unsigned::<usize>(flag, &cursor.flag_value(flag)?)?;
            set_once(&mut flag_values.floor_components_per_block, floor, flag)
        }
        "--maximum-low-bit-block-run" => {
            let flag = "--maximum-low-bit-block-run";
            let run = parse_unsigned::<usize>(flag, &cursor.flag_value(flag)?)?;
            set_once(&mut flag_values.maximum_low_bit_block_run, run, flag)
        }
        "--json" => set_switch(&mut flag_values.render_json, "--json"),
        _ => Err(UsageError::UnknownArgument(argument)),
    }
}

/// The inspect sub-verb: one directory and `--json`.
fn parse_inspect_arguments(arguments: &[OsString]) -> Result<QuantizeInspectArguments, UsageError> {
    let mut cursor = QuantizeArgumentCursor::new(arguments);
    let mut artifact_directory = None;
    let mut render_json = false;
    while let Some(argument) = cursor.next_argument() {
        if argument == "--json" {
            set_switch(&mut render_json, "--json")?;
        } else if argument.starts_with('-') || artifact_directory.is_some() {
            return Err(UsageError::UnknownArgument(argument));
        } else {
            artifact_directory = Some(PathBuf::from(argument));
        }
    }
    Ok(QuantizeInspectArguments {
        artifact_directory: required(artifact_directory, "ARTIFACT_DIR")?,
        render_json,
    })
}

/// `quantize evaluate ARTIFACT_DIR --reference DIR [--prompts N]
/// [--sequence-length N] [--json]`, with OptiQ's defaults of 64 prompts of
/// at most 256 tokens.
fn parse_evaluate_arguments(
    arguments: &[OsString],
) -> Result<QuantizeEvaluateArguments, UsageError> {
    let mut cursor = QuantizeArgumentCursor::new(arguments);
    let mut flag_values = EvaluateFlagValues::default();
    while let Some(argument) = cursor.next_argument() {
        apply_evaluate_argument(&mut flag_values, &mut cursor, argument)?;
    }
    Ok(QuantizeEvaluateArguments {
        artifact_directory: required(flag_values.artifact_directory, "ARTIFACT_DIR")?,
        reference_directory: required(flag_values.reference_directory, "--reference")?,
        prompt_count: flag_values
            .prompt_count
            .unwrap_or(DEFAULT_EVALUATION_PROMPT_COUNT),
        sequence_length_tokens: flag_values
            .sequence_length_tokens
            .unwrap_or(DEFAULT_EVALUATION_SEQUENCE_LENGTH_TOKENS),
        render_json: flag_values.render_json,
    })
}

/// The evaluate sub-verb's flags, accumulated before defaults are applied.
#[derive(Default)]
struct EvaluateFlagValues {
    artifact_directory: Option<PathBuf>,
    reference_directory: Option<PathBuf>,
    prompt_count: Option<usize>,
    sequence_length_tokens: Option<usize>,
    render_json: bool,
}

/// Applies one evaluate argument: a flag with its value, the switch, or the
/// positional artifact directory.
fn apply_evaluate_argument(
    flag_values: &mut EvaluateFlagValues,
    cursor: &mut QuantizeArgumentCursor<'_>,
    argument: String,
) -> Result<(), UsageError> {
    match argument.as_str() {
        "--reference" => {
            let reference = PathBuf::from(cursor.flag_value("--reference")?);
            set_once(
                &mut flag_values.reference_directory,
                reference,
                "--reference",
            )
        }
        "--prompts" => {
            let parsed_count = parse_positive("--prompts", &cursor.flag_value("--prompts")?)?;
            set_once(&mut flag_values.prompt_count, parsed_count, "--prompts")
        }
        "--sequence-length" => {
            let sequence_length_text = cursor.flag_value("--sequence-length")?;
            let parsed_length = parse_positive("--sequence-length", &sequence_length_text)?;
            set_once(
                &mut flag_values.sequence_length_tokens,
                parsed_length,
                "--sequence-length",
            )
        }
        "--json" => set_switch(&mut flag_values.render_json, "--json"),
        _ if argument.starts_with('-') || flag_values.artifact_directory.is_some() => {
            Err(UsageError::UnknownArgument(argument))
        }
        _ => {
            flag_values.artifact_directory = Some(PathBuf::from(argument));
            Ok(())
        }
    }
}

/// A whole number above zero: an evaluation of no prompts or no tokens
/// measures nothing.
fn parse_positive(flag: &'static str, unsigned_text: &str) -> Result<usize, UsageError> {
    let parsed_value = parse_unsigned::<usize>(flag, unsigned_text)?;
    if parsed_value == 0 {
        return Err(invalid_value(flag, "must be at least 1".to_owned()));
    }
    Ok(parsed_value)
}

/// The convert sub-verb's flags, accumulated before defaults are applied.
#[derive(Default)]
struct ConvertFlagValues {
    source_directory: Option<PathBuf>,
    output_directory: Option<PathBuf>,
    base_model_name: Option<String>,
    method: Option<QuantizeMethodArgument>,
    reference: Option<QuantizeReferenceArgument>,
    target_bits_per_weight: Option<f64>,
    candidate_bits: Option<Vec<u32>>,
    group_size: Option<u32>,
    calibration_sequence_count: Option<usize>,
    calibration_mix_path: Option<PathBuf>,
    floor_components_per_block: Option<usize>,
    range_search_maximum_bits: Option<u32>,
    is_range_search_disabled: bool,
    overwrite_output: bool,
}

fn parse_convert_arguments(arguments: &[OsString]) -> Result<QuantizeConvertArguments, UsageError> {
    let mut cursor = QuantizeArgumentCursor::new(arguments);
    let mut flag_values = ConvertFlagValues::default();
    while let Some(argument) = cursor.next_argument() {
        apply_convert_argument(&mut flag_values, &mut cursor, argument)?;
    }
    Ok(QuantizeConvertArguments {
        source_directory: required(flag_values.source_directory, "SOURCE_DIR")?,
        output_directory: required(flag_values.output_directory, "--output")?,
        base_model_name: flag_values.base_model_name,
        method: flag_values.method.unwrap_or(QuantizeMethodArgument::Exact),
        reference: flag_values
            .reference
            .unwrap_or(QuantizeReferenceArgument::Auto),
        target_bits_per_weight: flag_values
            .target_bits_per_weight
            .unwrap_or(DEFAULT_TARGET_BITS_PER_WEIGHT),
        candidate_bits: match flag_values.candidate_bits {
            Some(candidate_bits) => candidate_bits,
            None => DEFAULT_CANDIDATE_BITS.to_vec(),
        },
        group_size: flag_values.group_size.unwrap_or(DEFAULT_GROUP_SIZE),
        calibration_sequence_count: flag_values
            .calibration_sequence_count
            .unwrap_or(DEFAULT_CALIBRATION_SEQUENCE_COUNT),
        calibration_mix_path: flag_values.calibration_mix_path,
        floor_components_per_block: flag_values
            .floor_components_per_block
            .unwrap_or(DEFAULT_FLOOR_COMPONENTS_PER_BLOCK),
        range_search_enabled: !flag_values.is_range_search_disabled,
        range_search_maximum_bits: flag_values
            .range_search_maximum_bits
            .unwrap_or(DEFAULT_RANGE_SEARCH_MAXIMUM_BITS),
        overwrite_output: flag_values.overwrite_output,
    })
}

/// Applies one convert argument: a value-taking flag, a switch, or the
/// positional source directory.
fn apply_convert_argument(
    flag_values: &mut ConvertFlagValues,
    cursor: &mut QuantizeArgumentCursor<'_>,
    argument: String,
) -> Result<(), UsageError> {
    match argument.as_str() {
        "--output" => {
            let output = PathBuf::from(cursor.flag_value("--output")?);
            set_once(&mut flag_values.output_directory, output, "--output")
        }
        "--base-model" => {
            let base_model_name = cursor.flag_value("--base-model")?;
            set_once(
                &mut flag_values.base_model_name,
                base_model_name,
                "--base-model",
            )
        }
        "--method" => {
            let method = parse_method(&cursor.flag_value("--method")?)?;
            set_once(&mut flag_values.method, method, "--method")
        }
        "--reference" => {
            let reference = parse_reference(&cursor.flag_value("--reference")?)?;
            set_once(&mut flag_values.reference, reference, "--reference")
        }
        "--no-range-search" => set_switch(
            &mut flag_values.is_range_search_disabled,
            "--no-range-search",
        ),
        "--overwrite" => set_switch(&mut flag_values.overwrite_output, "--overwrite"),
        _ => apply_convert_numeric_or_path(flag_values, cursor, argument),
    }
}

/// The numeric and path-valued convert flags, and the positional source.
fn apply_convert_numeric_or_path(
    flag_values: &mut ConvertFlagValues,
    cursor: &mut QuantizeArgumentCursor<'_>,
    argument: String,
) -> Result<(), UsageError> {
    match argument.as_str() {
        "--target-bits-per-weight" => {
            let target = parse_target(&cursor.flag_value("--target-bits-per-weight")?)?;
            set_once(
                &mut flag_values.target_bits_per_weight,
                target,
                "--target-bits-per-weight",
            )
        }
        "--candidate-bits" => {
            let ladder = parse_candidate_bits(&cursor.flag_value("--candidate-bits")?)?;
            set_once(&mut flag_values.candidate_bits, ladder, "--candidate-bits")
        }
        "--group-size" => {
            let group_size = parse_group_size(&cursor.flag_value("--group-size")?)?;
            set_once(&mut flag_values.group_size, group_size, "--group-size")
        }
        "--calibration-sequences" => {
            let flag = "--calibration-sequences";
            let count = parse_unsigned::<usize>(flag, &cursor.flag_value(flag)?)?;
            if count == 0 {
                return Err(invalid_value(flag, "must be at least 1".to_owned()));
            }
            set_once(&mut flag_values.calibration_sequence_count, count, flag)
        }
        "--calibration-mix" => {
            let mix_path = PathBuf::from(cursor.flag_value("--calibration-mix")?);
            set_once(
                &mut flag_values.calibration_mix_path,
                mix_path,
                "--calibration-mix",
            )
        }
        "--floor-components-per-block" => {
            let flag = "--floor-components-per-block";
            let floor = parse_unsigned::<usize>(flag, &cursor.flag_value(flag)?)?;
            set_once(&mut flag_values.floor_components_per_block, floor, flag)
        }
        "--range-search-maximum-bits" => {
            let flag = "--range-search-maximum-bits";
            let maximum_bits = parse_unsigned::<u32>(flag, &cursor.flag_value(flag)?)?;
            set_once(
                &mut flag_values.range_search_maximum_bits,
                maximum_bits,
                flag,
            )
        }
        _ if argument.starts_with('-') || flag_values.source_directory.is_some() => {
            Err(UsageError::UnknownArgument(argument))
        }
        _ => {
            flag_values.source_directory = Some(PathBuf::from(argument));
            Ok(())
        }
    }
}

fn parse_method(method_text: &str) -> Result<QuantizeMethodArgument, UsageError> {
    match method_text {
        "exact" => Ok(QuantizeMethodArgument::Exact),
        "static" => Ok(QuantizeMethodArgument::Static),
        other => Err(invalid_value(
            "--method",
            format!("{other} is not exact or static"),
        )),
    }
}

fn parse_reference(reference_text: &str) -> Result<QuantizeReferenceArgument, UsageError> {
    match reference_text {
        "auto" => Ok(QuantizeReferenceArgument::Auto),
        "bf16" => Ok(QuantizeReferenceArgument::Bfloat16),
        "uniform-4-bit" => Ok(QuantizeReferenceArgument::Uniform4Bit),
        other => Err(invalid_value(
            "--reference",
            format!("{other} is not auto, bf16, or uniform-4-bit"),
        )),
    }
}

/// A target must be a finite, positive number of bits per weight.
fn parse_target(target_text: &str) -> Result<f64, UsageError> {
    match target_text.parse::<f64>() {
        Ok(target) if target.is_finite() && target > 0.0 => Ok(target),
        _ => Err(invalid_value(
            "--target-bits-per-weight",
            format!("{target_text} is not a finite positive number"),
        )),
    }
}

fn parse_group_size(group_size_text: &str) -> Result<u32, UsageError> {
    let group_size = parse_unsigned::<u32>("--group-size", group_size_text)?;
    match GroupSize::new(group_size) {
        Ok(_) => Ok(group_size),
        Err(group_error) => Err(invalid_value("--group-size", group_error.to_string())),
    }
}

fn parse_unsigned<Unsigned: std::str::FromStr>(
    flag: &'static str,
    unsigned_text: &str,
) -> Result<Unsigned, UsageError> {
    match unsigned_text.parse::<Unsigned>() {
        Ok(unsigned_value) => Ok(unsigned_value),
        Err(_) => Err(invalid_value(
            flag,
            format!("{unsigned_text} is not a whole number"),
        )),
    }
}

fn set_once<Value>(
    slot: &mut Option<Value>,
    value: Value,
    flag: &'static str,
) -> Result<(), UsageError> {
    if slot.is_some() {
        return Err(UsageError::RepeatedArgument(flag));
    }
    *slot = Some(value);
    Ok(())
}

fn set_switch(switch: &mut bool, flag: &'static str) -> Result<(), UsageError> {
    if *switch {
        return Err(UsageError::RepeatedArgument(flag));
    }
    *switch = true;
    Ok(())
}

fn required<Value>(slot: Option<Value>, flag: &'static str) -> Result<Value, UsageError> {
    match slot {
        Some(value) => Ok(value),
        None => Err(UsageError::MissingValue(flag)),
    }
}

const fn invalid_value(flag: &'static str, reason: String) -> UsageError {
    UsageError::InvalidQuantizeValue { flag, reason }
}

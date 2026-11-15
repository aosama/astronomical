//! Arguments for `astronomical decision`, power-user tools for decision models.
//!
//! - `decision calibrate EXAMPLES [--model MODEL_ID] --output DIR`: fit a
//!   decision model's temperatures to labelled examples and write a
//!   calibrated copy of the model to DIR.
//! - `decision train EXAMPLES [--model MODEL_ID] --output DIR`: train the
//!   model's final layers on labelled examples and write the trained copy
//!   to DIR.

use std::ffi::OsString;
use std::path::PathBuf;

use crate::arguments::UsageError;

const MODEL_FLAG: &str = "--model";
const OUTPUT_FLAG: &str = "--output";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecisionCommand {
    Calibrate(DecisionToolArguments),
    Train(DecisionToolArguments),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionToolArguments {
    /// The labelled examples, one JSON object per line.
    pub examples_file_path: PathBuf,
    /// The decision model to start from; the default decision model when absent.
    pub model_id: Option<String>,
    /// Where the new copy is written; must not exist yet.
    pub output_directory: PathBuf,
}

/// Parses everything after `astronomical decision`.
pub fn parse_decision_command(
    remaining_arguments: &[OsString],
) -> Result<DecisionCommand, UsageError> {
    let Some(sub_verb) = remaining_arguments.first() else {
        return Err(UsageError::UnknownDecisionTarget(String::new()));
    };
    match sub_verb.to_string_lossy().as_ref() {
        "calibrate" => {
            parse_tool_arguments(&remaining_arguments[1..]).map(DecisionCommand::Calibrate)
        }
        "train" => parse_tool_arguments(&remaining_arguments[1..]).map(DecisionCommand::Train),
        unknown_target => Err(UsageError::UnknownDecisionTarget(unknown_target.to_owned())),
    }
}

fn parse_tool_arguments(arguments: &[OsString]) -> Result<DecisionToolArguments, UsageError> {
    let mut examples_file_path = None;
    let mut model_id = None;
    let mut output_directory = None;
    let mut remaining_arguments = arguments.iter();
    while let Some(argument) = remaining_arguments.next() {
        let argument_text = argument.to_string_lossy().into_owned();
        match argument_text.as_str() {
            MODEL_FLAG => {
                let model_text = flag_value(remaining_arguments.next(), MODEL_FLAG)?;
                set_once(&mut model_id, model_text, MODEL_FLAG)?;
            }
            OUTPUT_FLAG => {
                let output_text = flag_value(remaining_arguments.next(), OUTPUT_FLAG)?;
                set_once(
                    &mut output_directory,
                    PathBuf::from(output_text),
                    OUTPUT_FLAG,
                )?;
            }
            _ if argument_text.starts_with('-') || examples_file_path.is_some() => {
                return Err(UsageError::UnknownArgument(argument_text));
            }
            _ => examples_file_path = Some(PathBuf::from(argument_text)),
        }
    }
    let Some(examples_file_path) = examples_file_path else {
        return Err(UsageError::DecisionCalibrateExamplesRequired);
    };
    let Some(output_directory) = output_directory else {
        return Err(UsageError::MissingValue(OUTPUT_FLAG));
    };
    Ok(DecisionToolArguments {
        examples_file_path,
        model_id,
        output_directory,
    })
}

fn flag_value(
    next_argument: Option<&OsString>,
    flag_name: &'static str,
) -> Result<String, UsageError> {
    let Some(next_argument) = next_argument else {
        return Err(UsageError::MissingValue(flag_name));
    };
    let value_text = next_argument.to_string_lossy().into_owned();
    if value_text.is_empty() || value_text.starts_with('-') {
        return Err(UsageError::MissingValue(flag_name));
    }
    Ok(value_text)
}

fn set_once<Value>(
    slot: &mut Option<Value>,
    value: Value,
    flag_name: &'static str,
) -> Result<(), UsageError> {
    if slot.is_some() {
        return Err(UsageError::RepeatedArgument(flag_name));
    }
    *slot = Some(value);
    Ok(())
}

#[cfg(test)]
#[path = "decision_arguments_tests.rs"]
mod tests;

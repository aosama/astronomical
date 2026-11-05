//! Arguments for `astronomical vision`, an in-process verb with three sub-verbs.
//!
//! - `vision inspect MODEL [--json]`: whether a model has a vision tower.
//! - `vision match TEXT_MODEL TOWER_MODEL [--json]`: whether a tower shares a text
//!   model's base.
//! - `vision add TEXT_MODEL --tower TOWER_MODEL --output DIR [--json]`: build a new
//!   model from the two. The match runs inside it and cannot be skipped.
//!
//! A model is `organization/name[@revision]` or a local directory. Models stay
//! text here, because telling those apart means looking at the filesystem, and
//! that failure belongs with the verb rather than with the grammar.

use std::ffi::OsString;
use std::path::PathBuf;

use crate::arguments::UsageError;

const JSON_FLAG: &str = "--json";
const TOWER_FLAG: &str = "--tower";
const OUTPUT_FLAG: &str = "--output";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VisionCommand {
    Inspect(VisionInspectArguments),
    Match(VisionMatchArguments),
    Add(VisionAddArguments),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionInspectArguments {
    pub model: String,
    pub render_json: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionMatchArguments {
    pub text_model: String,
    pub tower_model: String,
    pub render_json: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionAddArguments {
    pub text_model: String,
    pub tower_model: String,
    pub output_directory: PathBuf,
    pub render_json: bool,
}

/// Parses everything after `astronomical vision`.
pub fn parse_vision_command(remaining_arguments: &[OsString]) -> Result<VisionCommand, UsageError> {
    let Some(sub_verb) = remaining_arguments.first() else {
        return Err(UsageError::MissingCommand);
    };
    let sub_verb_arguments = &remaining_arguments[1..];
    match sub_verb.to_string_lossy().as_ref() {
        "inspect" => parse_inspect(sub_verb_arguments).map(VisionCommand::Inspect),
        "match" => parse_match(sub_verb_arguments).map(VisionCommand::Match),
        "add" => parse_add(sub_verb_arguments).map(VisionCommand::Add),
        unknown_target => Err(UsageError::UnknownVisionTarget(unknown_target.to_owned())),
    }
}

fn parse_inspect(arguments: &[OsString]) -> Result<VisionInspectArguments, UsageError> {
    let (mut model_texts, render_json) = split_models_and_json(arguments, 1)?;
    let Some(model) = model_texts.pop() else {
        return Err(UsageError::MissingValue("MODEL"));
    };
    Ok(VisionInspectArguments { model, render_json })
}

fn parse_match(arguments: &[OsString]) -> Result<VisionMatchArguments, UsageError> {
    let (mut model_texts, render_json) = split_models_and_json(arguments, 2)?;
    let tower_model = model_texts.pop();
    let text_model = model_texts.pop();
    match (text_model, tower_model) {
        (Some(text_model), Some(tower_model)) => Ok(VisionMatchArguments {
            text_model,
            tower_model,
            render_json,
        }),
        (None, Some(_)) => Err(UsageError::MissingValue("TOWER_MODEL")),
        _ => Err(UsageError::MissingValue("TEXT_MODEL")),
    }
}

fn parse_add(arguments: &[OsString]) -> Result<VisionAddArguments, UsageError> {
    let mut text_model = None;
    let mut tower_model = None;
    let mut output_directory = None;
    let mut render_json = false;
    let mut remaining_arguments = arguments.iter();
    while let Some(argument) = remaining_arguments.next() {
        let argument_text = argument.to_string_lossy().into_owned();
        match argument_text.as_str() {
            JSON_FLAG if render_json => return Err(UsageError::RepeatedArgument(JSON_FLAG)),
            JSON_FLAG => render_json = true,
            TOWER_FLAG => {
                let tower_text = flag_value(remaining_arguments.next(), TOWER_FLAG)?;
                set_once(&mut tower_model, tower_text, TOWER_FLAG)?;
            }
            OUTPUT_FLAG => {
                let output_text = flag_value(remaining_arguments.next(), OUTPUT_FLAG)?;
                set_once(
                    &mut output_directory,
                    PathBuf::from(output_text),
                    OUTPUT_FLAG,
                )?;
            }
            _ if argument_text.starts_with('-') || text_model.is_some() => {
                return Err(UsageError::UnknownArgument(argument_text));
            }
            _ => text_model = Some(argument_text),
        }
    }
    let Some(text_model) = text_model else {
        return Err(UsageError::MissingValue("TEXT_MODEL"));
    };
    let Some(tower_model) = tower_model else {
        return Err(UsageError::MissingValue(TOWER_FLAG));
    };
    let Some(output_directory) = output_directory else {
        return Err(UsageError::MissingValue(OUTPUT_FLAG));
    };
    Ok(VisionAddArguments {
        text_model,
        tower_model,
        output_directory,
        render_json,
    })
}

fn flag_value(
    next_argument: Option<&OsString>,
    flag_name: &'static str,
) -> Result<String, UsageError> {
    match next_argument {
        Some(next_argument) => {
            let value_text = next_argument.to_string_lossy().into_owned();
            if value_text.is_empty() || value_text.starts_with('-') {
                return Err(UsageError::MissingValue(flag_name));
            }
            Ok(value_text)
        }
        None => Err(UsageError::MissingValue(flag_name)),
    }
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

fn split_models_and_json(
    arguments: &[OsString],
    maximum_models: usize,
) -> Result<(Vec<String>, bool), UsageError> {
    let mut model_texts = Vec::with_capacity(maximum_models);
    let mut render_json = false;
    for argument in arguments {
        let argument_text = argument.to_string_lossy().into_owned();
        if argument_text == JSON_FLAG {
            if render_json {
                return Err(UsageError::RepeatedArgument(JSON_FLAG));
            }
            render_json = true;
        } else if argument_text.starts_with('-') || model_texts.len() == maximum_models {
            return Err(UsageError::UnknownArgument(argument_text));
        } else {
            model_texts.push(argument_text);
        }
    }
    Ok((model_texts, render_json))
}

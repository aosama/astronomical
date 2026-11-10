//! `astronomical decide [STATE | --state-file PATH] (--ask QUESTION |
//! --questions JSON | --questions-file PATH) [--json-state] [--model
//! MODEL_ID]`, kept beside the command tree so `arguments.rs` stays a
//! dispatcher of verbs.

use std::{ffi::OsString, path::PathBuf};

use super::{UsageError, flag_value};
use crate::verbs::decide::{DecideArguments, DecideQuestionsSource};

/// The values `decide` collects before it checks them against each other.
/// File paths stay text here so every flag that takes a value is stored the
/// same way; they become paths once the invocation is known to be coherent.
#[derive(Default)]
struct CollectedDecideArguments {
    state_text: Option<String>,
    state_file_path_text: Option<String>,
    state_is_json: bool,
    ask: Option<String>,
    questions_json: Option<String>,
    questions_file_path_text: Option<String>,
    model_id: Option<String>,
}

pub(super) fn parse_decide_arguments(
    remaining_arguments: &[OsString],
) -> Result<DecideArguments, UsageError> {
    let mut collected_arguments = CollectedDecideArguments::default();
    let mut argument_index = 0;
    while argument_index < remaining_arguments.len() {
        let argument = &remaining_arguments[argument_index];
        if argument == "--json-state" {
            collected_arguments.state_is_json = true;
            argument_index += 1;
            continue;
        }
        if let Some((flag_name, flag_slot)) = value_flag_slot(&mut collected_arguments, argument) {
            if flag_slot.is_some() {
                return Err(UsageError::RepeatedArgument(flag_name));
            }
            *flag_slot = Some(flag_value(
                remaining_arguments,
                argument_index + 1,
                flag_name,
            )?);
            argument_index += 2;
            continue;
        }
        take_positional_state(&mut collected_arguments, argument)?;
        argument_index += 1;
    }
    decide_arguments_from(collected_arguments)
}

/// Where the value of `argument` is kept, when `argument` is one of the
/// flags that takes a value.
fn value_flag_slot<'collected>(
    collected_arguments: &'collected mut CollectedDecideArguments,
    argument: &OsString,
) -> Option<(&'static str, &'collected mut Option<String>)> {
    if argument == "--model" {
        Some(("--model", &mut collected_arguments.model_id))
    } else if argument == "--ask" {
        Some(("--ask", &mut collected_arguments.ask))
    } else if argument == "--questions" {
        Some(("--questions", &mut collected_arguments.questions_json))
    } else if argument == "--state-file" {
        Some((
            "--state-file",
            &mut collected_arguments.state_file_path_text,
        ))
    } else if argument == "--questions-file" {
        Some((
            "--questions-file",
            &mut collected_arguments.questions_file_path_text,
        ))
    } else {
        None
    }
}

/// Takes `argument` as the state. Anything that looks like a flag, and any
/// second positional argument, is an argument `decide` does not know.
fn take_positional_state(
    collected_arguments: &mut CollectedDecideArguments,
    argument: &OsString,
) -> Result<(), UsageError> {
    let Some(positional_text) = argument.to_str() else {
        return Err(UsageError::UnknownArgument(
            argument.to_string_lossy().into_owned(),
        ));
    };
    if positional_text.starts_with('-') || collected_arguments.state_text.is_some() {
        return Err(UsageError::UnknownArgument(positional_text.to_owned()));
    }
    collected_arguments.state_text = Some(positional_text.to_owned());
    Ok(())
}

/// Checks that the collected values name one state source at most and
/// exactly one questions source.
fn decide_arguments_from(
    collected_arguments: CollectedDecideArguments,
) -> Result<DecideArguments, UsageError> {
    if collected_arguments.state_text.is_some()
        && collected_arguments.state_file_path_text.is_some()
    {
        return Err(UsageError::DecideStateConflict);
    }
    let questions_source = match (
        collected_arguments.ask,
        collected_arguments.questions_json,
        collected_arguments.questions_file_path_text,
    ) {
        (Some(question_text), None, None) => DecideQuestionsSource::Ask(question_text),
        (None, Some(questions_json), None) => DecideQuestionsSource::InlineJson(questions_json),
        (None, None, Some(file_path_text)) => {
            DecideQuestionsSource::File(PathBuf::from(file_path_text))
        }
        (None, None, None) => return Err(UsageError::DecideQuestionsRequired),
        _ => return Err(UsageError::DecideQuestionsConflict),
    };
    // No state at all parses: the journey resolves stdin and fails at
    // runtime when even stdin is empty.
    Ok(DecideArguments {
        state_text: collected_arguments.state_text,
        state_file_path: collected_arguments.state_file_path_text.map(PathBuf::from),
        state_is_json: collected_arguments.state_is_json,
        questions_source,
        model_id: collected_arguments.model_id,
    })
}

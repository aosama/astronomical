//! `astronomical embed TEXT | --file PATH [--model MODEL_ID]`, kept beside
//! the command tree so `arguments.rs` stays a dispatcher of verbs.

use std::{ffi::OsString, path::PathBuf};

use super::{UsageError, flag_value};
use crate::verbs::embed::EmbedArguments;

pub(super) fn parse_embed_arguments(
    remaining_arguments: &[OsString],
) -> Result<EmbedArguments, UsageError> {
    let mut text = None;
    let mut file_path = None;
    let mut model_id = None;
    let mut argument_index = 0;
    while argument_index < remaining_arguments.len() {
        let argument = &remaining_arguments[argument_index];
        if argument == "--model" {
            if model_id.is_some() {
                return Err(UsageError::RepeatedArgument("--model"));
            }
            model_id = Some(flag_value(
                remaining_arguments,
                argument_index + 1,
                "--model",
            )?);
            argument_index += 2;
            continue;
        }
        if argument == "--file" {
            if file_path.is_some() {
                return Err(UsageError::RepeatedArgument("--file"));
            }
            let file_value = flag_value(remaining_arguments, argument_index + 1, "--file")?;
            file_path = Some(PathBuf::from(file_value));
            argument_index += 2;
            continue;
        }
        if argument.to_string_lossy().starts_with('-') {
            return Err(UsageError::UnknownArgument(
                argument.to_string_lossy().into_owned(),
            ));
        }
        if text.is_some() {
            return Err(UsageError::UnknownArgument(
                argument.to_string_lossy().into_owned(),
            ));
        }
        let raw_text = argument
            .to_str()
            .ok_or_else(|| UsageError::UnknownArgument(argument.to_string_lossy().into_owned()))?;
        text = Some(raw_text.to_owned());
        argument_index += 1;
    }
    if text.is_some() && file_path.is_some() {
        return Err(UsageError::EmbedInputConflict);
    }
    // No input at all parses: the journey resolves stdin and fails at
    // runtime when even stdin is empty.
    Ok(EmbedArguments {
        text,
        file_path,
        model_id,
    })
}

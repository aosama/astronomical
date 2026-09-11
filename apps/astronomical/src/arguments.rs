//! Parses the user-facing `astronomical` command tree. Launch is the only
//! product verb in this slice.

use std::ffi::OsString;

use crate::errors::UsageError;

const HELP_TEXT: &str = concat!(
    "Astronomical launch\n\n",
    "Usage: astronomical launch [tool]\n",
    "       astronomical launch opencode [--model MODEL_ID]\n",
    "       astronomical --help\n",
    "       astronomical --version\n\n",
    "Launch a coding harness against the local Astronomical Library.\n\n",
    "Commands:\n",
    "  launch [tool]   Start a supported harness (OpenCode in this release)\n\n",
    "Options:\n",
    "  --model MODEL_ID   Library chat model to use when several exist and stdin is not a terminal\n",
    "  -v, --verbose      Print launch timings on stderr\n",
    "  -h, --help         Show this help\n",
    "  --version          Show the CLI version\n",
);

/// Parsed CLI invocation before any loopback work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliCommand {
    Help,
    Version,
    Launch(LaunchArguments),
}

/// Launch-specific arguments after `astronomical launch`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchArguments {
    pub tool_slug: Option<String>,
    pub model_id: Option<String>,
}

pub fn help_text() -> &'static str {
    HELP_TEXT
}

pub fn parse_command(
    process_arguments: impl IntoIterator<Item = OsString>,
) -> Result<CliCommand, UsageError> {
    let supplied_arguments = process_arguments.into_iter().skip(1).collect::<Vec<_>>();
    if supplied_arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        return Ok(CliCommand::Help);
    }
    if supplied_arguments
        .iter()
        .any(|argument| argument == "--version")
    {
        return Ok(CliCommand::Version);
    }
    let remaining_arguments = supplied_arguments
        .into_iter()
        .filter(|argument| argument != "--verbose" && argument != "-v")
        .collect::<Vec<_>>();
    if remaining_arguments.is_empty() {
        return Err(UsageError::MissingCommand);
    }

    let command_name = remaining_arguments[0].to_str().ok_or_else(|| {
        UsageError::UnknownCommand(remaining_arguments[0].to_string_lossy().into_owned())
    })?;
    if command_name != "launch" {
        return Err(UsageError::UnknownCommand(command_name.to_owned()));
    }

    let mut tool_slug = None;
    let mut model_id = None;
    let mut argument_index = 1;
    while argument_index < remaining_arguments.len() {
        let argument = &remaining_arguments[argument_index];
        if argument == "--model" {
            if model_id.is_some() {
                return Err(UsageError::RepeatedArgument("--model"));
            }
            let raw_model_id = remaining_arguments
                .get(argument_index + 1)
                .ok_or(UsageError::MissingValue("--model"))?
                .to_str()
                .ok_or(UsageError::MissingValue("--model"))?;
            if raw_model_id.is_empty() || raw_model_id.starts_with('-') {
                return Err(UsageError::MissingValue("--model"));
            }
            model_id = Some(raw_model_id.to_owned());
            argument_index += 2;
            continue;
        }
        if argument.to_string_lossy().starts_with('-') {
            return Err(UsageError::UnknownArgument(
                argument.to_string_lossy().into_owned(),
            ));
        }
        if tool_slug.is_some() {
            return Err(UsageError::MultipleTools);
        }
        let raw_tool_slug = argument
            .to_str()
            .ok_or_else(|| UsageError::UnknownArgument(argument.to_string_lossy().into_owned()))?;
        tool_slug = Some(raw_tool_slug.to_owned());
        argument_index += 1;
    }

    Ok(CliCommand::Launch(LaunchArguments {
        tool_slug,
        model_id,
    }))
}

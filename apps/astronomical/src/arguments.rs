//! Parses the user-facing `astronomical` command tree. Launch drives a
//! harness; schema and validate are ephemeral in-process verbs (issue #821).

use std::{ffi::OsString, path::PathBuf};

use crate::embed_arguments::EmbedArguments;
use crate::errors::UsageError;
use crate::models_arguments::ModelsCommand;
use crate::respond_arguments::RespondArguments;
use crate::schema_arguments::SchemaArguments;
use crate::validate_config_arguments::ValidateConfigArguments;

const HELP_TEXT: &str = concat!(
    "Astronomical\n\n",
    "Usage: astronomical launch [tool]\n",
    "       astronomical launch opencode [--model MODEL_ID]\n",
    "       astronomical respond PROMPT [--model MODEL_ID] [--no-stream]\n",
    "       astronomical embed [TEXT | --file PATH] [--model MODEL_ID]\n",
    "       astronomical models list | supported | default [MODEL_ID] | download MODEL_ID\n",
    "       astronomical status\n",
    "       astronomical schema object --name NAME (--string|--int|--double|--boolean) PROPERTY...\n",
    "       astronomical validate config [--instance stable|development] [--json]\n",
    "       astronomical --help\n",
    "       astronomical --version\n\n",
    "Launch a coding harness against the local Astronomical Library, manage\n",
    "the model library through the daemon, or run ephemeral in-process\n",
    "utilities against the instance configuration.\n\n",
    "Commands:\n",
    "  launch [tool]    Start a supported harness (OpenCode in this release)\n",
    "  respond PROMPT   One-shot chat answer; the daemon loads or downloads the\n",
    "                   model automatically when it is not resident yet\n",
    "  embed [TEXT]     One-shot embedding vector as one JSON document; the daemon\n",
    "                   loads or downloads the model automatically when needed\n",
    "  models list      Models installed on this Mac\n",
    "  models supported Release catalog: what can be downloaded, with local state\n",
    "  models default   Show the effective default model; with MODEL_ID, download it\n",
    "                   first when this Mac lacks it, then persist it\n",
    "  models download  Start (or resume) a download and wait, with live progress\n",
    "  status           Worker state, resident model, default model, active download\n",
    "  schema object    Build a strict JSON object schema for structured output\n",
    "  validate config  Report effective values of an instance configuration\n\n",
    "Options:\n",
    "  --model MODEL_ID   Model to use (default: the daemon's effective default model)\n",
    "  --no-stream        Print the finished respond answer once instead of streaming\n",
    "  --file PATH        Embed the file's contents instead of TEXT or stdin\n",
    "  --instance NAME    Which instance to inspect for validate config (default: development)\n",
    "  --json             Render the validate config report as JSON\n",
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
    Respond(RespondArguments),
    Embed(EmbedArguments),
    Models(ModelsCommand),
    Status,
    Schema(SchemaArguments),
    ValidateConfig(ValidateConfigArguments),
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
    match command_name {
        "launch" => parse_launch_arguments(&remaining_arguments[1..]).map(CliCommand::Launch),
        "respond" => parse_respond_arguments(&remaining_arguments[1..]).map(CliCommand::Respond),
        "embed" => parse_embed_arguments(&remaining_arguments[1..]).map(CliCommand::Embed),
        "models" => parse_models_command(&remaining_arguments[1..]).map(CliCommand::Models),
        "status" => {
            if remaining_arguments.len() > 1 {
                return Err(UsageError::UnknownArgument(
                    remaining_arguments[1].to_string_lossy().into_owned(),
                ));
            }
            Ok(CliCommand::Status)
        }
        "schema" => parse_schema_command(&remaining_arguments[1..]).map(CliCommand::Schema),
        "validate" => {
            parse_validate_command(&remaining_arguments[1..]).map(CliCommand::ValidateConfig)
        }
        other => Err(UsageError::UnknownCommand(other.to_owned())),
    }
}

fn parse_schema_command(
    remaining_arguments: &[OsString],
) -> Result<crate::schema_arguments::SchemaArguments, UsageError> {
    let schema_target = remaining_arguments
        .first()
        .and_then(|argument| argument.to_str())
        .ok_or(UsageError::MissingCommand)?;
    if schema_target != "object" {
        return Err(UsageError::UnknownSchemaTarget(schema_target.to_owned()));
    }
    crate::schema_arguments::parse_schema_arguments(&remaining_arguments[1..])
}

fn parse_validate_command(
    remaining_arguments: &[OsString],
) -> Result<crate::validate_config_arguments::ValidateConfigArguments, UsageError> {
    let validate_target = remaining_arguments
        .first()
        .and_then(|argument| argument.to_str())
        .ok_or(UsageError::MissingCommand)?;
    if validate_target != "config" {
        return Err(UsageError::UnknownValidateTarget(
            validate_target.to_owned(),
        ));
    }
    crate::validate_config_arguments::parse_validate_config_arguments(&remaining_arguments[1..])
}

fn parse_models_command(remaining_arguments: &[OsString]) -> Result<ModelsCommand, UsageError> {
    let subcommand = remaining_arguments
        .first()
        .and_then(|argument| argument.to_str())
        .ok_or(UsageError::ModelsSubcommandRequired)?;
    let trailing_arguments = &remaining_arguments[1..];
    match subcommand {
        "list" => {
            reject_trailing_arguments(trailing_arguments)?;
            Ok(ModelsCommand::List)
        }
        "supported" => {
            reject_trailing_arguments(trailing_arguments)?;
            Ok(ModelsCommand::Supported)
        }
        "default" => {
            let model_id = match trailing_arguments.first() {
                Some(raw_model_id) if !raw_model_id.to_string_lossy().starts_with('-') => {
                    if trailing_arguments.len() > 1 {
                        return Err(UsageError::UnknownArgument(
                            trailing_arguments[1].to_string_lossy().into_owned(),
                        ));
                    }
                    Some(parse_positional_model_id(raw_model_id)?)
                }
                Some(raw_flag) => {
                    return Err(UsageError::UnknownArgument(
                        raw_flag.to_string_lossy().into_owned(),
                    ));
                }
                None => None,
            };
            Ok(ModelsCommand::Default { model_id })
        }
        "download" => {
            if trailing_arguments.is_empty() {
                return Err(UsageError::ModelsDownloadModelRequired);
            }
            let raw_model_id = &trailing_arguments[0];
            if raw_model_id.to_string_lossy().starts_with('-') {
                return Err(UsageError::UnknownArgument(
                    raw_model_id.to_string_lossy().into_owned(),
                ));
            }
            if trailing_arguments.len() > 1 {
                return Err(UsageError::UnknownArgument(
                    trailing_arguments[1].to_string_lossy().into_owned(),
                ));
            }
            Ok(ModelsCommand::Download {
                model_id: parse_positional_model_id(raw_model_id)?,
            })
        }
        other => Err(UsageError::UnknownModelsSubcommand(other.to_owned())),
    }
}

fn reject_trailing_arguments(trailing_arguments: &[OsString]) -> Result<(), UsageError> {
    if let Some(extra_argument) = trailing_arguments.first() {
        return Err(UsageError::UnknownArgument(
            extra_argument.to_string_lossy().into_owned(),
        ));
    }
    Ok(())
}

fn parse_positional_model_id(raw_model_id: &OsString) -> Result<String, UsageError> {
    let model_id = raw_model_id
        .to_str()
        .ok_or(UsageError::MissingValue("MODEL_ID"))?;
    if model_id.is_empty() {
        return Err(UsageError::MissingValue("MODEL_ID"));
    }
    Ok(model_id.to_owned())
}

/// Reads the value that must follow a value-taking flag. A missing,
/// invalid, empty, or flag-shaped value is the same usage failure: the
/// flag's value is what is missing.
fn flag_value(
    remaining_arguments: &[OsString],
    value_index: usize,
    flag_name: &'static str,
) -> Result<String, UsageError> {
    let raw_value = remaining_arguments
        .get(value_index)
        .ok_or(UsageError::MissingValue(flag_name))?
        .to_str()
        .ok_or(UsageError::MissingValue(flag_name))?;
    if raw_value.is_empty() || raw_value.starts_with('-') {
        return Err(UsageError::MissingValue(flag_name));
    }
    Ok(raw_value.to_owned())
}

fn parse_launch_arguments(remaining_arguments: &[OsString]) -> Result<LaunchArguments, UsageError> {
    let mut tool_slug = None;
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

    Ok(LaunchArguments {
        tool_slug,
        model_id,
    })
}

fn parse_respond_arguments(
    remaining_arguments: &[OsString],
) -> Result<RespondArguments, UsageError> {
    let mut prompt = None;
    let mut model_id = None;
    let mut no_stream = false;
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
        if argument == "--no-stream" {
            if no_stream {
                return Err(UsageError::RepeatedArgument("--no-stream"));
            }
            no_stream = true;
            argument_index += 1;
            continue;
        }
        if argument.to_string_lossy().starts_with('-') {
            return Err(UsageError::UnknownArgument(
                argument.to_string_lossy().into_owned(),
            ));
        }
        if prompt.is_some() {
            return Err(UsageError::UnknownArgument(
                argument.to_string_lossy().into_owned(),
            ));
        }
        let raw_prompt = argument
            .to_str()
            .ok_or_else(|| UsageError::UnknownArgument(argument.to_string_lossy().into_owned()))?;
        prompt = Some(raw_prompt.to_owned());
        argument_index += 1;
    }

    Ok(RespondArguments {
        prompt: prompt.ok_or(UsageError::RespondPromptRequired)?,
        model_id,
        no_stream,
    })
}

fn parse_embed_arguments(remaining_arguments: &[OsString]) -> Result<EmbedArguments, UsageError> {
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

//! Arguments for `astronomical speak [TEXT | --file PATH] [--voice NAME]
//! [--speed N] [--model ID] [--seed N] [--output PATH]`.

use std::ffi::OsString;
use std::path::PathBuf;

use astronomical_speech_contract::{FASTEST_SPEED, SLOWEST_SPEED};

use crate::arguments::{UsageError, flag_value};

/// Parsed `astronomical speak` invocation before any IPC work.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakArguments {
    /// The text as a positional argument; `None` reads `--file` or stdin.
    pub text: Option<String>,
    pub text_file: Option<PathBuf>,
    /// Exact speech model to demand; otherwise the daemon's choice.
    pub model_id: Option<String>,
    pub voice_name: Option<String>,
    /// 1.0, the voice's natural pace, when absent.
    pub speed: f32,
    /// Reproducibility seed; the daemon picks one and reports it when absent.
    pub seed: Option<u64>,
    /// Where the audio goes; a timestamped WAV in the working directory when absent.
    pub output_path: Option<PathBuf>,
}

// The parser only ever stores a finite pace from 0.25 to 4, so equality on
// it is total; the command enum the arguments sit in compares with `Eq`.
impl Eq for SpeakArguments {}

pub fn parse_speak_arguments(
    remaining_arguments: &[OsString],
) -> Result<SpeakArguments, UsageError> {
    let mut speak_arguments = SpeakArguments {
        text: None,
        text_file: None,
        model_id: None,
        voice_name: None,
        speed: 1.0,
        seed: None,
        output_path: None,
    };
    let mut is_speed_given = false;
    let mut argument_index = 0;
    while argument_index < remaining_arguments.len() {
        let argument = &remaining_arguments[argument_index];
        let Some(argument_text) = argument.to_str() else {
            return Err(UsageError::UnknownArgument(
                argument.to_string_lossy().into_owned(),
            ));
        };
        let Some(flag_name) = value_flag_name(argument_text) else {
            set_text(&mut speak_arguments, argument_text)?;
            argument_index += 1;
            continue;
        };
        let raw_value = flag_value(remaining_arguments, argument_index + 1, flag_name)?;
        if flag_name == "--speed" {
            if is_speed_given {
                return Err(UsageError::RepeatedArgument(flag_name));
            }
            is_speed_given = true;
        }
        set_value_flag(&mut speak_arguments, flag_name, raw_value)?;
        argument_index += 2;
    }
    if speak_arguments.text.is_some() && speak_arguments.text_file.is_some() {
        return Err(UsageError::SpeakTextGivenTwice);
    }
    Ok(speak_arguments)
}

fn set_text(speak_arguments: &mut SpeakArguments, argument_text: &str) -> Result<(), UsageError> {
    // One text only: a second positional is almost always unquoted text.
    if argument_text.starts_with('-') || speak_arguments.text.is_some() {
        return Err(UsageError::UnknownArgument(argument_text.to_owned()));
    }
    speak_arguments.text = Some(argument_text.to_owned());
    Ok(())
}

fn set_value_flag(
    speak_arguments: &mut SpeakArguments,
    flag_name: &'static str,
    raw_value: String,
) -> Result<(), UsageError> {
    match flag_name {
        "--file" => set_once(
            &mut speak_arguments.text_file,
            flag_name,
            PathBuf::from(raw_value),
        ),
        "--model" => set_once(&mut speak_arguments.model_id, flag_name, raw_value),
        "--voice" => set_once(&mut speak_arguments.voice_name, flag_name, raw_value),
        "--seed" => {
            let seed = raw_value
                .parse::<u64>()
                .map_err(|_| UsageError::InvalidSeed(raw_value.clone()))?;
            set_once(&mut speak_arguments.seed, flag_name, seed)
        }
        "--speed" => {
            speak_arguments.speed = parse_speed(&raw_value)?;
            Ok(())
        }
        _ => set_once(
            &mut speak_arguments.output_path,
            flag_name,
            PathBuf::from(raw_value),
        ),
    }
}

/// The OpenAI speech API's range of paces, the same range the daemon checks.
fn parse_speed(raw_speed: &str) -> Result<f32, UsageError> {
    match raw_speed.parse::<f32>() {
        Ok(speed) if (SLOWEST_SPEED..=FASTEST_SPEED).contains(&speed) => Ok(speed),
        _ => Err(UsageError::InvalidSpeakSpeed(raw_speed.to_owned())),
    }
}

/// The flags that take a value, as the static names the usage errors carry.
fn value_flag_name(argument_text: &str) -> Option<&'static str> {
    match argument_text {
        "--file" => Some("--file"),
        "--model" => Some("--model"),
        "--voice" => Some("--voice"),
        "--speed" => Some("--speed"),
        "--seed" => Some("--seed"),
        "--output" => Some("--output"),
        _ => None,
    }
}

fn set_once<Value>(
    slot: &mut Option<Value>,
    flag_name: &'static str,
    value: Value,
) -> Result<(), UsageError> {
    if slot.is_some() {
        return Err(UsageError::RepeatedArgument(flag_name));
    }
    *slot = Some(value);
    Ok(())
}

#[cfg(test)]
#[path = "speak_arguments_tests.rs"]
mod tests;

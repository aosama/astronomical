//! Arguments for `astronomical image PROMPT [--size WIDTHxHEIGHT] [--model ID]
//! [--seed N] [--output PATH]`.
//!
//! The grammar carries only what the user chooses. The step count and
//! guidance scale belong to the model family, so they are not flags: a
//! mistuned step count quarter-denoises an image while every structural
//! check still passes, and the daemon's family default is the safe value.

use std::ffi::OsString;
use std::path::PathBuf;

use crate::arguments::{UsageError, flag_value};

/// Square, and inside every catalog model's envelope. The daemon still
/// checks the size against the chosen model and names any violation.
pub const DEFAULT_IMAGE_SIDE_PIXELS: u32 = 1_024;

/// Parsed `astronomical image` invocation before any IPC work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageArguments {
    /// What to draw, exactly as supplied.
    pub prompt: String,
    pub width_pixels: u32,
    pub height_pixels: u32,
    /// Exact image model to demand, when given; otherwise the built-in one.
    pub model_id: Option<String>,
    /// Reproducibility seed; the daemon picks one and reports it when absent.
    pub seed: Option<u64>,
    /// Where the PNG goes; a timestamped name in the working directory when
    /// absent.
    pub output_path: Option<PathBuf>,
}

pub fn parse_image_arguments(
    remaining_arguments: &[OsString],
) -> Result<ImageArguments, UsageError> {
    let mut partial_arguments = PartialImageArguments::default();
    let mut argument_index = 0;
    while argument_index < remaining_arguments.len() {
        argument_index = partial_arguments.consume(remaining_arguments, argument_index)?;
    }
    partial_arguments.finish()
}

#[derive(Default)]
struct PartialImageArguments {
    prompt: Option<String>,
    size_pixels: Option<(u32, u32)>,
    model_id: Option<String>,
    seed: Option<u64>,
    output_path: Option<PathBuf>,
}

impl PartialImageArguments {
    /// Consumes the flag or positional at `argument_index` and returns the
    /// index of the next argument.
    fn consume(
        &mut self,
        remaining_arguments: &[OsString],
        argument_index: usize,
    ) -> Result<usize, UsageError> {
        let argument = &remaining_arguments[argument_index];
        let Some(argument_text) = argument.to_str() else {
            return Err(UsageError::UnknownArgument(
                argument.to_string_lossy().into_owned(),
            ));
        };
        match value_flag_name(argument_text) {
            Some(flag_name) => {
                let raw_value = flag_value(remaining_arguments, argument_index + 1, flag_name)?;
                self.set_value_flag(flag_name, raw_value)?;
                Ok(argument_index + 2)
            }
            None if argument_text.starts_with('-') => {
                Err(UsageError::UnknownArgument(argument_text.to_owned()))
            }
            None => {
                self.set_prompt(argument_text)?;
                Ok(argument_index + 1)
            }
        }
    }

    fn set_value_flag(
        &mut self,
        flag_name: &'static str,
        raw_value: String,
    ) -> Result<(), UsageError> {
        match flag_name {
            "--model" => set_once(&mut self.model_id, flag_name, raw_value),
            "--size" => set_once(
                &mut self.size_pixels,
                flag_name,
                parse_image_size(&raw_value)?,
            ),
            "--seed" => set_once(&mut self.seed, flag_name, parse_image_seed(&raw_value)?),
            _ => set_once(&mut self.output_path, flag_name, PathBuf::from(raw_value)),
        }
    }

    /// One prompt only, like `respond` and `embed`: a second positional is
    /// almost always an unquoted prompt.
    fn set_prompt(&mut self, prompt_text: &str) -> Result<(), UsageError> {
        if self.prompt.is_some() {
            return Err(UsageError::UnknownArgument(prompt_text.to_owned()));
        }
        self.prompt = Some(prompt_text.to_owned());
        Ok(())
    }

    fn finish(self) -> Result<ImageArguments, UsageError> {
        let prompt = match self.prompt {
            Some(prompt) if !prompt.trim().is_empty() => prompt,
            _ => return Err(UsageError::ImagePromptRequired),
        };
        let (width_pixels, height_pixels) = self
            .size_pixels
            .unwrap_or((DEFAULT_IMAGE_SIDE_PIXELS, DEFAULT_IMAGE_SIDE_PIXELS));
        Ok(ImageArguments {
            prompt,
            width_pixels,
            height_pixels,
            model_id: self.model_id,
            seed: self.seed,
            output_path: self.output_path,
        })
    }
}

/// The flags that take a value, as the static names the usage errors carry.
fn value_flag_name(argument_text: &str) -> Option<&'static str> {
    match argument_text {
        "--model" => Some("--model"),
        "--size" => Some("--size"),
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

/// `1024x576` (or `1024X576`): two positive whole numbers. Whether the model
/// can render that size is the daemon's call, made against its own envelope.
pub(crate) fn parse_image_size(raw_size: &str) -> Result<(u32, u32), UsageError> {
    let Some((raw_width, raw_height)) = raw_size.split_once(['x', 'X']) else {
        return Err(UsageError::InvalidImageSize(raw_size.to_owned()));
    };
    match (raw_width.parse::<u32>(), raw_height.parse::<u32>()) {
        (Ok(width_pixels), Ok(height_pixels)) if width_pixels > 0 && height_pixels > 0 => {
            Ok((width_pixels, height_pixels))
        }
        _ => Err(UsageError::InvalidImageSize(raw_size.to_owned())),
    }
}

fn parse_image_seed(raw_seed: &str) -> Result<u64, UsageError> {
    match raw_seed.parse::<u64>() {
        Ok(seed) => Ok(seed),
        Err(_parse_error) => Err(UsageError::InvalidSeed(raw_seed.to_owned())),
    }
}

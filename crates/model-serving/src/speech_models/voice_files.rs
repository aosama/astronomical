//! A voice file in a folder of `<voice name>.safetensors` files, as both
//! families keep their voice catalogs.
//!
//! A voice name follows the contract's `is_plain_voice_name`, the rule
//! discovery lists voices by, so a request can never name a path outside the
//! folder. Listing the voices is discovery's job (`astronomical_config`).

use std::path::{Path, PathBuf};

use astronomical_speech_contract::is_plain_voice_name;

use super::speech_model_error::SpeechModelError;

const VOICE_FILE_EXTENSION: &str = "safetensors";

/// The file of `voice_name` in `voices_directory`.
pub(crate) fn catalog_voice_path(
    voices_directory: &Path,
    voice_name: &str,
) -> Result<PathBuf, SpeechModelError> {
    if !is_plain_voice_name(voice_name) {
        return Err(SpeechModelError::Input(format!(
            "{voice_name:?} is not a voice name; names are letters, digits, hyphens and underscores"
        )));
    }
    Ok(voice_file_path_in(voices_directory, voice_name))
}

fn voice_file_path_in(voices_directory: &Path, voice_name: &str) -> PathBuf {
    voices_directory.join(format!("{voice_name}.{VOICE_FILE_EXTENSION}"))
}

#[cfg(test)]
#[path = "voice_files_tests.rs"]
mod tests;

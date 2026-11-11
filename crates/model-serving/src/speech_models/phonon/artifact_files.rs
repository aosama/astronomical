//! Where a Phonon artifact keeps its files. The published checkpoint ships
//! one voice, `default-voice.safetensors`, offered as `default`; more voices
//! (Gradium's catalog) go in `voices/`.

use std::path::{Path, PathBuf};

use crate::speech_models::speech_model_error::SpeechModelError;
use crate::speech_models::voice_files::{catalog_voice_names, catalog_voice_path};

pub(crate) const PHONON_CONFIGURATION_FILE_NAME: &str = "config.json";
pub(crate) const PHONON_WEIGHTS_FILE_NAME: &str = "model.safetensors";
pub(crate) const PHONON_TOKENIZER_FILE_NAME: &str = "tokenizer.json";
pub(crate) const PHONON_DEFAULT_VOICE_FILE_NAME: &str = "default-voice.safetensors";
pub(crate) const PHONON_DEFAULT_VOICE_NAME: &str = "default";
const PHONON_VOICES_DIRECTORY_NAME: &str = "voices";

/// The file of `voice_name`: the shipped voice for `default`, else a catalog voice.
pub(crate) fn voice_file_path(
    model_directory: &Path,
    voice_name: &str,
) -> Result<PathBuf, SpeechModelError> {
    if voice_name == PHONON_DEFAULT_VOICE_NAME {
        return Ok(model_directory.join(PHONON_DEFAULT_VOICE_FILE_NAME));
    }
    catalog_voice_path(
        &model_directory.join(PHONON_VOICES_DIRECTORY_NAME),
        voice_name,
    )
}

/// `default` when the artifact has its shipped voice, then the catalog, sorted.
pub(crate) fn voice_names(model_directory: &Path) -> Vec<String> {
    let mut voice_names = Vec::new();
    if model_directory
        .join(PHONON_DEFAULT_VOICE_FILE_NAME)
        .is_file()
    {
        voice_names.push(PHONON_DEFAULT_VOICE_NAME.to_owned());
    }
    voice_names.extend(catalog_voice_names(
        &model_directory.join(PHONON_VOICES_DIRECTORY_NAME),
    ));
    voice_names
}

#[cfg(test)]
#[path = "artifact_files_tests.rs"]
mod tests;

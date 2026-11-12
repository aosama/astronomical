//! Where a Phonon artifact keeps its files. Discovery owns the layout and
//! which voices are offered; this reads a voice the request named. The
//! shipped voice, `default-voice.safetensors`, is offered as `default`.

use std::path::{Path, PathBuf};

use crate::speech_models::speech_model_error::SpeechModelError;
use crate::speech_models::voice_files::catalog_voice_path;

pub(crate) use astronomical_config::{
    PHONON_CONFIGURATION_FILE_NAME, PHONON_TOKENIZER_FILE_NAME, PHONON_WEIGHTS_FILE_NAME,
};
use astronomical_config::{
    PHONON_DEFAULT_VOICE_FILE_NAME, PHONON_DEFAULT_VOICE_NAME, PHONON_VOICES_DIRECTORY_NAME,
};

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

#[cfg(test)]
#[path = "artifact_files_tests.rs"]
mod tests;

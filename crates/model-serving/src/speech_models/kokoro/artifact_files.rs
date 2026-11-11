//! Where a Kokoro artifact keeps its files: its configuration, its weights,
//! and one file per voice in `voices/`.

use std::path::{Path, PathBuf};

use crate::speech_models::speech_model_error::SpeechModelError;
use crate::speech_models::voice_files::{catalog_voice_names, catalog_voice_path};

pub(crate) const KOKORO_CONFIGURATION_FILE_NAME: &str = "config.json";
pub(crate) const KOKORO_WEIGHTS_FILE_NAME: &str = "kokoro-v1_0.safetensors";
const KOKORO_VOICES_DIRECTORY_NAME: &str = "voices";

/// The file of `voice_name`, such as `af_heart`.
pub(crate) fn voice_file_path(
    model_directory: &Path,
    voice_name: &str,
) -> Result<PathBuf, SpeechModelError> {
    catalog_voice_path(
        &model_directory.join(KOKORO_VOICES_DIRECTORY_NAME),
        voice_name,
    )
}

/// The voices the artifact offers, sorted.
pub(crate) fn voice_names(model_directory: &Path) -> Vec<String> {
    catalog_voice_names(&model_directory.join(KOKORO_VOICES_DIRECTORY_NAME))
}

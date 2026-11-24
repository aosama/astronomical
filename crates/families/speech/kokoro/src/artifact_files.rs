//! Where a Kokoro artifact keeps its files. Discovery owns the layout and
//! which voices are offered; this reads a voice the request named.

use std::path::{Path, PathBuf};

use config::KOKORO_VOICES_DIRECTORY_NAME;
pub use config::{KOKORO_CONFIGURATION_FILE_NAME, KOKORO_WEIGHTS_FILE_NAME};

use speech::speech_models::speech_model_error::SpeechModelError;
use speech::speech_models::voice_files::catalog_voice_path;

/// The file of `voice_name`, such as `af_heart`.
pub fn voice_file_path(
    model_directory: &Path,
    voice_name: &str,
) -> Result<PathBuf, SpeechModelError> {
    catalog_voice_path(
        &model_directory.join(KOKORO_VOICES_DIRECTORY_NAME),
        voice_name,
    )
}

#[cfg(test)]
#[path = "artifact_files_tests.rs"]
mod tests;

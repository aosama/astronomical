//! A folder of voice files, `<voice name>.safetensors` each, as both families
//! keep their voice catalogs.
//!
//! A voice name is letters, digits, hyphens and underscores only. The same
//! rule decides which files are listed and which names are accepted, so a
//! voice that is advertised can always be loaded, and a request can never
//! name a path outside the folder.

use std::path::{Path, PathBuf};

use super::speech_model_error::SpeechModelError;

const VOICE_FILE_EXTENSION: &str = "safetensors";

pub(crate) fn is_plain_voice_name(voice_name: &str) -> bool {
    !voice_name.is_empty()
        && voice_name.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '-'
        })
}

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

/// The voice names in `voices_directory`, sorted; a file whose stem is not a
/// plain voice name is left out. A missing folder holds no voices.
pub(crate) fn catalog_voice_names(voices_directory: &Path) -> Vec<String> {
    let Ok(voice_entries) = std::fs::read_dir(voices_directory) else {
        return Vec::new();
    };
    let mut voice_names: Vec<String> = voice_entries
        .filter_map(Result::ok)
        .filter_map(|voice_entry| voice_name_of(&voice_entry.path()))
        .collect();
    voice_names.sort();
    voice_names
}

fn voice_name_of(voice_path: &Path) -> Option<String> {
    let is_voice_file = voice_path
        .extension()
        .is_some_and(|extension| extension == VOICE_FILE_EXTENSION);
    let stem = voice_path.file_stem()?.to_str()?;
    (is_voice_file && is_plain_voice_name(stem)).then(|| stem.to_owned())
}

fn voice_file_path_in(voices_directory: &Path, voice_name: &str) -> PathBuf {
    voices_directory.join(format!("{voice_name}.{VOICE_FILE_EXTENSION}"))
}

#[cfg(test)]
#[path = "voice_files_tests.rs"]
mod tests;

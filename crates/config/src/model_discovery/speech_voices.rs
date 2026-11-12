//! The voices a speech model folder holds: `<voice name>.safetensors` files.
//!
//! Only names the contract's `is_plain_voice_name` accepts are listed, the
//! same rule a request's voice is checked by, so every advertised voice can
//! be asked for and loaded.

use std::path::Path;

use astronomical_speech_contract::is_plain_voice_name;

const VOICE_FILE_EXTENSION: &str = "safetensors";
/// More voice files than any published model ships; a folder past it is not
/// listed further, so discovery stays bounded on an unexpected folder.
const MAXIMUM_LISTED_VOICES: usize = 256;

/// The voice names in `voices_directory`, sorted. A missing folder holds none.
pub(super) fn catalog_voice_names(voices_directory: &Path) -> Vec<String> {
    let Ok(voice_entries) = std::fs::read_dir(voices_directory) else {
        return Vec::new();
    };
    let mut voice_names = Vec::new();
    for voice_entry in voice_entries.flatten().take(MAXIMUM_LISTED_VOICES) {
        if let Some(voice_name) = voice_name_of(&voice_entry.path()) {
            voice_names.push(voice_name);
        }
    }
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

#[cfg(test)]
#[path = "speech_voices_tests.rs"]
mod tests;

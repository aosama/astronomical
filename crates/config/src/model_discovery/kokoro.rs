//! Shallow discovery rules for Kokoro speech-model folders.
//!
//! A Kokoro folder (`mlx-community/Kokoro-82M-bf16`) has a `config.json`
//! without a `model_type`; what marks it is the StyleTTS2 configuration's
//! `istftnet`, `plbert` and `vocab` sections beside the
//! `kokoro-v1_0.safetensors` weights, with one file per voice in `voices/`.

use std::path::Path;

use super::bounded_artifact_file;
use super::speech_voices::catalog_voice_names;
use super::{ModelFamily, SpeechModelCapabilities};

pub const KOKORO_CONFIGURATION_FILE_NAME: &str = "config.json";
pub const KOKORO_WEIGHTS_FILE_NAME: &str = "kokoro-v1_0.safetensors";
pub const KOKORO_VOICES_DIRECTORY_NAME: &str = "voices";
/// iSTFTNet's output rate, fixed by the vocoder's upsampling.
const KOKORO_SAMPLE_RATE_HERTZ: u32 = 24_000;
/// The voice listeners rated highest among Kokoro's, the default when present.
const PREFERRED_DEFAULT_VOICE_NAME: &str = "af_heart";
/// Kokoro's American English voices, female and male. Its other voices
/// expect British English or other languages, which the American English
/// phonemizer would speak with the wrong sounds, so they are not offered.
const AMERICAN_ENGLISH_VOICE_PREFIXES: [&str; 2] = ["af_", "am_"];
const MAXIMUM_CONFIGURATION_BYTES: u64 = 1024 * 1024;
const MARKER_SECTIONS: [&str; 3] = ["istftnet", "plbert", "vocab"];

/// Whether `model_directory` holds Kokoro weights and its configuration.
pub(super) fn classifies_directory(model_directory: &Path) -> bool {
    if !model_directory.join(KOKORO_WEIGHTS_FILE_NAME).is_file() {
        return false;
    }
    bounded_artifact_file::read_json::<serde_json::Value>(
        &model_directory.join(KOKORO_CONFIGURATION_FILE_NAME),
        MAXIMUM_CONFIGURATION_BYTES,
    )
    .is_ok_and(|configuration| is_kokoro_configuration(&configuration))
}

/// Whether a parsed `config.json` is Kokoro's: the rule both disk discovery
/// and the download preflight classify by.
#[must_use]
pub fn is_kokoro_configuration(configuration: &serde_json::Value) -> bool {
    MARKER_SECTIONS.iter().all(|section_name| {
        configuration
            .get(section_name)
            .is_some_and(serde_json::Value::is_object)
    })
}

/// What a Kokoro folder offers, or `None` when it has no voice to speak in.
#[must_use]
pub fn kokoro_speech_capabilities(model_directory: &Path) -> Option<SpeechModelCapabilities> {
    let mut voice_names = catalog_voice_names(&model_directory.join(KOKORO_VOICES_DIRECTORY_NAME));
    voice_names.retain(|voice_name| is_american_english_voice(voice_name));
    let default_voice_name = if voice_names
        .iter()
        .any(|voice_name| voice_name == PREFERRED_DEFAULT_VOICE_NAME)
    {
        PREFERRED_DEFAULT_VOICE_NAME.to_owned()
    } else {
        voice_names.first()?.clone()
    };
    Some(SpeechModelCapabilities {
        model_family: ModelFamily::Kokoro,
        sample_rate_hertz: KOKORO_SAMPLE_RATE_HERTZ,
        voice_names,
        default_voice_name,
        supports_speed: true,
    })
}

fn is_american_english_voice(voice_name: &str) -> bool {
    AMERICAN_ENGLISH_VOICE_PREFIXES
        .iter()
        .any(|voice_prefix| voice_name.starts_with(voice_prefix))
}

#[cfg(test)]
#[path = "kokoro_tests.rs"]
mod tests;

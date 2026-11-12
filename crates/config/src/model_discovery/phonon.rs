//! Shallow discovery rules for Phonon speech-model folders.
//!
//! A Phonon folder (`gradium/phonon-31m-en`) has a `config.json` without a
//! `model_type`; what marks it is the flow language model's `flow_lm` and
//! the Mimi decoder's `mimi` sections beside `model.safetensors`, the
//! tokenizer and the one voice it ships, `default-voice.safetensors`. More
//! voices, from Gradium's catalog, go in `voices/`.

use std::path::Path;

use super::bounded_artifact_file;
use super::speech_voices::catalog_voice_names;
use super::{ModelFamily, SpeechModelCapabilities};

pub const PHONON_CONFIGURATION_FILE_NAME: &str = "config.json";
pub const PHONON_WEIGHTS_FILE_NAME: &str = "model.safetensors";
pub const PHONON_TOKENIZER_FILE_NAME: &str = "tokenizer.json";
pub const PHONON_DEFAULT_VOICE_FILE_NAME: &str = "default-voice.safetensors";
/// The name the shipped voice is offered under.
pub const PHONON_DEFAULT_VOICE_NAME: &str = "default";
pub const PHONON_VOICES_DIRECTORY_NAME: &str = "voices";
const MAXIMUM_CONFIGURATION_BYTES: u64 = 1024 * 1024;

/// Whether `model_directory` holds a complete Phonon checkpoint.
pub(super) fn classifies_directory(model_directory: &Path) -> bool {
    for required_file_name in [
        PHONON_WEIGHTS_FILE_NAME,
        PHONON_TOKENIZER_FILE_NAME,
        PHONON_DEFAULT_VOICE_FILE_NAME,
    ] {
        if !model_directory.join(required_file_name).is_file() {
            return false;
        }
    }
    phonon_sample_rate_hertz(model_directory).is_some()
}

/// What a Phonon folder offers: its shipped voice first, then the catalog.
#[must_use]
pub fn phonon_speech_capabilities(model_directory: &Path) -> Option<SpeechModelCapabilities> {
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
    Some(SpeechModelCapabilities {
        model_family: ModelFamily::Phonon,
        sample_rate_hertz: phonon_sample_rate_hertz(model_directory)?,
        default_voice_name: voice_names.first()?.clone(),
        voice_names,
        // Phonon has a duration conditioner, but its effect on pace is not
        // verified, so only the natural pace is offered.
        supports_speed: false,
    })
}

/// The Mimi decoder's output rate, which also proves both marker sections.
fn phonon_sample_rate_hertz(model_directory: &Path) -> Option<u32> {
    let configuration = bounded_artifact_file::read_json::<serde_json::Value>(
        &model_directory.join(PHONON_CONFIGURATION_FILE_NAME),
        MAXIMUM_CONFIGURATION_BYTES,
    )
    .ok()?;
    configured_sample_rate_hertz(&configuration)
}

/// Whether a parsed `config.json` is Phonon's: the rule both disk discovery
/// and the download preflight classify by.
#[must_use]
pub fn is_phonon_configuration(configuration: &serde_json::Value) -> bool {
    configured_sample_rate_hertz(configuration).is_some()
}

fn configured_sample_rate_hertz(configuration: &serde_json::Value) -> Option<u32> {
    configuration.get("flow_lm")?.as_object()?;
    let sample_rate_hertz = configuration.get("mimi")?.get("sample_rate")?.as_u64()?;
    u32::try_from(sample_rate_hertz)
        .ok()
        .filter(|rate| *rate > 0)
}

#[cfg(test)]
#[path = "phonon_tests.rs"]
mod tests;

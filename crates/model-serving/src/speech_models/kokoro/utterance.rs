//! One Kokoro utterance and the checks it must pass before the model runs.

use super::configuration::KokoroConfiguration;
use super::voice_pack::KokoroVoice;
use crate::speech_models::speech_model_error::SpeechModelError;

/// Kokoro's position table has 512 entries, two of them for the markers.
pub(crate) const KOKORO_MAXIMUM_PHONEMES: usize = 510;

/// One utterance to speak.
pub(crate) struct KokoroUtterance<'request> {
    pub(crate) phonemes: &'request str,
    pub(crate) voice: &'request KokoroVoice,
    /// 1.0 is the voice's own pace; 1.2 is a fifth faster.
    pub(crate) speed: f32,
    /// Seeds the unvoiced-sound noise, so the same request sounds the same.
    /// `None` speaks without the noise, for comparison with the reference.
    pub(crate) noise_seed: Option<u64>,
}

/// The token ids to read, after checking the utterance: 1 to 510 phonemes,
/// at least one of them in the vocabulary (otherwise the model would run on
/// the markers alone and make a short noise), and a positive finite speed.
pub(crate) fn validated_token_ids(
    configuration: &KokoroConfiguration,
    phonemes: &str,
    speed: f32,
) -> Result<Vec<i32>, SpeechModelError> {
    let phoneme_count = phonemes.chars().count();
    if phoneme_count == 0 || phoneme_count > KOKORO_MAXIMUM_PHONEMES {
        return Err(SpeechModelError::Input(format!(
            "one pass reads 1 to {KOKORO_MAXIMUM_PHONEMES} phonemes; this utterance has {phoneme_count}"
        )));
    }
    if !(speed.is_finite() && speed > 0.0) {
        return Err(SpeechModelError::Input(format!(
            "speed must be a positive number, not {speed}"
        )));
    }
    let token_ids = configuration.token_ids(phonemes);
    let has_spoken_phoneme = token_ids.len() > 2;
    if !has_spoken_phoneme {
        return Err(SpeechModelError::Input(
            "none of the phonemes is in Kokoro's vocabulary".to_owned(),
        ));
    }
    Ok(token_ids)
}

#[cfg(test)]
#[path = "utterance_tests.rs"]
mod tests;

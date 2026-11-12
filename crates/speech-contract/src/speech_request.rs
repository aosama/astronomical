//! One speech request and the checks no model changes.
//!
//! Whether the voice exists, whether the model can change its pace and how
//! much text it takes at once depend on the loaded model; the engine checks
//! those against what it loaded.

use crate::speech_failure::SpeechFailure;

/// The most characters one request may ask to be spoken, as the OpenAI
/// speech API allows.
pub const MAXIMUM_SPEECH_INPUT_CHARACTERS: usize = 4096;
/// The OpenAI speech API's range of paces.
pub const SLOWEST_SPEED: f32 = 0.25;
pub const FASTEST_SPEED: f32 = 4.0;

#[derive(Clone, Debug, PartialEq)]
pub struct SpeechRequest {
    pub text: String,
    /// `None` for the model's default voice.
    pub voice_name: Option<String>,
    /// 1.0 is the voice's natural pace.
    pub speed: f32,
    /// The same seed, text, voice and pace give the same audio. The daemon
    /// picks one for a caller who names none and reports it, as it does for
    /// images, so any result can be reproduced.
    pub seed: u64,
}

impl SpeechRequest {
    pub fn validate(&self) -> Result<(), SpeechFailure> {
        let character_count = self.text.chars().count();
        if self.text.trim().is_empty() {
            return Err(SpeechFailure::invalid_request("there is no text to speak"));
        }
        if character_count > MAXIMUM_SPEECH_INPUT_CHARACTERS {
            return Err(SpeechFailure::invalid_request(format!(
                "the text has {character_count} characters; the most one request takes is {MAXIMUM_SPEECH_INPUT_CHARACTERS}"
            )));
        }
        if !(self.speed.is_finite() && (SLOWEST_SPEED..=FASTEST_SPEED).contains(&self.speed)) {
            return Err(SpeechFailure::invalid_request(format!(
                "speed {} is outside {SLOWEST_SPEED} to {FASTEST_SPEED}",
                self.speed
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "speech_request_tests.rs"]
mod tests;

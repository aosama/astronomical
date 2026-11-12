//! Typed speech requests, the audio they produce, and worker-boundary checks.
//!
//! A request is spoken a segment at a time, so the worker sends each
//! segment's audio as it is ready: one bounded frame per segment rather than
//! one frame that grows with the text. Audio crosses as 16-bit
//! little-endian mono PCM in base64, the precision both response formats
//! keep, at a fifth of the size of floats written as JSON numbers.

use astronomical_speech_contract::{SpeechFailure, SpeechRequest};
use serde::{Deserialize, Serialize};

use crate::{RequestId, WorkerModelCapabilitiesValidationError};

/// One speech request sent to the local inference worker.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechCommand {
    pub request_id: RequestId,
    pub model: String,
    pub text: String,
    /// `None` for the model's default voice.
    pub voice_name: Option<String>,
    /// 1.0 is the voice's natural pace.
    pub speed: f32,
    /// Always present: the daemon picks one when the caller names none, and
    /// reports it, so every result can be reproduced.
    pub seed: u64,
}

impl SpeechCommand {
    /// Checks the request again after it crosses the worker trust boundary,
    /// returning it as the engine takes it.
    pub fn validated_request(&self) -> Result<SpeechRequest, SpeechFailure> {
        if self.model.trim().is_empty() {
            return Err(SpeechFailure::invalid_request("model id must not be empty"));
        }
        let speech_request = SpeechRequest {
            text: self.text.clone(),
            voice_name: self.voice_name.clone(),
            speed: self.speed,
            seed: self.seed,
        };
        speech_request.validate()?;
        Ok(speech_request)
    }
}

/// The audio of one spoken segment; it follows the previous segment's.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpokenAudioSegment {
    /// 16-bit little-endian mono samples.
    #[serde(with = "crate::base64_bytes")]
    pub pcm16_bytes: Vec<u8>,
}

impl SpokenAudioSegment {
    /// Whether the bytes are whole 16-bit samples.
    #[must_use]
    pub fn has_whole_samples(&self) -> bool {
        self.pcm16_bytes.len().is_multiple_of(2)
    }
}

/// How a whole request was spoken, sent once its last segment is.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechResultMetadata {
    pub sample_rate_hertz: u32,
    pub sample_count: u64,
    pub segment_count: u32,
    /// Segments the model stopped before their end (Phonon's frame budget
    /// ran out); their audio is kept.
    pub cut_short_segment_count: u32,
    pub voice_name: String,
    pub seed: u64,
}

/// What a loaded speech model offers.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSpeechCapabilities {
    pub sample_rate_hertz: u32,
    pub voice_names: Vec<String>,
    pub default_voice_name: String,
    /// Whether a request may ask for a pace other than 1.0.
    pub supports_speed: bool,
}

impl WorkerSpeechCapabilities {
    /// A speech advertisement must name a sample rate and a default voice
    /// among its voices, or no request could be routed by it.
    pub fn validate(&self) -> Result<(), WorkerModelCapabilitiesValidationError> {
        if self.sample_rate_hertz == 0 {
            return Err(WorkerModelCapabilitiesValidationError::ZeroSpeechSampleRate);
        }
        if !self.voice_names.contains(&self.default_voice_name) {
            return Err(WorkerModelCapabilitiesValidationError::MissingSpeechVoice);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "speech_tests.rs"]
mod tests;

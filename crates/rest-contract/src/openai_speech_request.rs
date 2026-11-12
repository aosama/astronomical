//! Strict request validation for the OpenAI-compatible speech endpoint,
//! `POST /v1/audio/speech`.
//!
//! The request follows OpenAI's shape with three deliberate differences:
//! `model` and `voice` may be left out (the daemon picks the model by its
//! usual rule and the model its default voice), `seed` is accepted so a
//! result can be reproduced, and `response_format` defaults to `wav`
//! because MP3, Opus, AAC and FLAC would each need an encoder this server
//! does not carry; a caller that leaves the field out still gets audio its
//! player can open.

use std::collections::BTreeMap;

use astronomical_speech_contract::{
    AudioFormat, FASTEST_SPEED, MAXIMUM_SPEECH_INPUT_CHARACTERS, SLOWEST_SPEED,
};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

/// The only `stream_format` offered: the audio itself, not server-sent events.
const STREAMED_AUDIO_FORMAT: &str = "audio";

/// One strict request to the local speech endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAiSpeechRequest {
    #[serde(default)]
    model: Option<String>,
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    /// OpenAI's voice direction for models that take it; ours do not, and
    /// OpenAI itself ignores it for such models, so it is accepted and unused.
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    stream_format: Option<String>,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

/// A speech request whose fields are each well formed; the voice and the
/// pace are checked against the chosen model later.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiSpeechRequestParts {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    pub speed: f32,
    pub response_format: AudioFormat,
    pub seed: Option<u64>,
}

impl OpenAiSpeechRequest {
    /// Validates and consumes public input before queue admission or model loading.
    pub fn into_parts(self) -> Result<OpenAiSpeechRequestParts, OpenAiSpeechValidationError> {
        if let Some((field_name, _)) = self.unknown_fields.first_key_value() {
            return Err(OpenAiSpeechValidationError::UnknownField {
                field_name: field_name.clone(),
            });
        }
        if self.input.trim().is_empty() {
            return Err(OpenAiSpeechValidationError::BlankInput);
        }
        if let Some(stream_format) = self.stream_format
            && stream_format != STREAMED_AUDIO_FORMAT
        {
            return Err(OpenAiSpeechValidationError::UnsupportedStreamFormat { stream_format });
        }
        let character_count = self.input.chars().count();
        if character_count > MAXIMUM_SPEECH_INPUT_CHARACTERS {
            return Err(OpenAiSpeechValidationError::InputTooLong { character_count });
        }
        Ok(OpenAiSpeechRequestParts {
            model: non_blank("model", self.model)?,
            voice: non_blank("voice", self.voice)?,
            speed: validated_speed(self.speed)?,
            response_format: validated_response_format(self.response_format)?,
            input: self.input,
            seed: self.seed,
        })
    }
}

/// Why a speech request was refused before it reached a model.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OpenAiSpeechValidationError {
    #[error("request field '{field_name}' is unknown")]
    UnknownField { field_name: String },
    #[error("input must not be blank")]
    BlankInput,
    #[error(
        "input has {character_count} characters; the most one request takes is {MAXIMUM_SPEECH_INPUT_CHARACTERS}"
    )]
    InputTooLong { character_count: usize },
    #[error("{parameter_name} must not be blank when given")]
    BlankField { parameter_name: &'static str },
    #[error("speed must be between {SLOWEST_SPEED} and {FASTEST_SPEED}, received {actual_speed}")]
    SpeedOutOfRange { actual_speed: f32 },
    #[error("response_format '{response_format}' is not supported; use 'wav' or 'pcm'")]
    UnsupportedResponseFormat { response_format: String },
    #[error("stream_format '{stream_format}' is not supported; only 'audio' is offered")]
    UnsupportedStreamFormat { stream_format: String },
}

impl OpenAiSpeechValidationError {
    /// The supported request field the error is about, for the error body's
    /// `param`; `None` for an unknown field, which the message names.
    #[must_use]
    pub fn parameter_name(&self) -> Option<&'static str> {
        match self {
            Self::UnknownField { .. } => None,
            Self::BlankInput | Self::InputTooLong { .. } => Some("input"),
            Self::BlankField { parameter_name } => Some(parameter_name),
            Self::SpeedOutOfRange { .. } => Some("speed"),
            Self::UnsupportedResponseFormat { .. } => Some("response_format"),
            Self::UnsupportedStreamFormat { .. } => Some("stream_format"),
        }
    }
}

fn non_blank(
    parameter_name: &'static str,
    field_value: Option<String>,
) -> Result<Option<String>, OpenAiSpeechValidationError> {
    match field_value {
        Some(text) if text.trim().is_empty() => {
            Err(OpenAiSpeechValidationError::BlankField { parameter_name })
        }
        other_value => Ok(other_value),
    }
}

fn validated_speed(speed: Option<f32>) -> Result<f32, OpenAiSpeechValidationError> {
    let actual_speed = speed.unwrap_or(1.0);
    if actual_speed.is_finite() && (SLOWEST_SPEED..=FASTEST_SPEED).contains(&actual_speed) {
        return Ok(actual_speed);
    }
    Err(OpenAiSpeechValidationError::SpeedOutOfRange { actual_speed })
}

fn validated_response_format(
    response_format: Option<String>,
) -> Result<AudioFormat, OpenAiSpeechValidationError> {
    let Some(format_name) = response_format else {
        return Ok(AudioFormat::Wav);
    };
    AudioFormat::from_name(&format_name).ok_or(
        OpenAiSpeechValidationError::UnsupportedResponseFormat {
            response_format: format_name,
        },
    )
}

#[cfg(test)]
#[path = "openai_speech_request_tests.rs"]
mod tests;

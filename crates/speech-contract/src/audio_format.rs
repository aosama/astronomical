//! The audio formats speech can be returned in.

/// Every format carries 16-bit little-endian mono samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioFormat {
    /// A RIFF WAV file.
    Wav,
    /// Bare samples, for a caller that knows the sample rate.
    Pcm,
}

impl AudioFormat {
    /// The OpenAI speech API's `response_format` names.
    pub fn from_name(format_name: &str) -> Option<Self> {
        match format_name {
            "wav" => Some(Self::Wav),
            "pcm" => Some(Self::Pcm),
            _ => None,
        }
    }

    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
        }
    }
}

#[cfg(test)]
#[path = "audio_format_tests.rs"]
mod tests;

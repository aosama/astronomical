//! What one Kokoro utterance computed on its way to the samples, read back
//! for the journey that holds this port against the PyTorch reference
//! (`kokoro_reference_speech`).

use astronomical_runtime_integration::MlxRuntime;

use super::speech_model::KokoroForwardPass;
use crate::speech_models::speech_model_error::SpeechModelError;

/// One utterance's durations, curves and samples, spoken without excitation
/// noise.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct KokoroSpeechTrace {
    /// Frames held per token, the two markers included.
    pub frame_counts: Vec<i32>,
    /// Pitch in hertz, two values per frame.
    pub pitch_curve: Vec<f32>,
    /// Energy, two values per frame.
    pub energy_curve: Vec<f32>,
    /// 24 kHz samples, 600 per frame.
    pub samples: Vec<f32>,
}

impl KokoroSpeechTrace {
    pub(super) fn read(
        runtime: &MlxRuntime,
        forward_pass: &KokoroForwardPass,
    ) -> Result<Self, SpeechModelError> {
        Ok(Self {
            frame_counts: forward_pass.frame_counts.clone(),
            pitch_curve: runtime.array_to_vec_f32(&forward_pass.pitch_curve)?,
            energy_curve: runtime.array_to_vec_f32(&forward_pass.energy_curve)?,
            samples: runtime.array_to_vec_f32(&forward_pass.samples)?,
        })
    }
}

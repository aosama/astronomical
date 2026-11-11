//! A Phonon voice: Mimi speaker latents from a short recording of the
//! voice, which the flow model reads as a prompt before the text.

use std::path::Path;

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::checkpoint_tensors::{CheckpointTensors, load_safetensors_file};
use crate::speech_models::speech_model_error::SpeechModelError;

/// The tensor Gradium's voice files hold, `[1, 512, frames]`.
const VOICE_TENSOR_NAME: &str = "speaker_wavs";

pub(crate) struct PhononVoice {
    /// `[frames, 512]`.
    pub(crate) speaker_latents: MlxArray,
}

impl PhononVoice {
    pub(crate) fn load(
        runtime: &MlxRuntime,
        voice_file_path: &Path,
    ) -> Result<Self, SpeechModelError> {
        let voice_tensors = load_safetensors_file(runtime, voice_file_path)?;
        let checkpoint = CheckpointTensors {
            runtime,
            tensors: &voice_tensors,
        };
        let stored_voice = checkpoint.tensor(VOICE_TENSOR_NAME)?;
        let [1, channel_count, frame_count] = stored_voice.shape()[..] else {
            return Err(SpeechModelError::Artifact(format!(
                "voice latents are {:?}, not [1, channels, frames]",
                stored_voice.shape()
            )));
        };
        let channels_first = runtime.reshape(&stored_voice, &[channel_count, frame_count])?;
        let speaker_latents = runtime.transpose_axes(&channels_first, &[1, 0])?;
        runtime.evaluate_arrays(&[&speaker_latents])?;
        Ok(Self { speaker_latents })
    }
}

//! A Kokoro voice: a table of 510 style vectors, one per utterance length.
//!
//! Each row is 256 wide; the first half shapes timbre (the decoder) and the
//! second half prosody (the predictor). Kokoro picks the row by how many
//! phonemes the utterance has, so short and long sentences each get the
//! style the voice was fitted with at that length.

use std::path::Path;

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::checkpoint_tensors::{CheckpointTensors, load_safetensors_file};
use crate::speech_models::speech_model_error::SpeechModelError;

const STYLE_WIDTH: i32 = 128;
const VOICE_TENSOR_NAME: &str = "voice";

pub(crate) struct KokoroVoice {
    /// `[rows, 256]`.
    style_rows: MlxArray,
}

/// The two halves of one style row, `[1, 128]` each.
pub(crate) struct UtteranceStyle {
    pub(crate) decoder_style: MlxArray,
    pub(crate) prosody_style: MlxArray,
}

impl KokoroVoice {
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
        let row_count = stored_voice.shape()[0];
        let style_rows = runtime.reshape(&stored_voice, &[row_count, 2 * STYLE_WIDTH])?;
        runtime.evaluate_arrays(&[&style_rows])?;
        Ok(Self { style_rows })
    }

    /// The style row for an utterance of `phoneme_count` phonemes.
    pub(crate) fn style_for(
        &self,
        runtime: &MlxRuntime,
        phoneme_count: usize,
    ) -> Result<UtteranceStyle, SpeechModelError> {
        let row_index = style_row_index(phoneme_count, self.style_rows.shape()[0]);
        let row = runtime.slice(
            &self.style_rows,
            &[row_index, 0],
            &[row_index + 1, 2 * STYLE_WIDTH],
            &[1, 1],
        )?;
        Ok(UtteranceStyle {
            decoder_style: runtime.slice(&row, &[0, 0], &[1, STYLE_WIDTH], &[1, 1])?,
            prosody_style: runtime.slice(
                &row,
                &[0, STYLE_WIDTH],
                &[1, 2 * STYLE_WIDTH],
                &[1, 1],
            )?,
        })
    }
}

/// Kokoro's `pack[len(phonemes) - 1]`, kept inside the table at both ends.
pub(crate) fn style_row_index(phoneme_count: usize, row_count: i32) -> i32 {
    (phoneme_count.max(1) as i32 - 1).min(row_count - 1)
}

#[cfg(test)]
#[path = "voice_pack_tests.rs"]
mod tests;

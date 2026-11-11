//! Which phoneme each audio frame belongs to.
//!
//! Kokoro builds a one-hot alignment matrix and multiplies phoneme states by
//! it; multiplying by a one-hot matrix only selects rows, so this port keeps
//! the alignment as row indices and gathers.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::speech_model_error::SpeechModelError;

/// `[frames]` phoneme indices, uploaded for a gather.
pub(crate) fn frame_alignment(
    runtime: &MlxRuntime,
    frame_counts: &[i32],
) -> Result<MlxArray, SpeechModelError> {
    let frame_to_phoneme = phoneme_index_per_frame(frame_counts);
    Ok(runtime.array_from_i32(&frame_to_phoneme, &[frame_to_phoneme.len() as i32])?)
}

/// Each phoneme index repeated for its frames; a count below 1 adds none.
pub(crate) fn phoneme_index_per_frame(frame_counts: &[i32]) -> Vec<i32> {
    let mut frame_to_phoneme = Vec::with_capacity(
        frame_counts
            .iter()
            .map(|count| (*count).max(0) as usize)
            .sum(),
    );
    for (phoneme_index, frame_count) in frame_counts.iter().enumerate() {
        for _ in 0..(*frame_count).max(0) {
            frame_to_phoneme.push(phoneme_index as i32);
        }
    }
    frame_to_phoneme
}

#[cfg(test)]
#[path = "frame_alignment_tests.rs"]
mod tests;

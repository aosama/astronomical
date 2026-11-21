//! Plays spoken audio on this Mac's speakers, the default for
//! `astronomical speak`: a person who asks a Mac to speak expects to hear it,
//! and saving a file is the less common need (`--output`).

use std::path::Path;
use std::process::Command;

use speech_contract::wav_bytes_from_pcm16;

use super::SpeakError;

/// Plays one WAV file to the end. Injected so tests can stand in for the
/// speakers without making a sound.
pub trait SpokenAudioPlayer {
    /// Plays the file and returns once it has finished, or why it could not.
    fn play_wav_file(&self, wav_path: &Path) -> Result<(), String>;
}

/// The Mac's own player, `afplay`, which ships with macOS and plays a WAV
/// file through the current output device until it ends.
pub struct AfplaySpeakers;

const AFPLAY_PATH: &str = "/usr/bin/afplay";

impl SpokenAudioPlayer for AfplaySpeakers {
    fn play_wav_file(&self, wav_path: &Path) -> Result<(), String> {
        let playback_status = Command::new(AFPLAY_PATH)
            .arg(wav_path)
            .status()
            .map_err(|start_error| format!("could not start {AFPLAY_PATH}: {start_error}"))?;
        if playback_status.success() {
            Ok(())
        } else {
            Err(format!("{AFPLAY_PATH} stopped with {playback_status}"))
        }
    }
}

/// Plays the samples through a WAV file in the temporary folder, removed
/// again whatever happens, so speaking aloud leaves nothing behind.
pub(super) fn play_spoken_audio(
    audio_player: &dyn SpokenAudioPlayer,
    pcm16_bytes: &[u8],
    sample_rate_hertz: u32,
) -> Result<(), SpeakError> {
    let Ok(wav_bytes) = wav_bytes_from_pcm16(pcm16_bytes, sample_rate_hertz) else {
        return Err(SpeakError::ReturnedAudioInvalid);
    };
    let playback_file = tempfile::Builder::new()
        .prefix("astronomical-speech-")
        .suffix(".wav")
        .tempfile()
        .map_err(|create_error| SpeakError::PlaybackFailed {
            cause: create_error.to_string(),
        })?;
    std::fs::write(playback_file.path(), wav_bytes).map_err(|write_error| {
        SpeakError::PlaybackFailed {
            cause: write_error.to_string(),
        }
    })?;
    audio_player
        .play_wav_file(playback_file.path())
        .map_err(|cause| SpeakError::PlaybackFailed { cause })
}

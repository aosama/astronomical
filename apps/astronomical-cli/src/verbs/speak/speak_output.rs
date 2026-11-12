//! Where spoken audio lands: the path decides the format (`.wav`, or `.pcm`
//! for bare 16-bit samples), and it is claimed before any work starts
//! (`reserved_output_file`).

use std::path::{Path, PathBuf};

use astronomical_speech_contract::{AudioFormat, wav_bytes_from_pcm16};

use super::SpeakError;
use crate::verbs::reserved_output_file::{
    OutputReservationError, ReservedOutputFile, timestamped_file_name,
};

/// A claimed output and the format its extension asks for.
pub(super) struct ReservedAudioOutput {
    pub(super) reserved_file: ReservedOutputFile,
    pub(super) audio_format: AudioFormat,
}

/// Picks the path (the requested one, or a timestamped WAV in
/// `default_directory`), reads its format, and claims it.
pub(super) fn reserve_audio_output(
    requested_path: Option<&Path>,
    default_directory: &Path,
) -> Result<ReservedAudioOutput, SpeakError> {
    let output_path = match requested_path {
        Some(requested_path) => requested_path.to_path_buf(),
        None => default_directory.join(timestamped_file_name("speech", "wav")),
    };
    let Some(audio_format) = audio_format_of(&output_path) else {
        return Err(SpeakError::OutputMustBeAudio { path: output_path });
    };
    let reserved_file =
        ReservedOutputFile::claim(&output_path).map_err(
            |reservation_error| match reservation_error {
                OutputReservationError::AlreadyExists => {
                    SpeakError::OutputExists { path: output_path }
                }
                OutputReservationError::Unwritable { cause } => SpeakError::OutputUnwritable {
                    path: output_path,
                    cause,
                },
            },
        )?;
    Ok(ReservedAudioOutput {
        reserved_file,
        audio_format,
    })
}

fn audio_format_of(output_path: &Path) -> Option<AudioFormat> {
    let extension = output_path.extension()?.to_str()?.to_ascii_lowercase();
    AudioFormat::from_name(&extension)
}

/// Writes the samples in the claimed format. Audio that is not whole
/// samples is a daemon fault, so the empty claim is removed instead.
pub(super) fn write_audio(
    reserved_output: ReservedAudioOutput,
    pcm16_bytes: &[u8],
    sample_rate_hertz: u32,
) -> Result<PathBuf, SpeakError> {
    let ReservedAudioOutput {
        reserved_file,
        audio_format,
    } = reserved_output;
    let audio_bytes = match audio_format {
        AudioFormat::Wav => wav_bytes_from_pcm16(pcm16_bytes, sample_rate_hertz).ok(),
        AudioFormat::Pcm => pcm16_bytes
            .len()
            .is_multiple_of(2)
            .then(|| pcm16_bytes.to_vec()),
    };
    let Some(audio_bytes) = audio_bytes else {
        reserved_file.discard();
        return Err(SpeakError::ReturnedAudioInvalid);
    };
    let output_path = reserved_file.path().to_path_buf();
    reserved_file
        .fill(&audio_bytes)
        .map_err(|cause| SpeakError::OutputUnwritable {
            path: output_path,
            cause,
        })
}

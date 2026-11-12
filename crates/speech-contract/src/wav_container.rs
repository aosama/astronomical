//! A RIFF WAV container around 16-bit mono PCM.

use thiserror::Error;

use crate::pcm16::pcm16_bytes;

const HEADER_LENGTH: usize = 44;
const FORMAT_CHUNK_LENGTH: u32 = 16;
const PCM_FORMAT_TAG: u16 = 1;
const CHANNEL_COUNT: u16 = 1;
const BYTES_PER_SAMPLE: u16 = 2;
const BITS_PER_SAMPLE: u16 = 16;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WavEncodingError {
    #[error("{sample_count} samples do not fit a WAV file, whose sizes are 32-bit")]
    TooLong { sample_count: usize },
    #[error("{byte_count} bytes are not whole 16-bit samples")]
    PartialSample { byte_count: usize },
}

/// `samples` at `sample_rate_hertz` as a WAV file.
pub fn wav_bytes(samples: &[f32], sample_rate_hertz: u32) -> Result<Vec<u8>, WavEncodingError> {
    wav_bytes_from_pcm16(&pcm16_bytes(samples), sample_rate_hertz)
}

/// 16-bit little-endian mono samples at `sample_rate_hertz` as a WAV file.
/// The daemon holds speech as these bytes, as the worker sends them.
pub fn wav_bytes_from_pcm16(
    pcm16: &[u8],
    sample_rate_hertz: u32,
) -> Result<Vec<u8>, WavEncodingError> {
    let bytes_per_sample = usize::from(BYTES_PER_SAMPLE);
    if !pcm16.len().is_multiple_of(bytes_per_sample) {
        return Err(WavEncodingError::PartialSample {
            byte_count: pcm16.len(),
        });
    }
    let too_long = WavEncodingError::TooLong {
        sample_count: pcm16.len() / bytes_per_sample,
    };
    let data_length = u32::try_from(pcm16.len()).map_err(|_| too_long.clone())?;
    let riff_length = data_length
        .checked_add((HEADER_LENGTH - 8) as u32)
        .ok_or(too_long)?;
    let byte_rate = sample_rate_hertz * u32::from(BYTES_PER_SAMPLE) * u32::from(CHANNEL_COUNT);
    let mut wav = Vec::with_capacity(HEADER_LENGTH + data_length as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_length.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&FORMAT_CHUNK_LENGTH.to_le_bytes());
    wav.extend_from_slice(&PCM_FORMAT_TAG.to_le_bytes());
    wav.extend_from_slice(&CHANNEL_COUNT.to_le_bytes());
    wav.extend_from_slice(&sample_rate_hertz.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&(BYTES_PER_SAMPLE * CHANNEL_COUNT).to_le_bytes());
    wav.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_length.to_le_bytes());
    wav.extend_from_slice(pcm16);
    Ok(wav)
}

#[cfg(test)]
#[path = "wav_container_tests.rs"]
mod tests;

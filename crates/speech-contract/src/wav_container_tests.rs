use super::{WavEncodingError, wav_bytes, wav_bytes_from_pcm16};

#[test]
fn should_write_a_44_byte_header_and_the_samples() {
    let wav = wav_bytes(&[0.0, 0.5], 24_000).expect("two samples fit");
    assert_eq!(wav.len(), 48);
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 40);
    assert_eq!(&wav[8..16], b"WAVEfmt ");
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
    assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 48_000);
    assert_eq!(&wav[36..40], b"data");
    assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 4);
    assert_eq!(&wav[44..], [0, 0, 0x00, 0x40]);
}

// Boundary: no samples is a valid, empty WAV.
#[test]
fn should_write_an_empty_wav_for_no_samples() {
    let wav = wav_bytes(&[], 24_000).expect("an empty file");
    assert_eq!(wav.len(), 44);
    assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 0);
}

// Positive: 16-bit samples the worker already encoded give the same file.
#[test]
fn should_wrap_pcm16_bytes_as_the_float_samples_would_be() {
    let from_floats = wav_bytes(&[0.0, 0.5], 24_000).expect("two samples fit");
    let from_pcm16 = wav_bytes_from_pcm16(&[0, 0, 0x00, 0x40], 24_000).expect("two samples fit");
    assert_eq!(from_pcm16, from_floats);
}

// Negative: an odd byte count is half a sample.
#[test]
fn should_refuse_bytes_that_are_not_whole_samples() {
    assert_eq!(
        wav_bytes_from_pcm16(&[0, 0, 7], 24_000),
        Err(WavEncodingError::PartialSample { byte_count: 3 })
    );
}

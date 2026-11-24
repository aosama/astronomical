//! Samples as 16-bit little-endian PCM.

/// Each sample clamped to [-1, 1] and scaled to `i16`. Models produce
/// samples that are only nominally within range, so clamping happens here,
/// once, for every format; a non-number becomes silence.
pub fn pcm16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let finite_sample = if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let pcm_sample = (finite_sample * f32::from(i16::MAX)).round() as i16;
        pcm_bytes.extend_from_slice(&pcm_sample.to_le_bytes());
    }
    pcm_bytes
}

#[cfg(test)]
#[path = "pcm16_tests.rs"]
mod tests;

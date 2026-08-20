//! Exact Diffusers-compatible clamp, scale, ties-to-even, and uint8 conversion.

use super::Flux2KleinImageEncodingError;

pub fn flux2_klein_reference_rgb_u8(
    decoded_rgb_values: &[f32],
) -> Result<Vec<u8>, Flux2KleinImageEncodingError> {
    if !decoded_rgb_values.len().is_multiple_of(3)
        || decoded_rgb_values.iter().any(|value| !value.is_finite())
    {
        return Err(Flux2KleinImageEncodingError::InvalidDecodedPixels);
    }
    let normalized_values = decoded_rgb_values
        .iter()
        .map(|decoded_value| decoded_value / 2.0 + 0.5)
        .collect::<Vec<_>>();
    normalized_rgb_u8(&normalized_values)
}

pub(super) fn normalized_rgb_u8(
    normalized_rgb_values: &[f32],
) -> Result<Vec<u8>, Flux2KleinImageEncodingError> {
    if !normalized_rgb_values.len().is_multiple_of(3)
        || normalized_rgb_values.iter().any(|value| !value.is_finite())
    {
        return Err(Flux2KleinImageEncodingError::InvalidDecodedPixels);
    }
    Ok(normalized_rgb_values
        .iter()
        .map(|normalized_value| (normalized_value.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8)
        .collect())
}

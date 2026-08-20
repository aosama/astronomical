//! Performs allocation-free structural validation of untrusted PNG completion payloads.

use crc32fast::Hasher;

use super::ImageGenerationCompletionValidationError;

const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const IHDR_DATA_BYTES: usize = 13;
const REQUIRED_BIT_DEPTH: u8 = 8;
const TRUECOLOR_COLOR_TYPE: u8 = 2;

pub(super) struct ValidatedPng {
    pub(super) width_pixels: u32,
    pub(super) height_pixels: u32,
    pub(super) decoded_rgb_byte_count: usize,
}

pub(super) fn validate_png_structure(
    png_bytes: &[u8],
    maximum_decoded_rgb_bytes: usize,
) -> Result<ValidatedPng, ImageGenerationCompletionValidationError> {
    if !png_bytes.starts_with(PNG_SIGNATURE) {
        return Err(ImageGenerationCompletionValidationError::InvalidPngEncoding);
    }

    let mut chunk_start = PNG_SIGNATURE.len();
    let mut width_pixels = None;
    let mut height_pixels = None;
    let mut source_bit_depth = None;
    let mut source_color_type = None;
    let mut idat_data_byte_count = 0usize;

    while chunk_start < png_bytes.len() {
        let chunk_header_end = chunk_start
            .checked_add(8)
            .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
        if chunk_header_end > png_bytes.len() {
            return Err(ImageGenerationCompletionValidationError::InvalidPngEncoding);
        }

        let chunk_data_byte_count = u32::from_be_bytes([
            png_bytes[chunk_start],
            png_bytes[chunk_start + 1],
            png_bytes[chunk_start + 2],
            png_bytes[chunk_start + 3],
        ]) as usize;
        let chunk_data_start = chunk_start + 8;
        let chunk_data_end = chunk_data_start
            .checked_add(chunk_data_byte_count)
            .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
        let chunk_end = chunk_data_end
            .checked_add(4)
            .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
        if chunk_end > png_bytes.len() {
            return Err(ImageGenerationCompletionValidationError::InvalidPngEncoding);
        }

        let chunk_type = &png_bytes[chunk_start + 4..chunk_data_start];
        let chunk_data = &png_bytes[chunk_data_start..chunk_data_end];
        let expected_crc = u32::from_be_bytes([
            png_bytes[chunk_data_end],
            png_bytes[chunk_data_end + 1],
            png_bytes[chunk_data_end + 2],
            png_bytes[chunk_data_end + 3],
        ]);
        let mut crc_hasher = Hasher::new();
        crc_hasher.update(chunk_type);
        crc_hasher.update(chunk_data);
        if crc_hasher.finalize() != expected_crc {
            return Err(ImageGenerationCompletionValidationError::InvalidPngEncoding);
        }

        match chunk_type {
            b"IHDR" => {
                if chunk_start != PNG_SIGNATURE.len()
                    || chunk_data.len() != IHDR_DATA_BYTES
                    || width_pixels.is_some()
                {
                    return Err(ImageGenerationCompletionValidationError::InvalidPngEncoding);
                }
                width_pixels = Some(u32::from_be_bytes([
                    chunk_data[0],
                    chunk_data[1],
                    chunk_data[2],
                    chunk_data[3],
                ]));
                height_pixels = Some(u32::from_be_bytes([
                    chunk_data[4],
                    chunk_data[5],
                    chunk_data[6],
                    chunk_data[7],
                ]));
                source_bit_depth = Some(chunk_data[8]);
                source_color_type = Some(chunk_data[9]);
            }
            b"IDAT" => {
                idat_data_byte_count = idat_data_byte_count
                    .checked_add(chunk_data.len())
                    .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
            }
            b"IEND" => {
                if !chunk_data.is_empty()
                    || chunk_end != png_bytes.len()
                    || idat_data_byte_count == 0
                {
                    return Err(ImageGenerationCompletionValidationError::InvalidPngEncoding);
                }
                let width_pixels = width_pixels
                    .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
                let height_pixels = height_pixels
                    .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
                if source_bit_depth != Some(REQUIRED_BIT_DEPTH)
                    || source_color_type != Some(TRUECOLOR_COLOR_TYPE)
                {
                    // Decoder output is insufficient evidence because palette PNGs expand to RGB8.
                    return Err(ImageGenerationCompletionValidationError::NonRgb8Png);
                }
                let decoded_rgb_byte_count = usize::try_from(width_pixels)
                    .ok()
                    .and_then(|width| {
                        usize::try_from(height_pixels)
                            .ok()
                            .and_then(|height| width.checked_mul(height))
                    })
                    .and_then(|pixel_count| pixel_count.checked_mul(3))
                    .ok_or(ImageGenerationCompletionValidationError::PngDecodeResourceLimit)?;
                if decoded_rgb_byte_count > maximum_decoded_rgb_bytes {
                    return Err(ImageGenerationCompletionValidationError::PngDecodeResourceLimit);
                }
                return Ok(ValidatedPng {
                    width_pixels,
                    height_pixels,
                    decoded_rgb_byte_count,
                });
            }
            _ => {}
        }

        chunk_start = chunk_end;
    }

    Err(ImageGenerationCompletionValidationError::InvalidPngEncoding)
}

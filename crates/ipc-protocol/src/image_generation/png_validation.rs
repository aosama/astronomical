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
        let chunk_bounds = png_chunk_bounds(png_bytes, chunk_start)?;
        let chunk_type = &png_bytes[chunk_start + 4..chunk_bounds.data_start];
        let chunk_data = &png_bytes[chunk_bounds.data_start..chunk_bounds.data_end];
        if !png_chunk_crc_matches(png_bytes, chunk_type, chunk_data, chunk_bounds.data_end) {
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
                let (ihdr_width, ihdr_height, ihdr_bit_depth, ihdr_color_type) =
                    parse_ihdr_fields(chunk_data);
                width_pixels = Some(ihdr_width);
                height_pixels = Some(ihdr_height);
                source_bit_depth = Some(ihdr_bit_depth);
                source_color_type = Some(ihdr_color_type);
            }
            b"IDAT" => {
                idat_data_byte_count = idat_data_byte_count
                    .checked_add(chunk_data.len())
                    .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
            }
            b"IEND" => {
                if !chunk_data.is_empty()
                    || chunk_bounds.chunk_end != png_bytes.len()
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
                let decoded_rgb_byte_count = decoded_rgb_byte_count_within_limit(
                    width_pixels,
                    height_pixels,
                    maximum_decoded_rgb_bytes,
                )?;
                return Ok(ValidatedPng {
                    width_pixels,
                    height_pixels,
                    decoded_rgb_byte_count,
                });
            }
            _ => {}
        }

        chunk_start = chunk_bounds.chunk_end;
    }

    Err(ImageGenerationCompletionValidationError::InvalidPngEncoding)
}

/// Byte ranges of one PNG chunk, all checked against the payload's real
/// length before any slicing happens. Untrusted length fields drive the
/// arithmetic, so every add is an overflow check and every end is a bounds
/// check against the slice the bytes actually came from.
struct PngChunkBounds {
    data_start: usize,
    data_end: usize,
    chunk_end: usize,
}

fn png_chunk_bounds(
    png_bytes: &[u8],
    chunk_start: usize,
) -> Result<PngChunkBounds, ImageGenerationCompletionValidationError> {
    let chunk_header_end = chunk_start
        .checked_add(8)
        .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
    if chunk_header_end > png_bytes.len() {
        return Err(ImageGenerationCompletionValidationError::InvalidPngEncoding);
    }
    let data_byte_count = u32::from_be_bytes([
        png_bytes[chunk_start],
        png_bytes[chunk_start + 1],
        png_bytes[chunk_start + 2],
        png_bytes[chunk_start + 3],
    ]) as usize;
    let data_start = chunk_start + 8;
    let data_end = data_start
        .checked_add(data_byte_count)
        .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
    let chunk_end = data_end
        .checked_add(4)
        .ok_or(ImageGenerationCompletionValidationError::InvalidPngEncoding)?;
    if chunk_end > png_bytes.len() {
        return Err(ImageGenerationCompletionValidationError::InvalidPngEncoding);
    }
    Ok(PngChunkBounds {
        data_start,
        data_end,
        chunk_end,
    })
}

/// Verifies a chunk's trailing CRC32 over its type and data bytes. The caller
/// has already proven the CRC field lies inside the payload, so indexing here
/// cannot panic and any mismatch rejects the whole payload as untrusted.
fn png_chunk_crc_matches(
    png_bytes: &[u8],
    chunk_type: &[u8],
    chunk_data: &[u8],
    chunk_data_end: usize,
) -> bool {
    let expected_crc = u32::from_be_bytes([
        png_bytes[chunk_data_end],
        png_bytes[chunk_data_end + 1],
        png_bytes[chunk_data_end + 2],
        png_bytes[chunk_data_end + 3],
    ]);
    let mut crc_hasher = Hasher::new();
    crc_hasher.update(chunk_type);
    crc_hasher.update(chunk_data);
    crc_hasher.finalize() == expected_crc
}

/// Reads the big-endian IHDR fields: width, height, bit depth, and color
/// type. Pure parsing — every semantic check stays with the caller.
fn parse_ihdr_fields(ihdr_data: &[u8]) -> (u32, u32, u8, u8) {
    (
        u32::from_be_bytes([ihdr_data[0], ihdr_data[1], ihdr_data[2], ihdr_data[3]]),
        u32::from_be_bytes([ihdr_data[4], ihdr_data[5], ihdr_data[6], ihdr_data[7]]),
        ihdr_data[8],
        ihdr_data[9],
    )
}

/// Computes width x height x 3 (one RGB triple per pixel) with every
/// multiplication checked, then enforces the caller's decoded-size ceiling.
/// Overflow and ceiling violations are the same failure because both mean the
/// payload could demand unbounded decode memory.
pub(super) fn decoded_rgb_byte_count_within_limit(
    width_pixels: u32,
    height_pixels: u32,
    maximum_decoded_rgb_bytes: usize,
) -> Result<usize, ImageGenerationCompletionValidationError> {
    usize::try_from(width_pixels)
        .ok()
        .and_then(|width| {
            usize::try_from(height_pixels)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixel_count| pixel_count.checked_mul(3))
        .ok_or(ImageGenerationCompletionValidationError::PngDecodeResourceLimit)
        .and_then(|decoded_rgb_byte_count| {
            if decoded_rgb_byte_count > maximum_decoded_rgb_bytes {
                return Err(ImageGenerationCompletionValidationError::PngDecodeResourceLimit);
            }
            Ok(decoded_rgb_byte_count)
        })
}

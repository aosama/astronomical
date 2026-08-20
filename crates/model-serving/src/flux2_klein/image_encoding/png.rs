//! Deterministic, lossless RGB PNG encoding through the workspace image crate.

use std::io::Cursor;

use image::ImageEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};

use super::{Flux2KleinImageEncodingError, flux2_klein_reference_rgb_u8};

pub struct Flux2KleinPngEncoder;

impl Flux2KleinPngEncoder {
    pub fn encode_decoded_rgb(
        width_pixels: u32,
        height_pixels: u32,
        decoded_rgb_values: &[f32],
    ) -> Result<Vec<u8>, Flux2KleinImageEncodingError> {
        let rgb_bytes = flux2_klein_reference_rgb_u8(decoded_rgb_values)?;
        Self::encode_rgb8(width_pixels, height_pixels, &rgb_bytes)
    }

    pub fn encode_rgb8(
        width_pixels: u32,
        height_pixels: u32,
        rgb_bytes: &[u8],
    ) -> Result<Vec<u8>, Flux2KleinImageEncodingError> {
        if width_pixels == 0 || height_pixels == 0 {
            return Err(Flux2KleinImageEncodingError::InvalidRgbGeometry);
        }
        let expected_bytes = usize::try_from(width_pixels)
            .ok()
            .and_then(|width| {
                usize::try_from(height_pixels)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(Flux2KleinImageEncodingError::GeometryOverflow)?;
        if rgb_bytes.len() != expected_bytes {
            return Err(Flux2KleinImageEncodingError::InvalidRgbGeometry);
        }
        let mut png_bytes = Vec::new();
        let encoder = PngEncoder::new_with_quality(
            Cursor::new(&mut png_bytes),
            CompressionType::Best,
            FilterType::Adaptive,
        );
        encoder
            .write_image(
                rgb_bytes,
                width_pixels,
                height_pixels,
                image::ExtendedColorType::Rgb8,
            )
            .map_err(Flux2KleinImageEncodingError::Png)?;
        Ok(png_bytes)
    }
}

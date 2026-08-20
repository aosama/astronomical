//! Model-derived image geometry and encoded-transport bounds.

use thiserror::Error;

const DIMENSION_MULTIPLE_PIXELS: u32 = 16;
const MAXIMUM_PIXEL_COUNT: u64 = 1_048_576;
const MAXIMUM_ASPECT_RATIO: u64 = 4;
const PACKED_LATENT_CHANNEL_COUNT: u64 = 128;
const PNG_FIXED_OVERHEAD_BYTES: u64 = 65_536;

/// A request whose dimensions cannot reach the model or transport boundary safely.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Flux2KleinDimensionError {
    #[error("FLUX.2 Klein image dimensions must be positive")]
    ZeroDimension,
    #[error("FLUX.2 Klein image {dimension} {actual_pixels} must be a multiple of 16")]
    Unaligned {
        dimension: &'static str,
        actual_pixels: u32,
    },
    #[error("FLUX.2 Klein image pixel count exceeds the supported profile")]
    PixelCountExceeded,
    #[error("FLUX.2 Klein image aspect ratio exceeds 4:1")]
    UnsupportedAspectRatio {
        width_pixels: u32,
        height_pixels: u32,
    },
    #[error("FLUX.2 Klein image geometry overflowed bounded arithmetic")]
    GeometryOverflow,
    #[error("worst-case lossless image transport exceeds the configured byte ceiling")]
    TransportLimitExceeded {
        required_bytes: u64,
        transport_limit_bytes: u64,
    },
}

/// Validated dimensions and exact allocation/transport geometry for one image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Flux2KleinImageDimensions {
    width_pixels: u32,
    height_pixels: u32,
    pixel_count: u64,
    latent_element_count: u64,
    maximum_png_bytes: u64,
    maximum_base64_png_bytes: u64,
}

impl Flux2KleinImageDimensions {
    pub fn validate(
        width_pixels: u32,
        height_pixels: u32,
        transport_limit_bytes: u64,
    ) -> Result<Self, Flux2KleinDimensionError> {
        if width_pixels == 0 || height_pixels == 0 {
            return Err(Flux2KleinDimensionError::ZeroDimension);
        }
        validate_alignment("width", width_pixels)?;
        validate_alignment("height", height_pixels)?;
        let pixel_count = u64::from(width_pixels)
            .checked_mul(u64::from(height_pixels))
            .ok_or(Flux2KleinDimensionError::GeometryOverflow)?;
        if pixel_count > MAXIMUM_PIXEL_COUNT {
            return Err(Flux2KleinDimensionError::PixelCountExceeded);
        }
        let long_edge = u64::from(width_pixels.max(height_pixels));
        let short_edge = u64::from(width_pixels.min(height_pixels));
        if long_edge > short_edge.saturating_mul(MAXIMUM_ASPECT_RATIO) {
            return Err(Flux2KleinDimensionError::UnsupportedAspectRatio {
                width_pixels,
                height_pixels,
            });
        }
        let latent_element_count = u64::from(width_pixels / DIMENSION_MULTIPLE_PIXELS)
            .checked_mul(u64::from(height_pixels / DIMENSION_MULTIPLE_PIXELS))
            .and_then(|token_count| token_count.checked_mul(PACKED_LATENT_CHANNEL_COUNT))
            .ok_or(Flux2KleinDimensionError::GeometryOverflow)?;
        // PNG scanlines add one filter byte per row. Fixed overhead deliberately
        // over-bounds chunk framing so admission never depends on encoder choices.
        let maximum_png_bytes = pixel_count
            .checked_mul(3)
            .and_then(|rgb_bytes| rgb_bytes.checked_add(u64::from(height_pixels)))
            .and_then(|scanline_bytes| scanline_bytes.checked_add(PNG_FIXED_OVERHEAD_BYTES))
            .ok_or(Flux2KleinDimensionError::GeometryOverflow)?;
        let maximum_base64_png_bytes = maximum_png_bytes
            .checked_add(2)
            .and_then(|rounded_bytes| rounded_bytes.checked_div(3))
            .and_then(|base64_groups| base64_groups.checked_mul(4))
            .ok_or(Flux2KleinDimensionError::GeometryOverflow)?;
        if maximum_base64_png_bytes > transport_limit_bytes {
            return Err(Flux2KleinDimensionError::TransportLimitExceeded {
                required_bytes: maximum_base64_png_bytes,
                transport_limit_bytes,
            });
        }
        Ok(Self {
            width_pixels,
            height_pixels,
            pixel_count,
            latent_element_count,
            maximum_png_bytes,
            maximum_base64_png_bytes,
        })
    }

    pub const fn width_pixels(&self) -> u32 {
        self.width_pixels
    }
    pub const fn height_pixels(&self) -> u32 {
        self.height_pixels
    }
    pub const fn pixel_count(&self) -> u64 {
        self.pixel_count
    }
    pub const fn latent_element_count(&self) -> u64 {
        self.latent_element_count
    }
    pub const fn maximum_png_bytes(&self) -> u64 {
        self.maximum_png_bytes
    }
    pub const fn maximum_base64_png_bytes(&self) -> u64 {
        self.maximum_base64_png_bytes
    }
}

fn validate_alignment(
    dimension: &'static str,
    actual_pixels: u32,
) -> Result<(), Flux2KleinDimensionError> {
    if actual_pixels.is_multiple_of(DIMENSION_MULTIPLE_PIXELS) {
        Ok(())
    } else {
        Err(Flux2KleinDimensionError::Unaligned {
            dimension,
            actual_pixels,
        })
    }
}

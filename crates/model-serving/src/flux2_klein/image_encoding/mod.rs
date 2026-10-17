//! Exact generated-pixel conversion and deterministic lossless PNG publication.

#[cfg(feature = "direct-mlx")]
mod error;
#[cfg(feature = "direct-mlx")]
pub(crate) mod png;
#[cfg(feature = "direct-mlx")]
mod reference_pixels;

#[cfg(feature = "direct-mlx")]
mod mlx_pixels;

#[cfg(feature = "direct-mlx")]
pub use error::Flux2KleinImageEncodingError;
#[cfg(feature = "direct-mlx")]
pub use png::Flux2KleinPngEncoder;
#[cfg(feature = "direct-mlx")]
pub use reference_pixels::flux2_klein_reference_rgb_u8;

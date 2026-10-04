//! Typed failures for exact generated-pixel conversion and PNG publication.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Flux2KleinImageEncodingError {
    #[error("decoded FLUX.2 Klein pixels must be finite RGB values")]
    InvalidDecodedPixels,
    #[error("generated RGB byte geometry does not match the declared image dimensions")]
    InvalidRgbGeometry,
    #[error("generated image geometry overflowed bounded arithmetic")]
    GeometryOverflow,
    #[error("lossless PNG encoding failed")]
    Png(#[source] image::ImageError),
    #[cfg(feature = "direct-mlx")]
    #[error("copying generated pixels from MLX failed")]
    Mlx(#[from] astronomical_runtime_integration::MlxRuntimeError),
}

#[cfg(feature = "direct-mlx")]
impl From<astronomical_mlx_c_rust::MlxCError> for Flux2KleinImageEncodingError {
    fn from(captured_error: astronomical_mlx_c_rust::MlxCError) -> Self {
        astronomical_runtime_integration::MlxRuntimeError::from(captured_error).into()
    }
}

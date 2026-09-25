//! The pure data contract of one Qwen-Image-2.1 render: the request, the completed image, and
//! the boundary sequence a render crosses.
//!
//! These types are deliberately free of MLX dependencies so the serving engine, its component
//! seam, and the hermetic lifecycle tests compile and run without the `direct-mlx` feature —
//! only the pipeline that *executes* a render and its MLX component owner need the runtime.
//! The synchronous `render` and the serving engine both consume this one contract, so a
//! journey over either is a proof for both.

use std::io::Cursor;

use image::ImageEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};

use crate::qwen_image_21::engine_error::QwenImage21EngineError;

/// One text-to-image render request.
pub struct QwenImage21RenderRequest {
    /// The prompt guiding generation.
    pub prompt: String,
    /// Target pixel width; floored to the nearest multiple of 32.
    pub width: usize,
    /// Target pixel height; floored to the nearest multiple of 32.
    pub height: usize,
    /// Denoising steps (the reference default is 40; fewer trade quality for speed).
    pub num_inference_steps: usize,
    /// Deterministic noise seed.
    pub seed: u64,
}

/// A completed render: RGB pixel bytes ready for PNG encoding.
#[derive(Clone)]
pub struct QwenImage21Rendered {
    /// Pixel width of the rendered image.
    pub width: usize,
    /// Pixel height of the rendered image.
    pub height: usize,
    /// `width * height` RGB triples, row-major, no padding.
    pub rgb_bytes: Vec<u8>,
}

impl QwenImage21Rendered {
    /// Lossless PNG bytes for the rendered image.
    ///
    /// The VAE reconstructs four channels (the pipeline treats images as RGBA); the first three
    /// carry the picture, so the render encodes RGB and drops the fourth.
    pub fn to_png_bytes(&self) -> Result<Vec<u8>, QwenImage21EngineError> {
        let mut png_bytes = Vec::new();
        let encoder = PngEncoder::new_with_quality(
            Cursor::new(&mut png_bytes),
            CompressionType::Best,
            FilterType::Adaptive,
        );
        encoder
            .write_image(
                &self.rgb_bytes,
                u32::try_from(self.width).map_err(|_| QwenImage21EngineError::Execution {
                    description: "the rendered width exceeds the PNG u32 range".to_owned(),
                })?,
                u32::try_from(self.height).map_err(|_| QwenImage21EngineError::Execution {
                    description: "the rendered height exceeds the PNG u32 range".to_owned(),
                })?,
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|source| QwenImage21EngineError::Execution {
                description: format!("the PNG encode failed: {source}"),
            })?;
        Ok(png_bytes)
    }
}

/// The next boundary a render crossing produces, in the order the reference executes them.
///
/// `Rendered` carries the payload so the engine can encode and publish without a second
/// component call; every other variant is a progress boundary.
#[derive(Clone)]
pub enum QwenImage21RenderAdvance {
    /// The pipeline's weights were mapped (the engine's preparation boundary).
    Preparing,
    /// Tokenization, encoding, and the system-block slice completed; the encoder is released.
    ConditioningCompleted,
    /// The schedule and seeded initial noise are ready.
    NoisePrepared,
    /// Denoising step `completed_steps` of `total_steps` finished.
    DenoisingStep {
        completed_steps: usize,
        total_steps: usize,
    },
    /// The transformer is released and the VAE decoded the latent grid into clamped pixels.
    DecodingCompleted,
    /// The clamped pixels were converted to RGB bytes.
    Rendered(QwenImage21Rendered),
}

/// `(x + 1) / 2 · 255` per channel, clamped to bytes, first three channels only.
// The feature-gated render session performs the conversion; the formula stays in the pure
// contract so both a journey and a hermetic fake can reach it.
#[cfg_attr(not(feature = "direct-mlx"), allow(dead_code))]
pub fn rgba_to_rgb8(rgba_values: &[f32]) -> Vec<u8> {
    let pixel_count = rgba_values.len() / 4;
    let mut rgb_bytes = Vec::with_capacity(pixel_count * 3);
    for pixel_index in 0..pixel_count {
        for channel_index in 0..3 {
            let channel = rgba_values[pixel_index * 4 + channel_index];
            let scaled = (channel + 1.0) * 0.5 * 255.0;
            rgb_bytes.push(scaled.clamp(0.0, 255.0) as u8);
        }
    }
    rgb_bytes
}

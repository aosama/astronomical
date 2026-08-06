//! CPU image decoding and Qwen3-VL patch packing for Qwen3.5.
//!
//! Source lineage: Rust translation of MLX-VLM's Qwen3-VL image processor
//! (MIT License). That processor identifies Hugging Face Transformers' Qwen2-VL
//! image processor (Apache License 2.0) as the source of its resize and packing
//! behavior. See third-party license notices for complete attribution.
//!
//! MLX-C begins after this stage: MLX has tensor resize/convolution primitives,
//! but not the encoded PNG/JPEG/WebP boundary needed here. This module decodes,
//! resizes, normalizes, and packs host pixels; `vision_model.rs` uploads the
//! resulting rows and delegates all learned tensor math to MLX.

use std::io::Cursor;

use image::{ImageReader, RgbImage, imageops::FilterType};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::vision_config::Qwen3_5VisionConfig;

const QWEN3_5_MOE_IMAGE_CHANNEL_COUNT: u32 = 3;
const QWEN3_5_MOE_MAXIMUM_ASPECT_RATIO: f64 = 200.0;

/// Patch grid produced by the Qwen3VL image processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3_5ImageGrid {
    pub temporal_patch_count: u32,
    pub height_patch_count: u32,
    pub width_patch_count: u32,
}

/// Pixel dimensions in height-then-width order, matching Qwen3VL processor math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3_5ImageDimensions {
    pub height_pixels: u32,
    pub width_pixels: u32,
}

/// CPU-side image processor output before the MLX vision tower consumes it.
///
/// Each row is one flattened patch in C,T,H,W order. Rows themselves use the
/// spatial-merger block order expected by the vision tower. A still image is
/// duplicated conceptually across the temporal patch dimension: both temporal
/// slots read the same source RGB pixel, matching upstream `np.repeat`.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3_5ProcessedImage {
    /// SHA-256 of the exact encoded PNG, JPEG, or WebP bytes received at the request boundary.
    pub encoded_image_sha256: [u8; 32],
    pub pixel_values: Vec<f32>,
    pub pixel_values_row_count: usize,
    pub pixel_values_column_count: usize,
    pub image_grid: Qwen3_5ImageGrid,
    pub resized_height_pixels: u32,
    pub resized_width_pixels: u32,
    pub image_token_count_after_spatial_merge: usize,
}

/// Native Rust implementation of the Qwen3VL image preprocessing.
#[derive(Debug, Clone)]
pub struct Qwen3_5ImageProcessor {
    patch_size_pixels: u32,
    temporal_patch_size: u32,
    spatial_merge_size: u32,
    minimum_image_pixels: u64,
    maximum_image_pixels: u64,
    image_mean_by_channel: [f32; 3],
    image_std_by_channel: [f32; 3],
    rescale_factor: f32,
}

#[derive(Debug, Error)]
pub enum Qwen3_5ImageProcessingError {
    #[error("failed to guess image format")]
    ImageFormatGuess { source: std::io::Error },
    #[error("failed to decode image")]
    ImageDecode { source: image::ImageError },
    #[error(
        "image aspect ratio {aspect_ratio:.2} exceeds maximum {maximum_aspect_ratio:.2} for {height_pixels}x{width_pixels} image"
    )]
    AspectRatioTooLarge {
        height_pixels: u32,
        width_pixels: u32,
        aspect_ratio: f64,
        maximum_aspect_ratio: f64,
    },
    #[error("image processor config is invalid: {message}")]
    InvalidProcessorConfig { message: String },
}

impl Qwen3_5ImageProcessor {
    /// Builds the image processor from a discovered vision configuration.
    ///
    /// The `patch_size`, `temporal_patch_size`, and `spatial_merge_size` are
    /// derived from the model's `vision_config`. The remaining parameters
    /// (`minimum_image_pixels`, `maximum_image_pixels`, `image_mean_by_channel`,
    /// `image_std_by_channel`, `rescale_factor`) are Qwen3.5-family constants
    /// that are the same across all models in this family.
    pub fn from_vision_config(vision_config: &Qwen3_5VisionConfig) -> Self {
        Self {
            patch_size_pixels: vision_config.patch_size(),
            temporal_patch_size: vision_config.temporal_patch_size(),
            spatial_merge_size: vision_config.spatial_merge_size(),
            minimum_image_pixels: 65_536,
            maximum_image_pixels: 16_777_216,
            image_mean_by_channel: [0.5, 0.5, 0.5],
            image_std_by_channel: [0.5, 0.5, 0.5],
            rescale_factor: 1.0 / 255.0,
        }
    }

    /// Decode image bytes and produce flattened Qwen3VL patch rows.
    pub fn process_image_bytes(
        &self,
        encoded_image_bytes: &[u8],
    ) -> Result<Qwen3_5ProcessedImage, Qwen3_5ImageProcessingError> {
        self.validate_config()?;
        let encoded_image_sha256 = Sha256::digest(encoded_image_bytes).into();
        let decoded_dynamic_image = ImageReader::new(Cursor::new(encoded_image_bytes))
            .with_guessed_format()
            .map_err(
                |format_guess_error| Qwen3_5ImageProcessingError::ImageFormatGuess {
                    source: format_guess_error,
                },
            )?
            .decode()
            .map_err(
                |image_decode_error| Qwen3_5ImageProcessingError::ImageDecode {
                    source: image_decode_error,
                },
            )?;

        self.process_rgb_image(decoded_dynamic_image.to_rgb8(), encoded_image_sha256)
    }

    /// Plan the Qwen3VL resize target for an image without decoding or allocating pixels.
    pub fn resized_dimensions_for_image(
        &self,
        original_height_pixels: u32,
        original_width_pixels: u32,
    ) -> Result<Qwen3_5ImageDimensions, Qwen3_5ImageProcessingError> {
        self.validate_config()?;
        self.smart_resize_dimensions(original_height_pixels, original_width_pixels)
    }

    /// Returns the maximum merged visual-token rows this processor can produce for one still image.
    #[must_use]
    pub fn maximum_image_token_count_after_spatial_merge(&self) -> usize {
        let patch_area_pixels =
            u64::from(self.patch_size_pixels).saturating_mul(u64::from(self.patch_size_pixels));
        let spatial_merge_area =
            u64::from(self.spatial_merge_size).saturating_mul(u64::from(self.spatial_merge_size));
        if patch_area_pixels == 0 || spatial_merge_area == 0 {
            return 0;
        }
        let maximum_patch_count = self.maximum_image_pixels / patch_area_pixels;
        let maximum_visual_token_count = maximum_patch_count / spatial_merge_area;
        usize::try_from(maximum_visual_token_count).unwrap_or(usize::MAX)
    }

    fn process_rgb_image(
        &self,
        decoded_rgb_image: RgbImage,
        encoded_image_sha256: [u8; 32],
    ) -> Result<Qwen3_5ProcessedImage, Qwen3_5ImageProcessingError> {
        let original_width_pixels = decoded_rgb_image.width();
        let original_height_pixels = decoded_rgb_image.height();
        let resized_dimensions =
            self.smart_resize_dimensions(original_height_pixels, original_width_pixels)?;

        let resized_rgb_image = image::imageops::resize(
            &decoded_rgb_image,
            resized_dimensions.width_pixels,
            resized_dimensions.height_pixels,
            FilterType::CatmullRom,
        );

        let image_grid = Qwen3_5ImageGrid {
            temporal_patch_count: 1,
            height_patch_count: resized_dimensions.height_pixels / self.patch_size_pixels,
            width_patch_count: resized_dimensions.width_pixels / self.patch_size_pixels,
        };
        let pixel_values_row_count = self.pixel_values_row_count(image_grid)?;
        let pixel_values_column_count = self.pixel_values_column_count()?;
        let image_token_count_after_spatial_merge =
            self.image_token_count_after_spatial_merge(pixel_values_row_count)?;
        let total_pixel_value_count = pixel_values_row_count
            .checked_mul(pixel_values_column_count)
            .ok_or_else(|| Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "pixel value count overflows this platform".to_owned(),
            })?;

        let pixel_values = self.flatten_pixel_values(
            &resized_rgb_image,
            image_grid,
            pixel_values_row_count,
            pixel_values_column_count,
            total_pixel_value_count,
        );

        Ok(Qwen3_5ProcessedImage {
            encoded_image_sha256,
            pixel_values,
            pixel_values_row_count,
            pixel_values_column_count,
            image_grid,
            resized_height_pixels: resized_dimensions.height_pixels,
            resized_width_pixels: resized_dimensions.width_pixels,
            image_token_count_after_spatial_merge,
        })
    }

    fn smart_resize_dimensions(
        &self,
        height_pixels: u32,
        width_pixels: u32,
    ) -> Result<Qwen3_5ImageDimensions, Qwen3_5ImageProcessingError> {
        if height_pixels == 0 || width_pixels == 0 {
            return Err(Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "image dimensions must be positive".to_owned(),
            });
        }

        let taller_side_pixels = height_pixels.max(width_pixels);
        let shorter_side_pixels = height_pixels.min(width_pixels);
        let aspect_ratio = f64::from(taller_side_pixels) / f64::from(shorter_side_pixels);
        if aspect_ratio > QWEN3_5_MOE_MAXIMUM_ASPECT_RATIO {
            return Err(Qwen3_5ImageProcessingError::AspectRatioTooLarge {
                height_pixels,
                width_pixels,
                aspect_ratio,
                maximum_aspect_ratio: QWEN3_5_MOE_MAXIMUM_ASPECT_RATIO,
            });
        }

        // Dimensions must be multiples of patch_size * merge_size, not merely
        // patch_size. That guarantees an integer patch grid and complete 2x2
        // merger blocks after the transformer.
        let resize_factor_pixels = self.patch_size_pixels * self.spatial_merge_size;
        let mut resized_height_pixels =
            round_to_nearest_factor(height_pixels, resize_factor_pixels);
        let mut resized_width_pixels = round_to_nearest_factor(width_pixels, resize_factor_pixels);
        let rounded_pixel_count =
            u64::from(resized_height_pixels) * u64::from(resized_width_pixels);

        // Preserve aspect ratio while fitting the checkpoint's pixel budget.
        // Floor is required when shrinking (never exceed max); ceil is required
        // when growing (reach min). These are the same formulas as upstream
        // `_smart_resize_image`, with final dimensions snapped to the factor.
        if rounded_pixel_count > self.maximum_image_pixels {
            let shrink_ratio = (f64::from(height_pixels) * f64::from(width_pixels)
                / self.maximum_image_pixels as f64)
                .sqrt();
            resized_height_pixels =
                shrink_to_factor(height_pixels, shrink_ratio, resize_factor_pixels);
            resized_width_pixels =
                shrink_to_factor(width_pixels, shrink_ratio, resize_factor_pixels);
        } else if rounded_pixel_count < self.minimum_image_pixels {
            let grow_ratio = (self.minimum_image_pixels as f64
                / (f64::from(height_pixels) * f64::from(width_pixels)))
            .sqrt();
            resized_height_pixels = grow_to_factor(height_pixels, grow_ratio, resize_factor_pixels);
            resized_width_pixels = grow_to_factor(width_pixels, grow_ratio, resize_factor_pixels);
        }

        Ok(Qwen3_5ImageDimensions {
            height_pixels: resized_height_pixels,
            width_pixels: resized_width_pixels,
        })
    }

    fn flatten_pixel_values(
        &self,
        resized_rgb_image: &RgbImage,
        image_grid: Qwen3_5ImageGrid,
        pixel_values_row_count: usize,
        pixel_values_column_count: usize,
        total_pixel_value_count: usize,
    ) -> Vec<f32> {
        let mut pixel_values = Vec::with_capacity(total_pixel_value_count);
        let resized_rgb_pixels = resized_rgb_image.as_raw();
        let resized_width_pixels = resized_rgb_image.width() as usize;
        let patch_size_pixels = self.patch_size_pixels as usize;
        let spatial_merge_size = self.spatial_merge_size as usize;
        let merged_patch_row_count = image_grid.height_patch_count as usize / spatial_merge_size;
        let merged_patch_column_count = image_grid.width_patch_count as usize / spatial_merge_size;

        // Packing order mirrors upstream reshape/transpose:
        // [grid_t, merged_h, merged_w, intra_h, intra_w, C, temporal, H, W].
        // The final four nested spatial loops keep all patches belonging to one
        // merger token adjacent, which later permits a zero-copy logical reshape.
        for _temporal_grid_index in 0..image_grid.temporal_patch_count {
            for merged_patch_row_index in 0..merged_patch_row_count {
                for merged_patch_column_index in 0..merged_patch_column_count {
                    for intra_merge_row_index in 0..spatial_merge_size {
                        for intra_merge_column_index in 0..spatial_merge_size {
                            let patch_row_index =
                                merged_patch_row_index * spatial_merge_size + intra_merge_row_index;
                            let patch_column_index = merged_patch_column_index * spatial_merge_size
                                + intra_merge_column_index;
                            self.append_one_flattened_patch(
                                resized_rgb_pixels,
                                resized_width_pixels,
                                patch_size_pixels,
                                patch_row_index,
                                patch_column_index,
                                &mut pixel_values,
                            );
                        }
                    }
                }
            }
        }

        debug_assert_eq!(
            pixel_values.len(),
            pixel_values_row_count * pixel_values_column_count
        );
        pixel_values
    }

    fn append_one_flattened_patch(
        &self,
        resized_rgb_pixels: &[u8],
        resized_width_pixels: usize,
        patch_size_pixels: usize,
        patch_row_index: usize,
        patch_column_index: usize,
        pixel_values: &mut Vec<f32>,
    ) {
        let temporal_patch_size = self.temporal_patch_size as usize;
        // Flatten one patch as C,T,H,W. For still images the same RGB source is
        // read for every temporal slot, exactly matching upstream temporal
        // duplication before patchification.
        for channel_index in 0..QWEN3_5_MOE_IMAGE_CHANNEL_COUNT as usize {
            for _temporal_patch_index in 0..temporal_patch_size {
                for patch_pixel_row_offset in 0..patch_size_pixels {
                    for patch_pixel_column_offset in 0..patch_size_pixels {
                        let image_pixel_row_index =
                            patch_row_index * patch_size_pixels + patch_pixel_row_offset;
                        let image_pixel_column_index =
                            patch_column_index * patch_size_pixels + patch_pixel_column_offset;
                        let rgb_storage_index = (image_pixel_row_index * resized_width_pixels
                            + image_pixel_column_index)
                            * QWEN3_5_MOE_IMAGE_CHANNEL_COUNT as usize
                            + channel_index;
                        // Checkpoint preprocessing is `(pixel / 255 - 0.5) / 0.5`,
                        // mapping byte range [0,255] to approximately [-1,1]. Keep
                        // the two operations explicit to match upstream Float32 order.
                        let rescaled_channel =
                            f32::from(resized_rgb_pixels[rgb_storage_index]) * self.rescale_factor;
                        let normalized_channel = (rescaled_channel
                            - self.image_mean_by_channel[channel_index])
                            / self.image_std_by_channel[channel_index];
                        pixel_values.push(normalized_channel);
                    }
                }
            }
        }
    }

    fn pixel_values_row_count(
        &self,
        image_grid: Qwen3_5ImageGrid,
    ) -> Result<usize, Qwen3_5ImageProcessingError> {
        let row_count = u64::from(image_grid.temporal_patch_count)
            .checked_mul(u64::from(image_grid.height_patch_count))
            .and_then(|temporal_height_count| {
                temporal_height_count.checked_mul(u64::from(image_grid.width_patch_count))
            })
            .ok_or_else(|| Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "image grid row count overflows u64".to_owned(),
            })?;
        usize::try_from(row_count).map_err(|_conversion_error| {
            Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "image grid row count overflows this platform".to_owned(),
            }
        })
    }

    fn pixel_values_column_count(&self) -> Result<usize, Qwen3_5ImageProcessingError> {
        let column_count = u64::from(QWEN3_5_MOE_IMAGE_CHANNEL_COUNT)
            .checked_mul(u64::from(self.temporal_patch_size))
            .and_then(|channel_temporal_count| {
                channel_temporal_count.checked_mul(u64::from(self.patch_size_pixels))
            })
            .and_then(|channel_temporal_height_count| {
                channel_temporal_height_count.checked_mul(u64::from(self.patch_size_pixels))
            })
            .ok_or_else(|| Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "pixel value column count overflows u64".to_owned(),
            })?;
        usize::try_from(column_count).map_err(|_conversion_error| {
            Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "pixel value column count overflows this platform".to_owned(),
            }
        })
    }

    fn image_token_count_after_spatial_merge(
        &self,
        pixel_values_row_count: usize,
    ) -> Result<usize, Qwen3_5ImageProcessingError> {
        let merge_area = self
            .spatial_merge_size
            .checked_mul(self.spatial_merge_size)
            .ok_or_else(|| Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "spatial merge area overflows u32".to_owned(),
            })? as usize;

        if !pixel_values_row_count.is_multiple_of(merge_area) {
            return Err(Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "image grid is not divisible by spatial merge area".to_owned(),
            });
        }

        Ok(pixel_values_row_count / merge_area)
    }

    fn validate_config(&self) -> Result<(), Qwen3_5ImageProcessingError> {
        if self.patch_size_pixels == 0 {
            return Err(Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "patch size must be positive".to_owned(),
            });
        }
        if self.temporal_patch_size == 0 {
            return Err(Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "temporal patch size must be positive".to_owned(),
            });
        }
        if self.spatial_merge_size == 0 {
            return Err(Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "spatial merge size must be positive".to_owned(),
            });
        }
        if self.minimum_image_pixels > self.maximum_image_pixels {
            return Err(Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "minimum image pixels must not exceed maximum image pixels".to_owned(),
            });
        }
        if self.image_std_by_channel.contains(&0.0) {
            return Err(Qwen3_5ImageProcessingError::InvalidProcessorConfig {
                message: "image standard deviation must be non-zero for every channel".to_owned(),
            });
        }
        Ok(())
    }
}

fn round_to_nearest_factor(pixel_count: u32, resize_factor_pixels: u32) -> u32 {
    ((f64::from(pixel_count) / f64::from(resize_factor_pixels)).round() as u32)
        * resize_factor_pixels
}

fn shrink_to_factor(pixel_count: u32, shrink_ratio: f64, resize_factor_pixels: u32) -> u32 {
    let resized_pixel_count =
        ((f64::from(pixel_count) / shrink_ratio / f64::from(resize_factor_pixels)).floor() as u32)
            * resize_factor_pixels;
    resized_pixel_count.max(resize_factor_pixels)
}

fn grow_to_factor(pixel_count: u32, grow_ratio: f64, resize_factor_pixels: u32) -> u32 {
    ((f64::from(pixel_count) * grow_ratio / f64::from(resize_factor_pixels)).ceil() as u32)
        * resize_factor_pixels
}

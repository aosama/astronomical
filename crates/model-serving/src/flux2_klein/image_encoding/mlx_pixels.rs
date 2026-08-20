//! One contiguous float32 host transfer followed by the exact reference pixel conversion.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::{
    Flux2KleinImageEncodingError, Flux2KleinPngEncoder, reference_pixels::normalized_rgb_u8,
};

impl Flux2KleinPngEncoder {
    pub fn encode_decoded_mlx_rgb(
        runtime: &MlxRuntime,
        decoded_channel_last_rgb: &MlxArray,
        width_pixels: u32,
        height_pixels: u32,
    ) -> Result<Vec<u8>, Flux2KleinImageEncodingError> {
        let mut attribution = PerformanceAttribution::disabled();
        Self::encode_decoded_mlx_rgb_with_performance_attribution(
            runtime,
            decoded_channel_last_rgb,
            width_pixels,
            height_pixels,
            &mut attribution,
        )
    }

    pub fn encode_decoded_mlx_rgb_with_performance_attribution(
        runtime: &MlxRuntime,
        decoded_channel_last_rgb: &MlxArray,
        width_pixels: u32,
        height_pixels: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Vec<u8>, Flux2KleinImageEncodingError> {
        let expected_shape = [
            1,
            i32::try_from(height_pixels)
                .map_err(|_| Flux2KleinImageEncodingError::GeometryOverflow)?,
            i32::try_from(width_pixels)
                .map_err(|_| Flux2KleinImageEncodingError::GeometryOverflow)?,
            3,
        ];
        if decoded_channel_last_rgb.shape() != expected_shape {
            return Err(Flux2KleinImageEncodingError::InvalidRgbGeometry);
        }
        let contiguous_pixels = performance_attribution.measure_operation(
            PerformanceOperation::ImagePixelConversionGraphConstruction,
            |_| {
                let half_scaled = runtime.multiply_scalar(decoded_channel_last_rgb, 0.5)?;
                let half = runtime.full(&[], 0.5, decoded_channel_last_rgb.dtype())?;
                let shifted = runtime.add(&half_scaled, &half)?;
                let normalized = runtime.clip(&shifted, 0.0, 1.0)?;
                let float32_pixels = runtime.astype(&normalized, MlxDtype::Float32)?;
                runtime.build_contiguous_row_major_copy(&float32_pixels)
            },
        )?;
        let rgb_bytes = performance_attribution.measure_operation(
            PerformanceOperation::ImagePixelTransfer,
            |_| {
                let normalized_values = contiguous_pixels.to_vec_f32()?;
                normalized_rgb_u8(&normalized_values)
            },
        )?;
        performance_attribution.measure_operation(PerformanceOperation::ImagePngEncoding, |_| {
            Self::encode_rgb8(width_pixels, height_pixels, &rgb_bytes)
        })
    }
}

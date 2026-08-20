//! Exact reversible mapping between transformer tokens and VAE spatial latents.

use super::Flux2KleinVaeError;
use crate::Flux2KleinImageDimensions;

pub const FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT: usize = 128;
pub const FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT: usize = 32;
const LATENT_PATCH_EDGE: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Flux2KleinPackedLatentLayout {
    batch_size: usize,
    packed_height: usize,
    packed_width: usize,
}

impl Flux2KleinPackedLatentLayout {
    pub fn for_image_dimensions(
        batch_size: usize,
        dimensions: &Flux2KleinImageDimensions,
    ) -> Result<Self, Flux2KleinVaeError> {
        Self::new(
            batch_size,
            usize::try_from(dimensions.height_pixels())
                .map_err(|_| Flux2KleinVaeError::latent_geometry("image height exceeds usize"))?
                / 16,
            usize::try_from(dimensions.width_pixels())
                .map_err(|_| Flux2KleinVaeError::latent_geometry("image width exceeds usize"))?
                / 16,
        )
    }

    pub fn new(
        batch_size: usize,
        packed_height: usize,
        packed_width: usize,
    ) -> Result<Self, Flux2KleinVaeError> {
        if batch_size == 0 || packed_height == 0 || packed_width == 0 {
            return Err(Flux2KleinVaeError::latent_geometry(
                "batch and packed spatial dimensions must be positive",
            ));
        }
        packed_height
            .checked_mul(packed_width)
            .and_then(|tokens| tokens.checked_mul(FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT))
            .and_then(|elements| elements.checked_mul(batch_size))
            .ok_or_else(|| Flux2KleinVaeError::latent_geometry("element count overflow"))?;
        Ok(Self {
            batch_size,
            packed_height,
            packed_width,
        })
    }

    #[must_use]
    pub fn packed_shape(&self) -> [usize; 3] {
        [
            self.batch_size,
            self.packed_height * self.packed_width,
            FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT,
        ]
    }

    #[must_use]
    pub const fn packed_spatial_shape(&self) -> [usize; 4] {
        [
            self.batch_size,
            self.packed_height,
            self.packed_width,
            FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT,
        ]
    }

    #[must_use]
    pub const fn unpatchified_shape(&self) -> [usize; 4] {
        [
            self.batch_size,
            self.packed_height * LATENT_PATCH_EDGE,
            self.packed_width * LATENT_PATCH_EDGE,
            FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
        ]
    }

    pub fn validate_packed_shape(&self, shape: &[i32]) -> Result<(), Flux2KleinVaeError> {
        let packed_shape = self.packed_shape();
        let expected = [
            i32::try_from(packed_shape[0])
                .map_err(|_| Flux2KleinVaeError::latent_geometry("batch exceeds i32"))?,
            i32::try_from(packed_shape[1])
                .map_err(|_| Flux2KleinVaeError::latent_geometry("tokens exceed i32"))?,
            i32::try_from(packed_shape[2])
                .map_err(|_| Flux2KleinVaeError::latent_geometry("channels exceed i32"))?,
        ];
        if shape != expected {
            return Err(Flux2KleinVaeError::latent_geometry(format!(
                "expected {:?}, received {shape:?}",
                self.packed_shape()
            )));
        }
        Ok(())
    }

    /// Returns the `[batch, token, packed_channel]` source for one NHWC latent value.
    #[must_use]
    pub fn unpatchified_source(
        &self,
        batch_index: usize,
        latent_row: usize,
        latent_column: usize,
        latent_channel: usize,
    ) -> Option<[usize; 3]> {
        let shape = self.unpatchified_shape();
        if batch_index >= shape[0]
            || latent_row >= shape[1]
            || latent_column >= shape[2]
            || latent_channel >= shape[3]
        {
            return None;
        }
        let packed_row = latent_row / LATENT_PATCH_EDGE;
        let packed_column = latent_column / LATENT_PATCH_EDGE;
        let patch_row = latent_row % LATENT_PATCH_EDGE;
        let patch_column = latent_column % LATENT_PATCH_EDGE;
        let packed_channel =
            ((latent_channel * LATENT_PATCH_EDGE + patch_row) * LATENT_PATCH_EDGE) + patch_column;
        Some([
            batch_index,
            packed_row * self.packed_width + packed_column,
            packed_channel,
        ])
    }
}

/// Scalar oracle for the inference-mode inverse BatchNorm used around packed latents.
pub fn flux2_klein_inverse_batch_norm_reference(
    packed_values: &[f32],
    channel_count: usize,
    running_mean: &[f32],
    running_variance: &[f32],
) -> Result<Vec<f32>, Flux2KleinVaeError> {
    if channel_count == 0
        || !packed_values.len().is_multiple_of(channel_count)
        || running_mean.len() != channel_count
        || running_variance.len() != channel_count
    {
        return Err(Flux2KleinVaeError::latent_geometry(
            "BatchNorm values and running statistics must share the channel geometry",
        ));
    }
    let mut restored_values = Vec::with_capacity(packed_values.len());
    for (value_index, packed_value) in packed_values.iter().copied().enumerate() {
        let channel_index = value_index % channel_count;
        let variance = running_variance[channel_index];
        if !packed_value.is_finite()
            || !running_mean[channel_index].is_finite()
            || !variance.is_finite()
            || variance < 0.0
        {
            return Err(Flux2KleinVaeError::latent_geometry(
                "BatchNorm inputs must be finite and variances nonnegative",
            ));
        }
        restored_values
            .push(packed_value * (variance + 0.000_1_f32).sqrt() + running_mean[channel_index]);
    }
    Ok(restored_values)
}

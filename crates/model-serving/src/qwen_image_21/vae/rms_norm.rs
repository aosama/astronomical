//! The VAE's RMS normalization expressed over channel-last MLX activations.
//!
//! Port of the reference `QwenImage21RMS_norm` (`F.normalize(x, dim=channels) * sqrt(dim) *
//! gamma`): the artifact stores only `gamma`, the bias defaults to zero, and `F.normalize`
//! divides by the L2 norm clamped from below at `1e-12`.
//!
//! This is the VAE's norm, which divides by the L2 norm. It is a different operator from the text
//! encoder's zero-centered RMSNorm (`qwen_image_21::norm` on the CPU, `mlx_math::fp32_zero_center_rms_norm`
//! on the runtime path), which divides by the root-mean-square and scales by `weight + 1`.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxSafetensors};

use super::QwenImage21VaeError;
use super::tensor_shape::{as_i32, validate_shape};

/// `torch.nn.functional.normalize` clamps the L2 norm at this floor before dividing.
const NORMALIZE_L2_FLOOR: f32 = 0.000_000_000_001;

#[derive(Debug)]
pub(super) struct QwenImage21VaeRmsNorm {
    channels: usize,
    scale: f32,
    gamma_float32: MlxArray,
}

impl QwenImage21VaeRmsNorm {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, QwenImage21VaeError> {
        let gamma = tensors.tensor(&format!("{prefix}.gamma"))?;
        validate_shape(prefix, "gamma", &gamma, &[channels])?;
        Ok(Self {
            channels,
            scale: (channels as f64).sqrt() as f32,
            gamma_float32: runtime.astype(&gamma, MlxDtype::Float32)?,
        })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        let shape = input.shape();
        if shape.len() != 4 || shape[3] != as_i32(self.channels, "RMS norm channels")? {
            return Err(QwenImage21VaeError::invalid_geometry(format!(
                "RMS norm expected NHWC channels {}, received {shape:?}",
                self.channels
            )));
        }
        // The channel axis is the last NHWC axis; the reference normalizes over channels too.
        let squared = runtime.multiply(input, input)?;
        let summed_squares = runtime.sum_axis(&squared, 3, true)?;
        let l2_norm = runtime.sqrt(&summed_squares)?;
        let floored_norm = runtime.clip(&l2_norm, NORMALIZE_L2_FLOOR, f32::MAX)?;
        // Evaluating the reduction first keeps the squared and summed tensors out of the graph
        // that the division and scaling would otherwise hold alive at full resolution.
        floored_norm.evaluate()?;
        let normalized = runtime.divide(input, &floored_norm)?;
        let scaled = runtime.multiply_scalar(&normalized, self.scale)?;
        let normalized_and_scaled = runtime.multiply(&scaled, &self.gamma_float32)?;
        let output = runtime.astype(&normalized_and_scaled, input.dtype())?;
        output.evaluate()?;
        Ok(output)
    }
}

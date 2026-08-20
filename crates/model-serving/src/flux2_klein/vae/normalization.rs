//! PyTorch-compatible GroupNorm expressed through MLX's fused non-affine LayerNorm.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxSafetensors};

use super::Flux2KleinVaeError;
use super::convolution::{as_i32, validate_shape};

const GROUP_NORM_EPSILON: f32 = 1e-6;
const GROUP_COUNT: usize = 32;

#[derive(Debug)]
pub(super) struct Flux2KleinGroupNorm {
    channels: usize,
    weight_float32: MlxArray,
    bias_float32: MlxArray,
}

impl Flux2KleinGroupNorm {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, Flux2KleinVaeError> {
        if !channels.is_multiple_of(GROUP_COUNT) {
            return Err(Flux2KleinVaeError::latent_geometry(
                "GroupNorm channels must be divisible by 32",
            ));
        }
        let weight = tensors.tensor(&format!("{prefix}.weight"))?;
        let bias = tensors.tensor(&format!("{prefix}.bias"))?;
        validate_shape(prefix, "weight", &weight, &[channels])?;
        validate_shape(prefix, "bias", &bias, &[channels])?;
        Ok(Self {
            channels,
            weight_float32: runtime.astype(&weight, MlxDtype::Float32)?,
            bias_float32: runtime.astype(&bias, MlxDtype::Float32)?,
        })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, Flux2KleinVaeError> {
        let shape = input.shape();
        if shape.len() != 4 || shape[3] != as_i32(self.channels, "GroupNorm channels")? {
            return Err(Flux2KleinVaeError::latent_geometry(format!(
                "GroupNorm expected NHWC channels {}, received {shape:?}",
                self.channels
            )));
        }
        let batch = shape[0];
        let spatial = shape[1]
            .checked_mul(shape[2])
            .ok_or_else(|| Flux2KleinVaeError::latent_geometry("GroupNorm spatial overflow"))?;
        let groups = as_i32(GROUP_COUNT, "GroupNorm group count")?;
        let group_width = as_i32(self.channels / GROUP_COUNT, "GroupNorm group width")?;
        let float_input = runtime.astype(input, MlxDtype::Float32)?;
        let grouped = runtime.reshape(&float_input, &[batch, spatial, groups, group_width])?;
        let pytorch_grouped = runtime.transpose_axes(&grouped, &[0, 2, 1, 3])?;
        let normalization_width = spatial
            .checked_mul(group_width)
            .ok_or_else(|| Flux2KleinVaeError::latent_geometry("GroupNorm width overflow"))?;
        let rows = runtime.reshape(&pytorch_grouped, &[batch, groups, normalization_width])?;
        let normalized_rows =
            runtime.layer_norm_without_weight_and_bias(&rows, GROUP_NORM_EPSILON)?;
        let grouped_output =
            runtime.reshape(&normalized_rows, &[batch, groups, spatial, group_width])?;
        let spatial_output = runtime.transpose_axes(&grouped_output, &[0, 2, 1, 3])?;
        let normalized =
            runtime.reshape(&spatial_output, &[shape[0], shape[1], shape[2], shape[3]])?;
        let scaled = runtime.multiply(&normalized, &self.weight_float32)?;
        let affine_float32 = runtime.add(&scaled, &self.bias_float32)?;
        Ok(runtime.astype(&affine_float32, input.dtype())?)
    }
}

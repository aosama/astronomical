//! Channel-last MLX convolution with one-time PyTorch weight-layout conversion.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use super::Flux2KleinVaeError;

#[derive(Debug)]
pub(super) struct Flux2KleinChannelLastConv2d {
    weight: MlxArray,
    bias: MlxArray,
    padding: i32,
}

impl Flux2KleinChannelLastConv2d {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel_edge: usize,
        padding: i32,
    ) -> Result<Self, Flux2KleinVaeError> {
        let pytorch_weight = tensors.tensor(&format!("{prefix}.weight"))?;
        let bias = tensors.tensor(&format!("{prefix}.bias"))?;
        validate_shape(
            prefix,
            "weight",
            &pytorch_weight,
            &[output_channels, input_channels, kernel_edge, kernel_edge],
        )?;
        validate_shape(prefix, "bias", &bias, &[output_channels])?;
        // PyTorch stores OIHW while MLX conv2d consumes OHWI.
        let weight = runtime.transpose_axes(&pytorch_weight, &[0, 2, 3, 1])?;
        Ok(Self {
            weight,
            bias,
            padding,
        })
    }

    pub(super) fn load_linear_as_pointwise(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, Flux2KleinVaeError> {
        let linear_weight = tensors.tensor(&format!("{prefix}.weight"))?;
        let bias = tensors.tensor(&format!("{prefix}.bias"))?;
        validate_shape(prefix, "weight", &linear_weight, &[channels, channels])?;
        validate_shape(prefix, "bias", &bias, &[channels])?;
        let channels_i32 = as_i32(channels, "attention channel count")?;
        let weight = runtime.reshape(&linear_weight, &[channels_i32, 1, 1, channels_i32])?;
        Ok(Self {
            weight,
            bias,
            padding: 0,
        })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        channel_last_input: &MlxArray,
    ) -> Result<MlxArray, Flux2KleinVaeError> {
        let convolution = runtime.conv2d(
            channel_last_input,
            &self.weight,
            [1, 1],
            [self.padding, self.padding],
            [1, 1],
            1,
        )?;
        Ok(runtime.add(&convolution, &self.bias)?)
    }
}

pub(super) fn validate_shape(
    prefix: &str,
    tensor_role: &str,
    tensor: &MlxArray,
    expected_shape: &[usize],
) -> Result<(), Flux2KleinVaeError> {
    let expected_i32 = expected_shape
        .iter()
        .map(|dimension| as_i32(*dimension, "weight dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    if tensor.shape() != expected_i32 {
        return Err(Flux2KleinVaeError::latent_geometry(format!(
            "{prefix}.{tensor_role} expected shape {expected_shape:?}, received {:?}",
            tensor.shape()
        )));
    }
    Ok(())
}

pub(super) fn as_i32(value: usize, role: &str) -> Result<i32, Flux2KleinVaeError> {
    i32::try_from(value).map_err(|_| {
        Flux2KleinVaeError::latent_geometry(format!("{role} exceeds the MLX integer range"))
    })
}

//! Channel-last MLX spatial convolution loading the reviewed artifact's MLX weights directly.
//!
//! The reviewed `mlx-community` artifact stores every VAE convolution as `[out, kh, kw, in]`
//! (`mlx_format: true`), which is exactly the OHWI order MLX conv2d consumes, so no PyTorch
//! OIHW transpose is needed here (unlike the FLUX.2 Klein loader).

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use super::QwenImage21VaeError;
use super::tensor_shape::validate_shape;

/// A spatial (or pointwise) convolution over `NHWC` activations with `OHWI` weights.
#[derive(Debug)]
pub(super) struct QwenImage21VaeSpatialConv {
    weight: MlxArray,
    bias: MlxArray,
    padding: i32,
}

impl QwenImage21VaeSpatialConv {
    pub(super) fn load(
        tensors: &MlxSafetensors,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel_edge: usize,
        padding: i32,
    ) -> Result<Self, QwenImage21VaeError> {
        let weight = tensors.tensor(&format!("{prefix}.weight"))?;
        let bias = tensors.tensor(&format!("{prefix}.bias"))?;
        validate_shape(
            prefix,
            "weight",
            &weight,
            &[output_channels, kernel_edge, kernel_edge, input_channels],
        )?;
        validate_shape(prefix, "bias", &bias, &[output_channels])?;
        Ok(Self {
            weight,
            bias,
            padding,
        })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        channel_last_input: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        // Convolution and bias are separate evaluations so the unbroadcast convolution result is
        // released before the bias broadcast exists — the decode's peak lives in these tensors.
        let convolution = runtime.conv2d(
            channel_last_input,
            &self.weight,
            [1, 1],
            [self.padding, self.padding],
            [1, 1],
            1,
        )?;
        convolution.evaluate()?;
        let biased_output = runtime.add(&convolution, &self.bias)?;
        biased_output.evaluate()?;
        Ok(biased_output)
    }
}

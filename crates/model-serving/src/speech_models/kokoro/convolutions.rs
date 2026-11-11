//! Kokoro's convolutions over `[length, channels]`, padded symmetrically as
//! StyleTTS2 and HiFi-GAN pad them (each output sees past and future).

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::convolution_weight_layout::{
    convolution_weight_for_mlx, pytorch_convolution_weight, transposed_convolution_weight_for_mlx,
};
use crate::speech_models::speech_model_error::SpeechModelError;

/// How one convolution slides: stride, symmetric padding, dilation, groups.
#[derive(Clone, Copy)]
pub(crate) struct ConvolutionGeometry {
    pub(crate) stride: i32,
    pub(crate) padding: i32,
    pub(crate) dilation: i32,
    pub(crate) groups: i32,
}

impl ConvolutionGeometry {
    /// Stride 1, the padding that keeps the length, no grouping.
    pub(crate) fn same_length(kernel_size: i32, dilation: i32) -> Self {
        Self {
            stride: 1,
            padding: (kernel_size * dilation - dilation) / 2,
            dilation,
            groups: 1,
        }
    }
}

pub(crate) struct KokoroConv1d {
    /// `[out, kernel, in / groups]`.
    weight: MlxArray,
    bias: Option<MlxArray>,
    geometry: ConvolutionGeometry,
}

impl KokoroConv1d {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        geometry: ConvolutionGeometry,
    ) -> Result<Self, SpeechModelError> {
        let pytorch_weight = pytorch_convolution_weight(checkpoint, prefix)?;
        Ok(Self {
            weight: convolution_weight_for_mlx(checkpoint.runtime, &pytorch_weight)?,
            bias: checkpoint.optional_tensor(&format!("{prefix}.bias"))?,
            geometry,
        })
    }

    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let batched_input = runtime.expand_dims(input, 0)?;
        let ConvolutionGeometry {
            stride,
            padding,
            dilation,
            groups,
        } = self.geometry;
        let convolved = runtime.conv1d(
            &batched_input,
            &self.weight,
            stride,
            padding,
            dilation,
            groups,
        )?;
        let unbatched = runtime.squeeze_axis(&convolved, 0)?;
        match &self.bias {
            Some(bias) => Ok(runtime.add(&unbatched, bias)?),
            None => Ok(unbatched),
        }
    }
}

/// A transposed convolution (upsampling) with PyTorch's padding semantics.
pub(crate) struct KokoroConvTranspose1d {
    /// `[out, kernel, in / groups]`.
    weight: MlxArray,
    bias: MlxArray,
    geometry: ConvolutionGeometry,
    output_padding: i32,
}

impl KokoroConvTranspose1d {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        geometry: ConvolutionGeometry,
        output_padding: i32,
    ) -> Result<Self, SpeechModelError> {
        let pytorch_weight = pytorch_convolution_weight(checkpoint, prefix)?;
        Ok(Self {
            weight: transposed_convolution_weight_for_mlx(
                checkpoint.runtime,
                &pytorch_weight,
                geometry.groups,
            )?,
            bias: checkpoint.tensor(&format!("{prefix}.bias"))?,
            geometry,
            output_padding,
        })
    }

    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let batched_input = runtime.expand_dims(input, 0)?;
        let ConvolutionGeometry {
            stride,
            padding,
            dilation,
            groups,
        } = self.geometry;
        let upsampled = runtime.conv_transpose1d(
            &batched_input,
            &self.weight,
            stride,
            padding,
            dilation,
            self.output_padding,
            groups,
        )?;
        Ok(runtime.add(&runtime.squeeze_axis(&upsampled, 0)?, &self.bias)?)
    }
}

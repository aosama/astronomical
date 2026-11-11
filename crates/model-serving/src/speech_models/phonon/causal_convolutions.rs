//! Phonon's causal convolutions over `[length, channels]`, as Mimi's
//! streaming convolutions behave on a whole utterance that started from
//! silence: each output depends only on the past.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::convolution_weight_layout::{
    convolution_weight_for_mlx, transposed_convolution_weight_for_mlx,
};
use crate::speech_models::speech_model_error::SpeechModelError;

/// How one causal convolution slides.
#[derive(Clone, Copy)]
pub(crate) struct CausalGeometry {
    pub(crate) stride: i32,
    pub(crate) dilation: i32,
    pub(crate) groups: i32,
}

impl CausalGeometry {
    pub(crate) const PLAIN: Self = Self {
        stride: 1,
        dilation: 1,
        groups: 1,
    };
}

/// Zeros pad the start by the kernel's reach less the stride, which is what
/// a streaming convolution's zero-initialised history amounts to.
pub(crate) struct CausalConv1d {
    /// `[out, kernel, in / groups]`.
    weight: MlxArray,
    bias: Option<MlxArray>,
    geometry: CausalGeometry,
    left_padding: i32,
}

impl CausalConv1d {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        geometry: CausalGeometry,
    ) -> Result<Self, SpeechModelError> {
        let pytorch_weight = checkpoint.tensor(&format!("{prefix}.weight"))?;
        let kernel_size = pytorch_weight.shape()[2];
        let kernel_reach = (kernel_size - 1) * geometry.dilation + 1;
        Ok(Self {
            weight: convolution_weight_for_mlx(checkpoint.runtime, &pytorch_weight)?,
            bias: checkpoint.optional_tensor(&format!("{prefix}.bias"))?,
            geometry,
            left_padding: kernel_reach - geometry.stride,
        })
    }

    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let channel_count = input.shape()[1];
        let padded = if self.left_padding > 0 {
            let silence = runtime.zeros(&[self.left_padding, channel_count], MlxDtype::Float32)?;
            runtime.concatenate_axis(&[&silence, input], 0)?
        } else {
            input.retain()?
        };
        let CausalGeometry {
            stride,
            dilation,
            groups,
        } = self.geometry;
        let convolved = runtime.conv1d(
            &runtime.expand_dims(&padded, 0)?,
            &self.weight,
            stride,
            0,
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

/// Every input step adds `kernel` samples starting at `step * stride`; the
/// last `kernel - stride` samples, which a stream would still be waiting to
/// complete, are dropped, so the output is exactly `stride` times as long.
pub(crate) struct CausalConvTranspose1d {
    /// `[out, kernel, in / groups]`.
    weight: MlxArray,
    bias: Option<MlxArray>,
    stride: i32,
    groups: i32,
}

impl CausalConvTranspose1d {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        stride: i32,
        groups: i32,
    ) -> Result<Self, SpeechModelError> {
        let pytorch_weight = checkpoint.tensor(&format!("{prefix}.weight"))?;
        Ok(Self {
            weight: transposed_convolution_weight_for_mlx(
                checkpoint.runtime,
                &pytorch_weight,
                groups,
            )?,
            bias: checkpoint.optional_tensor(&format!("{prefix}.bias"))?,
            stride,
            groups,
        })
    }

    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let batched_input = runtime.expand_dims(input, 0)?;
        let full_output = runtime.conv_transpose1d(
            &batched_input,
            &self.weight,
            self.stride,
            0,
            1,
            0,
            self.groups,
        )?;
        let upsampled = runtime.squeeze_axis(&full_output, 0)?;
        let kept_length = input.shape()[0] * self.stride;
        let channel_count = upsampled.shape()[1];
        let kept = runtime.slice(&upsampled, &[0, 0], &[kept_length, channel_count], &[1, 1])?;
        match &self.bias {
            Some(bias) => Ok(runtime.add(&kept, bias)?),
            None => Ok(kept),
        }
    }
}

#[cfg(test)]
#[path = "causal_convolutions_tests.rs"]
mod tests;

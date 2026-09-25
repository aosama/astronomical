//! Single-head spatial self-attention for the Qwen-Image-2.1 VAE middle block.
//!
//! Port of `QwenImage21AttentionBlock`: RMS-normalize, project to fused Q/K/V with one 1×1
//! convolution, attend across the `H*W` positions of the (single) frame with one head, project
//! back with a second 1×1 convolution, and add the identity.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use super::QwenImage21VaeError;
use super::convolution::QwenImage21VaeSpatialConv;
use super::rms_norm::QwenImage21VaeRmsNorm;
use super::tensor_shape::as_i32;

#[derive(Debug)]
pub(super) struct QwenImage21VaeAttentionBlock {
    channels: usize,
    norm: QwenImage21VaeRmsNorm,
    fused_query_key_value: QwenImage21VaeSpatialConv,
    output_projection: QwenImage21VaeSpatialConv,
}

impl QwenImage21VaeAttentionBlock {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, QwenImage21VaeError> {
        Ok(Self {
            channels,
            norm: QwenImage21VaeRmsNorm::load(
                runtime,
                tensors,
                &format!("{prefix}.norm"),
                channels,
            )?,
            fused_query_key_value: QwenImage21VaeSpatialConv::load(
                tensors,
                &format!("{prefix}.to_qkv"),
                channels,
                channels * 3,
                1,
                0,
            )?,
            output_projection: QwenImage21VaeSpatialConv::load(
                tensors,
                &format!("{prefix}.proj"),
                channels,
                channels,
                1,
                0,
            )?,
        })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        let shape = input.shape();
        if shape.len() != 4 || shape[3] != as_i32(self.channels, "attention channels")? {
            return Err(QwenImage21VaeError::invalid_geometry(format!(
                "attention expected NHWC channels {}, received {shape:?}",
                self.channels
            )));
        }
        let (batch, height, width) = (shape[0], shape[1], shape[2]);
        let channels = as_i32(self.channels, "attention channels")?;
        let normalized = self.norm.forward(runtime, input)?;
        let fused = self.fused_query_key_value.forward(runtime, &normalized)?;
        // The fused axis is ordered query, key, value — matching the reference `chunk(3, dim=-1)`.
        let query = runtime.slice(
            &fused,
            &[0, 0, 0, 0],
            &[batch, height, width, channels],
            &[1, 1, 1, 1],
        )?;
        let key = runtime.slice(
            &fused,
            &[0, 0, 0, channels],
            &[batch, height, width, 2 * channels],
            &[1, 1, 1, 1],
        )?;
        let value = runtime.slice(
            &fused,
            &[0, 0, 0, 2 * channels],
            &[batch, height, width, 3 * channels],
            &[1, 1, 1, 1],
        )?;
        let sequence_length = height
            .checked_mul(width)
            .ok_or_else(|| QwenImage21VaeError::invalid_geometry("attention sequence overflow"))?;
        let head = |array: &MlxArray| -> Result<MlxArray, QwenImage21VaeError> {
            let tokens = runtime.reshape(array, &[batch, sequence_length, 1, channels])?;
            Ok(runtime.transpose_axes(&tokens, &[0, 2, 1, 3])?)
        };
        let query_heads = head(&query)?;
        let key_heads = head(&key)?;
        let value_heads = head(&value)?;
        let scale = 1.0 / (self.channels as f32).sqrt();
        let attended =
            runtime.scaled_dot_product_attention(&query_heads, &key_heads, &value_heads, scale)?;
        let attended_tokens = runtime.transpose_axes(&attended, &[0, 2, 1, 3])?;
        let attended_pixels =
            runtime.reshape(&attended_tokens, &[batch, height, width, channels])?;
        let projected = self.output_projection.forward(runtime, &attended_pixels)?;
        Ok(runtime.add(input, &projected)?)
    }
}

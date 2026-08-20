//! One-head spatial self-attention using MLX fused scaled dot-product attention.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use super::convolution::as_i32;
use super::{Flux2KleinChannelLastConv2d, Flux2KleinGroupNorm, Flux2KleinVaeError};

#[derive(Debug)]
pub(super) struct Flux2KleinVaeMiddleAttention {
    channels: usize,
    group_norm: Flux2KleinGroupNorm,
    query: Flux2KleinChannelLastConv2d,
    key: Flux2KleinChannelLastConv2d,
    value: Flux2KleinChannelLastConv2d,
    output: Flux2KleinChannelLastConv2d,
}

impl Flux2KleinVaeMiddleAttention {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, Flux2KleinVaeError> {
        Ok(Self {
            channels,
            group_norm: Flux2KleinGroupNorm::load(
                runtime,
                tensors,
                &format!("{prefix}.group_norm"),
                channels,
            )?,
            query: Flux2KleinChannelLastConv2d::load_linear_as_pointwise(
                runtime,
                tensors,
                &format!("{prefix}.to_q"),
                channels,
            )?,
            key: Flux2KleinChannelLastConv2d::load_linear_as_pointwise(
                runtime,
                tensors,
                &format!("{prefix}.to_k"),
                channels,
            )?,
            value: Flux2KleinChannelLastConv2d::load_linear_as_pointwise(
                runtime,
                tensors,
                &format!("{prefix}.to_v"),
                channels,
            )?,
            output: Flux2KleinChannelLastConv2d::load_linear_as_pointwise(
                runtime,
                tensors,
                &format!("{prefix}.to_out.0"),
                channels,
            )?,
        })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, Flux2KleinVaeError> {
        let shape = input.shape();
        if shape.len() != 4 || shape[3] != as_i32(self.channels, "attention channels")? {
            return Err(Flux2KleinVaeError::latent_geometry(format!(
                "middle attention expected NHWC channels {}, received {shape:?}",
                self.channels
            )));
        }
        let normalized = self.group_norm.forward(runtime, input)?;
        let query = self.query.forward(runtime, &normalized)?;
        let key = self.key.forward(runtime, &normalized)?;
        let value = self.value.forward(runtime, &normalized)?;
        let sequence_length = shape[1]
            .checked_mul(shape[2])
            .ok_or_else(|| Flux2KleinVaeError::latent_geometry("attention sequence overflow"))?;
        let channels = as_i32(self.channels, "attention channels")?;
        let query_heads = runtime.transpose_axes(
            &runtime.reshape(&query, &[shape[0], sequence_length, 1, channels])?,
            &[0, 2, 1, 3],
        )?;
        let key_heads = runtime.transpose_axes(
            &runtime.reshape(&key, &[shape[0], sequence_length, 1, channels])?,
            &[0, 2, 1, 3],
        )?;
        let value_heads = runtime.transpose_axes(
            &runtime.reshape(&value, &[shape[0], sequence_length, 1, channels])?,
            &[0, 2, 1, 3],
        )?;
        let channels_float = f32::from(u16::try_from(self.channels).map_err(|_| {
            Flux2KleinVaeError::latent_geometry("attention channels exceed exact f32 range")
        })?);
        let attended_heads = runtime.scaled_dot_product_attention(
            &query_heads,
            &key_heads,
            &value_heads,
            1.0 / channels_float.sqrt(),
        )?;
        let attended_sequence = runtime.transpose_axes(&attended_heads, &[0, 2, 1, 3])?;
        let attended = runtime.reshape(
            &attended_sequence,
            &[shape[0], shape[1], shape[2], channels],
        )?;
        let projected = self.output.forward(runtime, &attended)?;
        Ok(runtime.add(input, &projected)?)
    }
}

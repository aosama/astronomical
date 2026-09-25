//! Residual up block: three residual stages, then nearest 2× upsampling with a channel shuffle
//! shortcut.
//!
//! Port of `QwenImage21ResidualUpBlock` (the artifact sets `is_residual: true`) for the
//! single-frame image decode. Two reference details collapse on this path:
//!
//! 1. The `upsampler.time_conv` and its channel-interleave temporal doubling only run when the
//!    feature cache already holds a state, and `_decode` drives one frame through a freshly
//!    cleared cache — so a whole-image decode never executes the temporal path, and the
//!    upsampler is exactly nearest 2× spatial followed by the stored 3×3 convolution.
//! 2. The parameter-free `avg_shortcut` (`QwenImage21DupUp3D`, `first_chunk = true`) reduces to a
//!    pixel shuffle whose source channel for output channel `oc` and intra-tile position
//!    `(a, b)` is `(oc * factor + (factor_t - 1) * 4 + a * 2 + b) // repeats`. That mapping is
//!    derived from the loaded channel counts at load time, not hardcoded.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use super::QwenImage21VaeError;
use super::convolution::QwenImage21VaeSpatialConv;
use super::resnet::QwenImage21VaeResnetBlock;
use super::tensor_shape::as_i32;

/// Nearest-neighbor replication factor per spatial axis (`factor_s` in the reference).
const SPATIAL_UPSAMPLE_FACTOR: usize = 2;

/// The parameter-free pixel-shuffle shortcut the reference calls `QwenImage21DupUp3D`.
#[derive(Debug)]
pub(super) struct QwenImage21VaeDupUpShortcut {
    output_channels: usize,
    temporal_factor: usize,
    channel_repeats: usize,
}

impl QwenImage21VaeDupUpShortcut {
    pub(super) fn load(
        input_channels: usize,
        output_channels: usize,
        temporal_upsample: bool,
    ) -> Result<Self, QwenImage21VaeError> {
        let temporal_factor = if temporal_upsample { 2 } else { 1 };
        let factor = temporal_factor * SPATIAL_UPSAMPLE_FACTOR * SPATIAL_UPSAMPLE_FACTOR;
        if output_channels * factor % input_channels != 0 {
            return Err(QwenImage21VaeError::invalid_geometry(format!(
                "dup-up shortcut needs {output_channels} * {factor} to divide by {input_channels}"
            )));
        }
        Ok(Self {
            output_channels,
            temporal_factor,
            channel_repeats: output_channels * factor / input_channels,
        })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        let shape = input.shape();
        let (batch, height, width) = (shape[0], shape[1], shape[2]);
        let input_channels = shape[3];
        let channel_repeats = as_i32(self.channel_repeats, "dup-up channel repeats")?;
        let output_channels = as_i32(self.output_channels, "dup-up output channels")?;
        let temporal_factor = as_i32(self.temporal_factor, "dup-up temporal factor")?;

        // `repeat_interleave(repeats, dim=channel)`: channel j becomes `repeats` copies of
        // channel j // repeats.
        let column = runtime.reshape(input, &[batch, height, width, input_channels, 1])?;
        let replicated = runtime.broadcast_to(
            &column,
            &[batch, height, width, input_channels, channel_repeats],
        )?;
        let interleaved = runtime.reshape(
            &replicated,
            &[batch, height, width, input_channels * channel_repeats],
        )?;
        // Factor the expanded channel axis exactly as the reference `view` does.
        let factored = runtime.reshape(
            &interleaved,
            &[batch, height, width, output_channels, temporal_factor, 2, 2],
        )?;
        // `first_chunk` keeps only the final temporal slot; a whole-image decode has exactly one
        // frame, so this leaves one output frame. The slice keeps the slot as a size-1 axis, and
        // the reshape below consumes it.
        let kept_temporal_slot = runtime.slice(
            &factored,
            &[0, 0, 0, 0, temporal_factor - 1, 0, 0],
            &[batch, height, width, output_channels, temporal_factor, 2, 2],
            &[1, 1, 1, 1, 1, 1, 1],
        )?;
        kept_temporal_slot.evaluate()?;
        let frame = runtime.reshape(
            &kept_temporal_slot,
            &[batch, height, width, output_channels, 2, 2],
        )?;
        // Arrange the intra-tile positions next to the pixel axes they expand, with the output
        // channel last: `(h, a) → row`, `(w, b) → column`.
        let arranged = runtime.transpose_axes(&frame, &[0, 1, 4, 2, 5, 3])?;
        let rows = height
            .checked_mul(as_i32(SPATIAL_UPSAMPLE_FACTOR, "spatial factor")?)
            .ok_or_else(|| QwenImage21VaeError::invalid_geometry("dup-up row overflow"))?;
        let columns = width
            .checked_mul(as_i32(SPATIAL_UPSAMPLE_FACTOR, "spatial factor")?)
            .ok_or_else(|| QwenImage21VaeError::invalid_geometry("dup-up column overflow"))?;
        let upsampled = runtime.reshape(&arranged, &[batch, rows, columns, output_channels])?;
        upsampled.evaluate()?;
        Ok(upsampled)
    }
}

#[derive(Debug)]
pub(super) struct QwenImage21VaeResidualUpBlock {
    resnets: [QwenImage21VaeResnetBlock; 3],
    upsampler_convolution: Option<QwenImage21VaeSpatialConv>,
    shortcut: Option<QwenImage21VaeDupUpShortcut>,
}

impl QwenImage21VaeResidualUpBlock {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        block_index: usize,
        input_channels: usize,
        output_channels: usize,
        temporal_upsample: bool,
        has_upsampler: bool,
    ) -> Result<Self, QwenImage21VaeError> {
        let prefix = format!("decoder.up_blocks.{block_index}");
        let first_resnet_input = input_channels;
        let resnets = [
            QwenImage21VaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{prefix}.resnets.0"),
                first_resnet_input,
                output_channels,
            )?,
            QwenImage21VaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{prefix}.resnets.1"),
                output_channels,
                output_channels,
            )?,
            QwenImage21VaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{prefix}.resnets.2"),
                output_channels,
                output_channels,
            )?,
        ];
        let upsampler_convolution = has_upsampler
            .then(|| {
                QwenImage21VaeSpatialConv::load(
                    tensors,
                    &format!("{prefix}.upsampler.resample.1"),
                    output_channels,
                    output_channels,
                    3,
                    1,
                )
            })
            .transpose()?;
        let shortcut = has_upsampler
            .then(|| {
                QwenImage21VaeDupUpShortcut::load(
                    input_channels,
                    output_channels,
                    temporal_upsample,
                )
            })
            .transpose()?;
        Ok(Self {
            resnets,
            upsampler_convolution,
            shortcut,
        })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        // The shortcut sees the block input, so it must be computed against the pre-resnet
        // tensor; MLX graph values are immutable, so the reference `x_copy` is just `input`.
        let shortcut = match (&self.shortcut, &self.upsampler_convolution) {
            (Some(dup_up), Some(_)) => Some(dup_up.forward(runtime, input)?),
            (None, None) => None,
            _ => {
                return Err(QwenImage21VaeError::invalid_geometry(
                    "up block requires the upsampler and its shortcut together",
                ));
            }
        };
        // Each stage is evaluated by the component it delegates to, so the peak stays at one
        // residual's activations instead of the whole block graph at full resolution.
        let mut hidden = self.resnets[0].forward(runtime, input)?;
        hidden = self.resnets[1].forward(runtime, &hidden)?;
        hidden = self.resnets[2].forward(runtime, &hidden)?;
        if let Some(upsampler) = &self.upsampler_convolution {
            let doubled_rows = runtime.repeat_axis(&hidden, 2, 1)?;
            doubled_rows.evaluate()?;
            let doubled_pixels = runtime.repeat_axis(&doubled_rows, 2, 2)?;
            doubled_pixels.evaluate()?;
            hidden = upsampler.forward(runtime, &doubled_pixels)?;
            hidden.evaluate()?;
        }
        match shortcut {
            Some(projected) => {
                let combined = runtime.add(&hidden, &projected)?;
                combined.evaluate()?;
                Ok(combined)
            }
            None => Ok(hidden),
        }
    }
}

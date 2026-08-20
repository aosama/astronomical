//! FLUX.2 decoder residual block with explicit optional channel projection.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use super::{Flux2KleinChannelLastConv2d, Flux2KleinGroupNorm, Flux2KleinVaeError};

#[derive(Debug)]
pub(super) struct Flux2KleinVaeResnetBlock {
    norm1: Flux2KleinGroupNorm,
    conv1: Flux2KleinChannelLastConv2d,
    norm2: Flux2KleinGroupNorm,
    conv2: Flux2KleinChannelLastConv2d,
    shortcut: Option<Flux2KleinChannelLastConv2d>,
}

impl Flux2KleinVaeResnetBlock {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self, Flux2KleinVaeError> {
        Ok(Self {
            norm1: Flux2KleinGroupNorm::load(
                runtime,
                tensors,
                &format!("{prefix}.norm1"),
                input_channels,
            )?,
            conv1: Flux2KleinChannelLastConv2d::load(
                runtime,
                tensors,
                &format!("{prefix}.conv1"),
                input_channels,
                output_channels,
                3,
                1,
            )?,
            norm2: Flux2KleinGroupNorm::load(
                runtime,
                tensors,
                &format!("{prefix}.norm2"),
                output_channels,
            )?,
            conv2: Flux2KleinChannelLastConv2d::load(
                runtime,
                tensors,
                &format!("{prefix}.conv2"),
                output_channels,
                output_channels,
                3,
                1,
            )?,
            shortcut: (input_channels != output_channels)
                .then(|| {
                    Flux2KleinChannelLastConv2d::load(
                        runtime,
                        tensors,
                        &format!("{prefix}.conv_shortcut"),
                        input_channels,
                        output_channels,
                        1,
                        0,
                    )
                })
                .transpose()?,
        })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, Flux2KleinVaeError> {
        let normalized1 = self.norm1.forward(runtime, input)?;
        let activated1 = runtime.silu(&normalized1)?;
        let convolved1 = self.conv1.forward(runtime, &activated1)?;
        let normalized2 = self.norm2.forward(runtime, &convolved1)?;
        let activated2 = runtime.silu(&normalized2)?;
        let convolved2 = self.conv2.forward(runtime, &activated2)?;
        match &self.shortcut {
            Some(shortcut) => {
                let projected_residual = shortcut.forward(runtime, input)?;
                Ok(runtime.add(&convolved2, &projected_residual)?)
            }
            None => Ok(runtime.add(&convolved2, input)?),
        }
    }
}

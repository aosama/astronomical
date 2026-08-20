//! Three residual layers followed by optional nearest-neighbor upsampling and convolution.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use super::{Flux2KleinChannelLastConv2d, Flux2KleinVaeError, Flux2KleinVaeResnetBlock};

#[derive(Debug)]
pub(super) struct Flux2KleinVaeUpBlock {
    resnets: [Flux2KleinVaeResnetBlock; 3],
    upsample: Option<Flux2KleinChannelLastConv2d>,
}

impl Flux2KleinVaeUpBlock {
    pub(super) fn load(
        runtime: &MlxRuntime,
        tensors: &MlxSafetensors,
        block_index: usize,
        input_channels: usize,
        output_channels: usize,
        has_upsample: bool,
    ) -> Result<Self, Flux2KleinVaeError> {
        let prefix = format!("decoder.up_blocks.{block_index}");
        let resnets = [
            Flux2KleinVaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{prefix}.resnets.0"),
                input_channels,
                output_channels,
            )?,
            Flux2KleinVaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{prefix}.resnets.1"),
                output_channels,
                output_channels,
            )?,
            Flux2KleinVaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{prefix}.resnets.2"),
                output_channels,
                output_channels,
            )?,
        ];
        let upsample = has_upsample
            .then(|| {
                Flux2KleinChannelLastConv2d::load(
                    runtime,
                    tensors,
                    &format!("{prefix}.upsamplers.0.conv"),
                    output_channels,
                    output_channels,
                    3,
                    1,
                )
            })
            .transpose()?;
        Ok(Self { resnets, upsample })
    }

    pub(super) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, Flux2KleinVaeError> {
        let mut hidden_states = self.resnets[0].forward(runtime, input)?;
        hidden_states = self.resnets[1].forward(runtime, &hidden_states)?;
        hidden_states = self.resnets[2].forward(runtime, &hidden_states)?;
        if let Some(upsample_convolution) = &self.upsample {
            let repeated_rows = runtime.repeat_axis(&hidden_states, 2, 1)?;
            let repeated_pixels = runtime.repeat_axis(&repeated_rows, 2, 2)?;
            hidden_states = upsample_convolution.forward(runtime, &repeated_pixels)?;
        }
        Ok(hidden_states)
    }
}

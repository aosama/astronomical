//! Complete native MLX FLUX.2 Klein decoder.

use std::fs::File;

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::convolution::validate_shape;
use super::{
    FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT, FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
    Flux2KleinChannelLastConv2d, Flux2KleinGroupNorm, Flux2KleinPackedLatentLayout,
    Flux2KleinVaeDecodeAdvance, Flux2KleinVaeError, Flux2KleinVaeMiddleAttention,
    Flux2KleinVaeResnetBlock, Flux2KleinVaeTilingConfig, Flux2KleinVaeUpBlock,
};

const DECODER_CHANNELS: [usize; 4] = [512, 512, 256, 128];
pub(super) const OUTPUT_CHANNELS: usize = 3;
pub(super) const BATCH_NORM_EPSILON: f32 = 0.000_1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flux2KleinVaeDecodeMode {
    Complete,
    /// Retained only as a acceptance seam; production rejects independent tiles.
    Tiled(Flux2KleinVaeTilingConfig),
}

#[derive(Debug)]
pub struct Flux2KleinVaeDecoder {
    pub(super) running_mean: MlxArray,
    pub(super) running_variance: MlxArray,
    pub(super) post_quant_conv: Flux2KleinChannelLastConv2d,
    pub(super) conv_in: Flux2KleinChannelLastConv2d,
    pub(super) middle_resnet_before_attention: Flux2KleinVaeResnetBlock,
    pub(super) middle_attention: Flux2KleinVaeMiddleAttention,
    pub(super) middle_resnet_after_attention: Flux2KleinVaeResnetBlock,
    pub(super) up_blocks: [Flux2KleinVaeUpBlock; 4],
    pub(super) output_norm: Flux2KleinGroupNorm,
    pub(super) conv_out: Flux2KleinChannelLastConv2d,
}

impl Flux2KleinVaeDecoder {
    pub fn load(runtime: &MlxRuntime, vae_weights_file: File) -> Result<Self, Flux2KleinVaeError> {
        let mut attribution = PerformanceAttribution::disabled();
        Self::load_with_performance_attribution(runtime, vae_weights_file, &mut attribution)
    }

    pub fn load_with_performance_attribution(
        runtime: &MlxRuntime,
        vae_weights_file: File,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, Flux2KleinVaeError> {
        let file_read_metrics = performance_attribution.positional_file_read_metrics();
        let tensors = performance_attribution
            .measure_operation(PerformanceOperation::ImageVaeComponentMapping, |_| {
                runtime.load_safetensors(vae_weights_file, file_read_metrics)
            })?;
        performance_attribution
            .measure_operation(PerformanceOperation::ImageVaeComponentLoading, |_| {
                Self::bind(runtime, &tensors)
            })
    }

    fn bind(runtime: &MlxRuntime, tensors: &MlxSafetensors) -> Result<Self, Flux2KleinVaeError> {
        let running_mean = tensors.tensor("bn.running_mean")?;
        let running_variance = tensors.tensor("bn.running_var")?;
        validate_shape(
            "bn",
            "running_mean",
            &running_mean,
            &[FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT],
        )?;
        validate_shape(
            "bn",
            "running_var",
            &running_variance,
            &[FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT],
        )?;
        let middle_prefix = "decoder.mid_block";
        Ok(Self {
            running_mean,
            running_variance,
            post_quant_conv: Flux2KleinChannelLastConv2d::load(
                runtime,
                tensors,
                "post_quant_conv",
                FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
                FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
                1,
                0,
            )?,
            conv_in: Flux2KleinChannelLastConv2d::load(
                runtime,
                tensors,
                "decoder.conv_in",
                FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
                DECODER_CHANNELS[0],
                3,
                1,
            )?,
            middle_resnet_before_attention: Flux2KleinVaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{middle_prefix}.resnets.0"),
                DECODER_CHANNELS[0],
                DECODER_CHANNELS[0],
            )?,
            middle_attention: Flux2KleinVaeMiddleAttention::load(
                runtime,
                tensors,
                &format!("{middle_prefix}.attentions.0"),
                DECODER_CHANNELS[0],
            )?,
            middle_resnet_after_attention: Flux2KleinVaeResnetBlock::load(
                runtime,
                tensors,
                &format!("{middle_prefix}.resnets.1"),
                DECODER_CHANNELS[0],
                DECODER_CHANNELS[0],
            )?,
            up_blocks: [
                Flux2KleinVaeUpBlock::load(runtime, tensors, 0, 512, 512, true)?,
                Flux2KleinVaeUpBlock::load(runtime, tensors, 1, 512, 512, true)?,
                Flux2KleinVaeUpBlock::load(runtime, tensors, 2, 512, 256, true)?,
                Flux2KleinVaeUpBlock::load(runtime, tensors, 3, 256, 128, false)?,
            ],
            output_norm: Flux2KleinGroupNorm::load(
                runtime,
                tensors,
                "decoder.conv_norm_out",
                DECODER_CHANNELS[3],
            )?,
            conv_out: Flux2KleinChannelLastConv2d::load(
                runtime,
                tensors,
                "decoder.conv_out",
                DECODER_CHANNELS[3],
                OUTPUT_CHANNELS,
                3,
                1,
            )?,
        })
    }

    pub fn decode_packed_latents(
        &self,
        runtime: &MlxRuntime,
        packed_latents: &MlxArray,
        layout: Flux2KleinPackedLatentLayout,
        mode: Flux2KleinVaeDecodeMode,
    ) -> Result<MlxArray, Flux2KleinVaeError> {
        let mut attribution = PerformanceAttribution::disabled();
        self.decode_packed_latents_with_performance_attribution(
            runtime,
            packed_latents,
            layout,
            mode,
            &mut attribution,
        )
    }

    pub fn decode_packed_latents_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        packed_latents: &MlxArray,
        layout: Flux2KleinPackedLatentLayout,
        mode: Flux2KleinVaeDecodeMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Flux2KleinVaeError> {
        let mut state = self.start_decode_packed_latents_with_performance_attribution(
            runtime,
            packed_latents,
            layout,
            mode,
            performance_attribution,
        )?;
        loop {
            match self.advance_decode_with_performance_attribution(
                runtime,
                state,
                performance_attribution,
            )? {
                Flux2KleinVaeDecodeAdvance::Decoding(next_state) => state = next_state,
                Flux2KleinVaeDecodeAdvance::PixelsReady(decoded_pixels) => {
                    return Ok(decoded_pixels);
                }
            }
        }
    }
}

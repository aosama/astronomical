//! Complete native MLX Qwen-Image-2.1 VAE decoder.
//!
//! Port of `AutoencoderKLQwenImage21` + `QwenImage21Decoder3d` for the artifact configuration
//! (`is_residual: true`, no patchify, tiling off). Weight provenance is the reviewed
//! `mlx-community/Qwen-Image-2.1-MLX-4bit` VAE `model.safetensors`: 238 unquantized F32 tensors
//! already stored in MLX conv order.

use std::fs::File;

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensors};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::QwenImage21VaeError;
use super::convolution::QwenImage21VaeSpatialConv;
use super::decode_stages::{DecodeAdvance, DecodeStage, DecodeState};
use super::latent_geometry::{
    QWEN_IMAGE_21_LATENT_CHANNEL_COUNT, QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT,
};
use super::mid_block::QwenImage21VaeMidBlock;
use super::rms_norm::QwenImage21VaeRmsNorm;
use super::tensor_shape::as_i32;
use super::up_block::QwenImage21VaeResidualUpBlock;

/// Latent channels entering the decoder (`z_dim`, after `post_quant_conv`). The transformer emits
/// the same width, so both sides read the one geometry constant rather than restating 64.
const LATENT_CHANNEL_COUNT: usize = QWEN_IMAGE_21_LATENT_CHANNEL_COUNT;
/// Decoded pixel channels — the reference converts condition images to RGBA, so the VAE
/// reconstructs four channels, not three.
const OUTPUT_CHANNEL_COUNT: usize = QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT;
/// Decoder width after `conv_in` and through the middle block (`decoder_base_dim * dim_mult[-1]`).
const MIDDLE_CHANNEL_COUNT: usize = 1152;
/// Width entering `conv_out` after the last up block.
const OUTPUT_HEAD_CHANNEL_COUNT: usize = 144;
/// `(input, output)` channel pairs per residual up block, derived from the weight manifest.
const UP_BLOCK_CHANNELS: [(usize, usize); 5] = [
    (1152, 1152),
    (1152, 1152),
    (1152, 576),
    (576, 288),
    (288, 144),
];
/// Whether each up block temporally upsamples (`upsample3d`); the final block never upsamples.
const UP_BLOCK_TEMPORAL_UPSAMPLE: [bool; 5] = [true, true, true, false, false];

/// Per-channel latent normalization constants from the reviewed artifact's `vae/config.json`;
/// the pipeline denormalizes samples with `latents * latents_std + latents_mean` before decode.
const LATENTS_MEAN: [f32; LATENT_CHANNEL_COUNT] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651,
    0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391,
    -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342,
    -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368,
    0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162,
    1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
const LATENTS_STD: [f32; LATENT_CHANNEL_COUNT] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904,
    2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075,
    3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149,
    3.7662, 3.6746, 3.4563, 3.8161,
];

#[derive(Debug)]
pub struct QwenImage21VaeDecoder {
    latents_mean: MlxArray,
    latents_std: MlxArray,
    post_quant_conv: QwenImage21VaeSpatialConv,
    convolution_input: QwenImage21VaeSpatialConv,
    mid_block: QwenImage21VaeMidBlock,
    up_blocks: [QwenImage21VaeResidualUpBlock; 5],
    output_norm: QwenImage21VaeRmsNorm,
    output_convolution: QwenImage21VaeSpatialConv,
}

impl QwenImage21VaeDecoder {
    pub fn load(runtime: &MlxRuntime, vae_weights_file: File) -> Result<Self, QwenImage21VaeError> {
        let mut attribution = PerformanceAttribution::disabled();
        Self::load_with_performance_attribution(runtime, vae_weights_file, &mut attribution)
    }

    pub fn load_with_performance_attribution(
        runtime: &MlxRuntime,
        vae_weights_file: File,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, QwenImage21VaeError> {
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

    fn bind(runtime: &MlxRuntime, tensors: &MlxSafetensors) -> Result<Self, QwenImage21VaeError> {
        let latents_mean = runtime.array_from_f32(
            &LATENTS_MEAN,
            &[as_i32(LATENT_CHANNEL_COUNT, "mean width")?],
        )?;
        let latents_std =
            runtime.array_from_f32(&LATENTS_STD, &[as_i32(LATENT_CHANNEL_COUNT, "std width")?])?;
        let up_blocks = UP_BLOCK_CHANNELS
            .iter()
            .enumerate()
            .map(|(block_index, &(input_channels, output_channels))| {
                // The final up block has no upsampler; the reference's last block only reshapes
                // channels ahead of `conv_out`.
                let temporal_upsample = UP_BLOCK_TEMPORAL_UPSAMPLE[block_index];
                let has_upsampler = block_index + 1 < UP_BLOCK_CHANNELS.len();
                QwenImage21VaeResidualUpBlock::load(
                    runtime,
                    tensors,
                    block_index,
                    input_channels,
                    output_channels,
                    temporal_upsample,
                    has_upsampler,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            latents_mean,
            latents_std,
            post_quant_conv: QwenImage21VaeSpatialConv::load(
                tensors,
                "post_quant_conv",
                LATENT_CHANNEL_COUNT,
                LATENT_CHANNEL_COUNT,
                1,
                0,
            )?,
            convolution_input: QwenImage21VaeSpatialConv::load(
                tensors,
                "decoder.conv_in",
                LATENT_CHANNEL_COUNT,
                MIDDLE_CHANNEL_COUNT,
                3,
                1,
            )?,
            mid_block: QwenImage21VaeMidBlock::load(
                runtime,
                tensors,
                "decoder.mid_block",
                MIDDLE_CHANNEL_COUNT,
            )?,
            up_blocks: up_blocks.try_into().map_err(|_| {
                QwenImage21VaeError::invalid_geometry("the decoder needs exactly five up blocks")
            })?,
            output_norm: QwenImage21VaeRmsNorm::load(
                runtime,
                tensors,
                "decoder.norm_out",
                OUTPUT_HEAD_CHANNEL_COUNT,
            )?,
            output_convolution: QwenImage21VaeSpatialConv::load(
                tensors,
                "decoder.conv_out",
                OUTPUT_HEAD_CHANNEL_COUNT,
                OUTPUT_CHANNEL_COUNT,
                3,
                1,
            )?,
        })
    }

    /// Reverses the pipeline's per-channel latent normalization
    /// (`latents = latents * latents_std + latents_mean`).
    ///
    /// The reference denormalizes inside the pipeline just before `vae.decode`; the constants are
    /// the decoder's own, so the step happens here and callers hand `decode_image_latents` the
    /// latents straight from the denoise loop.
    fn denormalize_latents(
        &self,
        runtime: &MlxRuntime,
        normalized_latents: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        validate_latent_shape(normalized_latents)?;
        let scaled = runtime.multiply(normalized_latents, &self.latents_std)?;
        Ok(runtime.add(&scaled, &self.latents_mean)?)
    }

    /// Decodes normalized latents `[B, H, W, 64]` into clamped RGBA pixels
    /// `[B, H * 16, W * 16, 4]`, without performance attribution.
    pub fn decode_image_latents(
        &self,
        runtime: &MlxRuntime,
        normalized_latents: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        let mut attribution = PerformanceAttribution::disabled();
        self.decode_image_latents_with_performance_attribution(
            runtime,
            normalized_latents,
            &mut attribution,
        )
    }

    /// Decodes normalized latents `[B, H, W, 64]` into clamped RGBA pixels
    /// `[B, H * 16, W * 16, 4]`, reporting each stage's graph construction and synchronization.
    ///
    /// The graph is built and evaluated one stage at a time so intermediate tensors are released
    /// between stages instead of accumulating in wired memory: at 1024×1024 a phase-granular
    /// decode builds the largest up-block graph whole and the memory ceiling rejects the
    /// allocation, even with every other component already released.
    pub fn decode_image_latents_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        normalized_latents: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        let denormalized = performance_attribution.measure_operation(
            PerformanceOperation::ImageVaeCompleteDecodeGraphConstruction,
            |_| self.denormalize_latents(runtime, normalized_latents),
        )?;
        let post_quant = self.post_quant_conv.forward(runtime, &denormalized)?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageVaeDecodeSynchronizationWait,
            |_| post_quant.evaluate(),
        )?;
        let mut state = DecodeState {
            hidden_states: post_quant,
            next_stage: DecodeStage::ConvolutionInput,
        };
        loop {
            match self.advance_decode_with_performance_attribution(
                runtime,
                state,
                performance_attribution,
            )? {
                DecodeAdvance::Decoding(next_state) => state = next_state,
                DecodeAdvance::PixelsReady(decoded_pixels) => return Ok(decoded_pixels),
            }
        }
    }

    /// Builds and evaluates exactly one stage, then reports the stage that follows it.
    fn advance_decode_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        state: DecodeState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<DecodeAdvance, QwenImage21VaeError> {
        let DecodeState {
            hidden_states,
            next_stage,
        } = state;
        let advanced = performance_attribution.measure_operation(
            PerformanceOperation::ImageVaeCompleteDecodeGraphConstruction,
            |_| match next_stage {
                DecodeStage::ConvolutionInput => {
                    self.convolution_input.forward(runtime, &hidden_states)
                }
                DecodeStage::MiddleResnetBeforeAttention => self
                    .mid_block
                    .forward_resnet_before_attention(runtime, &hidden_states),
                DecodeStage::MiddleAttention => {
                    self.mid_block.forward_attention(runtime, &hidden_states)
                }
                DecodeStage::MiddleResnetAfterAttention => self
                    .mid_block
                    .forward_resnet_after_attention(runtime, &hidden_states),
                DecodeStage::UpBlock(block_index) => {
                    self.up_blocks[block_index].forward(runtime, &hidden_states)
                }
                DecodeStage::OutputHead => self.forward_output_head(runtime, &hidden_states),
            },
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageVaeDecodeSynchronizationWait,
            |_| advanced.evaluate(),
        )?;
        match next_stage.next(self.up_blocks.len()) {
            Some(next_stage) => Ok(DecodeAdvance::Decoding(DecodeState {
                hidden_states: advanced,
                next_stage,
            })),
            None => Ok(DecodeAdvance::PixelsReady(clamp_decoded_pixels(
                runtime, &advanced,
            )?)),
        }
    }

    /// `norm_out → SiLU → conv_out`; the caller clamps to the reference's `[-1, 1]` output range.
    fn forward_output_head(
        &self,
        runtime: &MlxRuntime,
        hidden_states: &MlxArray,
    ) -> Result<MlxArray, QwenImage21VaeError> {
        let normalized = self.output_norm.forward(runtime, hidden_states)?;
        let activated = runtime.silu(&normalized)?;
        self.output_convolution.forward(runtime, &activated)
    }
}

/// Clamps decoded pixels to the reference `_decode` output range.
pub(super) fn clamp_decoded_pixels(
    runtime: &MlxRuntime,
    decoded_pixels: &MlxArray,
) -> Result<MlxArray, QwenImage21VaeError> {
    Ok(runtime.clip(decoded_pixels, -1.0, 1.0)?)
}

fn validate_latent_shape(latents: &MlxArray) -> Result<(), QwenImage21VaeError> {
    let shape = latents.shape();
    if shape.len() == 4 && shape[3] == as_i32(LATENT_CHANNEL_COUNT, "latent channels")? {
        return Ok(());
    }
    Err(QwenImage21VaeError::invalid_geometry(format!(
        "decoding expects NHWC latents with {LATENT_CHANNEL_COUNT} channels, received {shape:?}"
    )))
}

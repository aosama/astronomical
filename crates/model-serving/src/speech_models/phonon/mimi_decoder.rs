//! The Mimi codec's decoder: 12.5 Hz latents to 24 kHz audio.
//!
//! A latent is projected to 512 channels, upsampled 16x by a depthwise
//! transposed convolution to 200 Hz, passed through a two-layer transformer
//! that sees the last 250 positions, and turned into samples by a SEANet
//! decoder that upsamples another 6 x 5 x 4 = 120x. Every layer is causal, so
//! decoding a whole utterance at once gives the same samples a stream of
//! frames would.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::causal_convolutions::{CausalConv1d, CausalConvTranspose1d, CausalGeometry};
use super::causal_transformer::{CausalTransformer, TransformerShape};
use super::configuration::MimiDecoderShape;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::activations::elu;
use crate::speech_models::speech_model_error::SpeechModelError;

/// Latents are at 12.5 Hz and the decoder transformer at 200 Hz.
const LATENT_UPSAMPLING: i32 = 16;
const MODEL_PREFIX: &str = "mimi.decoder.model";

/// One SEANet stage: upsample, then a residual block.
struct UpsamplingStage {
    upsampler: CausalConvTranspose1d,
    residual_narrowing: CausalConv1d,
    residual_widening: CausalConv1d,
}

pub(crate) struct MimiDecoder {
    latent_projection: CausalConv1d,
    frame_upsampler: CausalConvTranspose1d,
    transformer: CausalTransformer,
    /// Positions the transformer reads per call. A whole utterance at once
    /// would need attention scores for every pair of positions (about 450 MB
    /// a layer for a full 50-token chunk); blocks of one window keep it
    /// under 8 MB, and the attention history makes the result identical.
    transformer_block_length: usize,
    initial_convolution: CausalConv1d,
    stages: Vec<UpsamplingStage>,
    final_convolution: CausalConv1d,
}

impl MimiDecoder {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        shape: &MimiDecoderShape,
    ) -> Result<Self, SpeechModelError> {
        let model_width = checkpoint
            .tensor("mimi.quantizer.output_proj.weight")?
            .shape()[0];
        // SEANet numbers its modules: 0 is the first convolution, then each
        // stage takes three slots (activation, upsampler, residual block).
        let final_slot = 2 + 3 * shape.upsampling_ratios.len();
        Ok(Self {
            latent_projection: CausalConv1d::load(
                checkpoint,
                "mimi.quantizer.output_proj",
                CausalGeometry::PLAIN,
            )?,
            frame_upsampler: CausalConvTranspose1d::load(
                checkpoint,
                "mimi.upsample.convtr.convtr",
                LATENT_UPSAMPLING,
                model_width,
            )?,
            transformer: load_transformer(checkpoint, shape)?,
            transformer_block_length: shape.attention_context,
            initial_convolution: CausalConv1d::load(
                checkpoint,
                &format!("{MODEL_PREFIX}.0.conv"),
                CausalGeometry::PLAIN,
            )?,
            stages: load_stages(checkpoint, &shape.upsampling_ratios)?,
            final_convolution: CausalConv1d::load(
                checkpoint,
                &format!("{MODEL_PREFIX}.{final_slot}.conv"),
                CausalGeometry::PLAIN,
            )?,
        })
    }

    /// `[frames, latent width]` denormalised latents to `[frames * 1920]` samples.
    pub(crate) fn decode(
        &self,
        runtime: &MlxRuntime,
        latents: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let projected = self.latent_projection.forward(runtime, latents)?;
        let upsampled = self.frame_upsampler.forward(runtime, &projected)?;
        let contextualised = self.contextualise_in_blocks(runtime, &upsampled)?;
        let mut features = self.initial_convolution.forward(runtime, &contextualised)?;
        for stage in &self.stages {
            features = stage.forward(runtime, &features)?;
        }
        let samples = self
            .final_convolution
            .forward(runtime, &elu(runtime, &features)?)?;
        let sample_count = samples.shape()[0];
        Ok(runtime.reshape(&samples, &[sample_count])?)
    }

    /// The transformer over `[positions, width]`, one block at a time.
    pub(crate) fn contextualise_in_blocks(
        &self,
        runtime: &MlxRuntime,
        positions: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let [position_count, width] = positions.shape()[..] else {
            return Err(SpeechModelError::Input(
                "Mimi expects [positions, width]".to_owned(),
            ));
        };
        let mut history = self.transformer.empty_history();
        let mut contextualised_blocks = Vec::new();
        let mut block_start = 0;
        while block_start < position_count {
            let block_end =
                (block_start + self.transformer_block_length as i32).min(position_count);
            let block =
                runtime.slice(positions, &[block_start, 0], &[block_end, width], &[1, 1])?;
            let contextualised_block = self.transformer.forward(runtime, &block, &mut history)?;
            runtime.evaluate_arrays(&[&contextualised_block])?;
            contextualised_blocks.push(contextualised_block);
            block_start = block_end;
        }
        let block_refs: Vec<&MlxArray> = contextualised_blocks.iter().collect();
        Ok(runtime.concatenate_axis(&block_refs, 0)?)
    }
}

impl UpsamplingStage {
    fn load(
        checkpoint: &CheckpointTensors<'_>,
        stage_index: usize,
        ratio: i32,
    ) -> Result<Self, SpeechModelError> {
        let upsampler_slot = 2 + 3 * stage_index;
        let residual_prefix = format!("{MODEL_PREFIX}.{}.block", upsampler_slot + 1);
        Ok(Self {
            upsampler: CausalConvTranspose1d::load(
                checkpoint,
                &format!("{MODEL_PREFIX}.{upsampler_slot}.convtr"),
                ratio,
                1,
            )?,
            residual_narrowing: CausalConv1d::load(
                checkpoint,
                &format!("{residual_prefix}.1.conv"),
                CausalGeometry::PLAIN,
            )?,
            residual_widening: CausalConv1d::load(
                checkpoint,
                &format!("{residual_prefix}.3.conv"),
                CausalGeometry::PLAIN,
            )?,
        })
    }

    /// ELU, upsample, then `x + widen(elu(narrow(elu(x))))`.
    fn forward(
        &self,
        runtime: &MlxRuntime,
        features: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let upsampled = self.upsampler.forward(runtime, &elu(runtime, features)?)?;
        let narrowed = self
            .residual_narrowing
            .forward(runtime, &elu(runtime, &upsampled)?)?;
        let residual = self
            .residual_widening
            .forward(runtime, &elu(runtime, &narrowed)?)?;
        Ok(runtime.add(&upsampled, &residual)?)
    }
}

/// Two layers with layer scale, each position seeing the last
/// `attention_context` positions (itself included).
fn load_transformer(
    checkpoint: &CheckpointTensors<'_>,
    shape: &MimiDecoderShape,
) -> Result<CausalTransformer, SpeechModelError> {
    let transformer_shape = TransformerShape {
        prefix: "mimi.decoder_transformer.transformer",
        layer_count: shape.layer_count,
        head_count: shape.head_count,
        rotary_max_period: shape.rotary_max_period,
        attention_window: Some(shape.attention_context),
        has_branch_scales: shape.has_layer_scale,
    };
    CausalTransformer::load(checkpoint, &transformer_shape)
}

fn load_stages(
    checkpoint: &CheckpointTensors<'_>,
    upsampling_ratios: &[i32],
) -> Result<Vec<UpsamplingStage>, SpeechModelError> {
    let mut stages = Vec::with_capacity(upsampling_ratios.len());
    for (stage_index, ratio) in upsampling_ratios.iter().enumerate() {
        stages.push(UpsamplingStage::load(checkpoint, stage_index, *ratio)?);
    }
    Ok(stages)
}

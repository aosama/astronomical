//! The flow head: turns one frame of noise into one Mimi latent, conditioned
//! on the backbone's output for that frame (`SimpleMLPAdaLN` in Pocket TTS).
//!
//! Phonon samples with Lagrangian self-distillation in a single step: the
//! latent is the noise plus one prediction of the flow from time 0 to 1.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::time_embedding::single_step_time_embedding;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::activations::modulate;
use crate::speech_models::mlx_layers::layer_norm::LayerNorm;
use crate::speech_models::mlx_layers::linear::Linear;
use crate::speech_models::speech_model_error::SpeechModelError;

const RESIDUAL_NORM_EPSILON: f32 = 1e-6;
const PREFIX: &str = "flow_lm.flow_net";

struct ResidualBlock {
    input_norm: LayerNorm,
    first_projection: Linear,
    second_projection: Linear,
    /// SiLU(conditioning) to shift, scale and gate, `3 * width` wide.
    modulation: Linear,
}

pub(crate) struct FlowHead {
    condition_projection: Linear,
    noise_projection: Linear,
    residual_blocks: Vec<ResidualBlock>,
    final_norm: LayerNorm,
    final_modulation: Linear,
    final_projection: Linear,
    /// The single step's time conditioning, `[1, width]`.
    step_time_embedding: MlxArray,
    width: i32,
}

impl FlowHead {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        depth: usize,
    ) -> Result<Self, SpeechModelError> {
        let mut residual_blocks = Vec::with_capacity(depth);
        for block_index in 0..depth {
            residual_blocks.push(ResidualBlock::load(
                checkpoint,
                &format!("{PREFIX}.res_blocks.{block_index}"),
            )?);
        }
        let width = checkpoint
            .tensor(&format!("{PREFIX}.cond_embed.bias"))?
            .shape()[0];
        Ok(Self {
            condition_projection: Linear::load(checkpoint, &format!("{PREFIX}.cond_embed"))?,
            noise_projection: Linear::load(checkpoint, &format!("{PREFIX}.input_proj"))?,
            residual_blocks,
            final_norm: LayerNorm::plain(checkpoint.runtime, width, RESIDUAL_NORM_EPSILON)?,
            final_modulation: Linear::load(
                checkpoint,
                &format!("{PREFIX}.final_layer.adaLN_modulation.1"),
            )?,
            final_projection: Linear::load(checkpoint, &format!("{PREFIX}.final_layer.linear"))?,
            step_time_embedding: single_step_time_embedding(
                checkpoint,
                &format!("{PREFIX}.time_embed"),
            )?,
            width,
        })
    }

    /// `[1, model width]` backbone output and `[1, latent width]` noise to the
    /// frame's latent, `[1, latent width]`.
    pub(crate) fn sample(
        &self,
        runtime: &MlxRuntime,
        backbone_output: &MlxArray,
        noise: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let projected_condition = self
            .condition_projection
            .forward(runtime, backbone_output)?;
        let conditioning = runtime.add(&self.step_time_embedding, &projected_condition)?;
        let activated_conditioning = runtime.silu(&conditioning)?;
        let mut flow_states = self.noise_projection.forward(runtime, noise)?;
        for block in &self.residual_blocks {
            flow_states =
                self.residual_step(runtime, block, &flow_states, &activated_conditioning)?;
        }
        let final_shift_and_scale = self
            .final_modulation
            .forward(runtime, &activated_conditioning)?;
        let shift = self.modulation_chunk(runtime, &final_shift_and_scale, 0)?;
        let scale = self.modulation_chunk(runtime, &final_shift_and_scale, 1)?;
        let normalized = self.final_norm.forward(runtime, &flow_states)?;
        let flow = self
            .final_projection
            .forward(runtime, &modulate(runtime, &normalized, &shift, &scale)?)?;
        Ok(runtime.add(noise, &flow)?)
    }

    fn residual_step(
        &self,
        runtime: &MlxRuntime,
        block: &ResidualBlock,
        flow_states: &MlxArray,
        activated_conditioning: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let block_modulation = block.modulation.forward(runtime, activated_conditioning)?;
        let shift = self.modulation_chunk(runtime, &block_modulation, 0)?;
        let scale = self.modulation_chunk(runtime, &block_modulation, 1)?;
        let gate = self.modulation_chunk(runtime, &block_modulation, 2)?;
        let modulated = modulate(
            runtime,
            &block.input_norm.forward(runtime, flow_states)?,
            &shift,
            &scale,
        )?;
        let expanded = runtime.silu(&block.first_projection.forward(runtime, &modulated)?)?;
        let update = block.second_projection.forward(runtime, &expanded)?;
        Ok(runtime.add(flow_states, &runtime.multiply(&gate, &update)?)?)
    }

    /// The `chunk_index`-th `width`-wide chunk of a `[1, k * width]` modulation.
    fn modulation_chunk(
        &self,
        runtime: &MlxRuntime,
        modulation: &MlxArray,
        chunk_index: i32,
    ) -> Result<MlxArray, SpeechModelError> {
        let chunk_start = chunk_index * self.width;
        Ok(runtime.slice(
            modulation,
            &[0, chunk_start],
            &[1, chunk_start + self.width],
            &[1, 1],
        )?)
    }
}

impl ResidualBlock {
    fn load(
        checkpoint: &CheckpointTensors<'_>,
        block_prefix: &str,
    ) -> Result<Self, SpeechModelError> {
        Ok(Self {
            input_norm: LayerNorm::load(
                checkpoint,
                &format!("{block_prefix}.in_ln"),
                RESIDUAL_NORM_EPSILON,
            )?,
            first_projection: Linear::load(checkpoint, &format!("{block_prefix}.mlp.0"))?,
            second_projection: Linear::load(checkpoint, &format!("{block_prefix}.mlp.2"))?,
            modulation: Linear::load(checkpoint, &format!("{block_prefix}.adaLN_modulation.1"))?,
        })
    }
}

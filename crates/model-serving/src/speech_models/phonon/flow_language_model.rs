//! Phonon's flow language model: a causal transformer that reads the voice
//! prompt and the text tokens, then emits one 32-wide Mimi latent per 80 ms
//! frame, each conditioned on the frames before it, together with a logit
//! that says when speech has ended.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::attention_history::AttentionHistory;
use super::causal_transformer::{CausalTransformer, TransformerShape};
use super::configuration::PhononConfiguration;
use super::flow_head::FlowHead;
use super::frame_conditioning::{CONDITIONERS, frame_conditioning};
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::layer_norm::LayerNorm;
use crate::speech_models::mlx_layers::linear::Linear;
use crate::speech_models::speech_model_error::SpeechModelError;

const OUTPUT_NORM_EPSILON: f32 = 1e-5;

pub(crate) struct FlowLanguageModel {
    /// One row per text token id.
    text_embeddings: MlxArray,
    speaker_projection: Linear,
    /// What every generated frame's input has added, `[1, width]`.
    frame_conditioning: MlxArray,
    latent_input: Linear,
    backbone: CausalTransformer,
    output_norm: LayerNorm,
    end_of_speech: Linear,
    flow_head: FlowHead,
    /// The first frame's input, before any latent exists, `[1, latent width]`.
    pub(crate) start_latent: MlxArray,
    pub(crate) latent_mean: MlxArray,
    pub(crate) latent_deviation: MlxArray,
}

/// One generated frame.
pub(crate) struct GeneratedFrame {
    /// Normalised latent, `[1, latent width]`; the next frame's input.
    pub(crate) latent: MlxArray,
    pub(crate) end_of_speech_logit: f32,
}

impl FlowLanguageModel {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        configuration: &PhononConfiguration,
    ) -> Result<Self, SpeechModelError> {
        let flow = &configuration.flow;
        let backbone_shape = TransformerShape {
            prefix: "flow_lm.transformer",
            layer_count: flow.layer_count,
            head_count: flow.head_count,
            rotary_max_period: flow.rotary_max_period,
            attention_window: None,
            has_branch_scales: false,
        };
        let start_latent = checkpoint.tensor("flow_lm.bos_emb")?;
        Ok(Self {
            text_embeddings: checkpoint.tensor(&format!(
                "{CONDITIONERS}.transcript_in_segment.embed.weight"
            ))?,
            speaker_projection: Linear::load(
                checkpoint,
                &format!("{CONDITIONERS}.speaker_wavs.output_proj"),
            )?,
            frame_conditioning: frame_conditioning(
                checkpoint,
                &configuration.speaker_count_values,
                &configuration.duration_conditioner,
            )?,
            latent_input: Linear::load(checkpoint, "flow_lm.input_linear")?,
            backbone: CausalTransformer::load(checkpoint, &backbone_shape)?,
            output_norm: LayerNorm::load(checkpoint, "flow_lm.out_norm", OUTPUT_NORM_EPSILON)?,
            end_of_speech: Linear::load(checkpoint, "flow_lm.out_eos")?,
            flow_head: FlowHead::load(checkpoint, flow.flow_depth)?,
            start_latent: checkpoint
                .runtime
                .reshape(&start_latent, &[1, flow.latent_width])?,
            latent_mean: checkpoint.tensor("flow_lm.emb_mean")?,
            latent_deviation: checkpoint.tensor("flow_lm.emb_std")?,
        })
    }

    pub(crate) fn empty_history(&self) -> AttentionHistory {
        self.backbone.empty_history()
    }

    /// Reads a voice: `[latent frames, 512]` speaker latents, projected.
    pub(crate) fn prompt_voice(
        &self,
        runtime: &MlxRuntime,
        history: &mut AttentionHistory,
        speaker_latents: &MlxArray,
    ) -> Result<(), SpeechModelError> {
        let voice_embedding = self.speaker_projection.forward(runtime, speaker_latents)?;
        self.read_prompt(runtime, history, &voice_embedding)
    }

    /// Reads the text to speak.
    pub(crate) fn prompt_text(
        &self,
        runtime: &MlxRuntime,
        history: &mut AttentionHistory,
        token_ids: &[i32],
    ) -> Result<(), SpeechModelError> {
        let token_array = runtime.array_from_i32(token_ids, &[token_ids.len() as i32])?;
        let text_embedding = runtime.take_axis(&self.text_embeddings, &token_array, 0)?;
        self.read_prompt(runtime, history, &text_embedding)
    }

    /// A prompt's outputs are not used; reading it only fills the history.
    fn read_prompt(
        &self,
        runtime: &MlxRuntime,
        history: &mut AttentionHistory,
        prompt_embedding: &MlxArray,
    ) -> Result<(), SpeechModelError> {
        let prompt_states = self.backbone.forward(runtime, prompt_embedding, history)?;
        runtime.evaluate_arrays(&[&prompt_states])?;
        Ok(())
    }

    /// The frame after `previous_latent` (or [`Self::start_latent`]), from
    /// `[1, latent width]` noise.
    pub(crate) fn next_frame(
        &self,
        runtime: &MlxRuntime,
        history: &mut AttentionHistory,
        previous_latent: &MlxArray,
        noise: &MlxArray,
    ) -> Result<GeneratedFrame, SpeechModelError> {
        let projected_latent = self.latent_input.forward(runtime, previous_latent)?;
        let frame_input = runtime.add(&projected_latent, &self.frame_conditioning)?;
        let backbone_states = self.backbone.forward(runtime, &frame_input, history)?;
        let backbone_output = self.output_norm.forward(runtime, &backbone_states)?;
        let end_of_speech_logit = self.end_of_speech.forward(runtime, &backbone_output)?;
        let latent = self.flow_head.sample(runtime, &backbone_output, noise)?;
        runtime.evaluate_arrays(&[&latent, &end_of_speech_logit])?;
        let end_of_speech_logit = runtime
            .array_to_vec_f32(&end_of_speech_logit)?
            .first()
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        Ok(GeneratedFrame {
            latent,
            end_of_speech_logit,
        })
    }
}

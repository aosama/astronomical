//! Kokoro's phoneme-level ALBERT (PL-BERT): phoneme ids in, 768-wide context
//! states out, from which the prosody predictor reads how long and how high
//! each sound should be.
//!
//! ALBERT shares one transformer layer across all twelve passes, so the
//! checkpoint holds a single layer that is loaded once and applied twelve
//! times. Every sequence here is one unpadded utterance, so no attention mask
//! is needed. Activation is ALBERT's default `gelu_new`, the tanh form.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::layer_norm::LayerNorm;
use crate::speech_models::mlx_layers::linear::Linear;
use crate::speech_models::speech_model_error::SpeechModelError;

/// HuggingFace ALBERT's `layer_norm_eps`.
const ALBERT_LAYER_NORM_EPSILON: f32 = 1e-12;
const SHARED_LAYER_PREFIX: &str = "bert.encoder.albert_layer_groups.0.albert_layers.0";

/// Token, position and segment embeddings, summed and normalised.
struct AlbertEmbeddings {
    word_embeddings: MlxArray,
    position_embeddings: MlxArray,
    /// Every utterance is one segment, so only the first segment row is used.
    first_segment_embedding: MlxArray,
    embedding_norm: LayerNorm,
    embedding_to_hidden: Linear,
}

/// The one transformer layer every pass reuses.
struct SharedAlbertLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    attention_norm: LayerNorm,
    feed_forward: Linear,
    feed_forward_output: Linear,
    layer_output_norm: LayerNorm,
}

pub(crate) struct AlbertEncoder {
    embeddings: AlbertEmbeddings,
    shared_layer: SharedAlbertLayer,
    attention_head_count: i32,
    pass_count: usize,
}

impl AlbertEncoder {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        attention_head_count: usize,
        pass_count: usize,
    ) -> Result<Self, SpeechModelError> {
        Ok(Self {
            embeddings: AlbertEmbeddings::load(checkpoint)?,
            shared_layer: SharedAlbertLayer::load(checkpoint)?,
            attention_head_count: attention_head_count as i32,
            pass_count,
        })
    }

    /// `[length]` token ids to `[length, 768]` context states.
    pub(crate) fn encode(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let mut hidden_states = self.embeddings.embed(runtime, token_ids)?;
        for _ in 0..self.pass_count {
            hidden_states = self.encode_one_pass(runtime, &hidden_states)?;
        }
        Ok(hidden_states)
    }

    fn encode_one_pass(
        &self,
        runtime: &MlxRuntime,
        hidden_states: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let layer = &self.shared_layer;
        let attended = self.self_attention(runtime, hidden_states)?;
        let attention_residual = runtime.add(
            hidden_states,
            &layer.attention_output.forward(runtime, &attended)?,
        )?;
        let attention_states = layer.attention_norm.forward(runtime, &attention_residual)?;
        let expanded =
            runtime.gelu_tanh(&layer.feed_forward.forward(runtime, &attention_states)?)?;
        let feed_forward_states = layer.feed_forward_output.forward(runtime, &expanded)?;
        layer.layer_output_norm.forward(
            runtime,
            &runtime.add(&feed_forward_states, &attention_states)?,
        )
    }

    fn self_attention(
        &self,
        runtime: &MlxRuntime,
        hidden_states: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let [sequence_length, hidden_width] = hidden_states.shape()[..] else {
            return Err(SpeechModelError::Input(
                "ALBERT expects [length, width] states".to_owned(),
            ));
        };
        let layer = &self.shared_layer;
        let head_width = hidden_width / self.attention_head_count;
        let queries =
            self.split_into_heads(runtime, &layer.query.forward(runtime, hidden_states)?)?;
        let keys = self.split_into_heads(runtime, &layer.key.forward(runtime, hidden_states)?)?;
        let values =
            self.split_into_heads(runtime, &layer.value.forward(runtime, hidden_states)?)?;
        let scale = 1.0 / (head_width as f32).sqrt();
        let attended = runtime.scaled_dot_product_attention(&queries, &keys, &values, scale)?;
        let heads_last = runtime.transpose_axes(&attended, &[0, 2, 1, 3])?;
        Ok(runtime.reshape(&heads_last, &[sequence_length, hidden_width])?)
    }

    /// `[length, width]` to `[1, heads, length, head width]`.
    fn split_into_heads(
        &self,
        runtime: &MlxRuntime,
        projected: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let [sequence_length, hidden_width] = projected.shape()[..] else {
            return Err(SpeechModelError::Input(
                "ALBERT expects [length, width] states".to_owned(),
            ));
        };
        let head_count = self.attention_head_count;
        let per_head = runtime.reshape(
            projected,
            &[1, sequence_length, head_count, hidden_width / head_count],
        )?;
        Ok(runtime.transpose_axes(&per_head, &[0, 2, 1, 3])?)
    }
}

impl AlbertEmbeddings {
    fn load(checkpoint: &CheckpointTensors<'_>) -> Result<Self, SpeechModelError> {
        let segment_embeddings =
            checkpoint.tensor("bert.embeddings.token_type_embeddings.weight")?;
        let embedding_width = segment_embeddings.shape()[1];
        Ok(Self {
            word_embeddings: checkpoint.tensor("bert.embeddings.word_embeddings.weight")?,
            position_embeddings: checkpoint.tensor("bert.embeddings.position_embeddings.weight")?,
            first_segment_embedding: checkpoint.runtime.slice(
                &segment_embeddings,
                &[0, 0],
                &[1, embedding_width],
                &[1, 1],
            )?,
            embedding_norm: LayerNorm::load(
                checkpoint,
                "bert.embeddings.LayerNorm",
                ALBERT_LAYER_NORM_EPSILON,
            )?,
            embedding_to_hidden: Linear::load(
                checkpoint,
                "bert.encoder.embedding_hidden_mapping_in",
            )?,
        })
    }

    fn embed(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let sequence_length = token_ids.shape()[0];
        let positions: Vec<i32> = (0..sequence_length).collect();
        let position_ids = runtime.array_from_i32(&positions, &[sequence_length])?;
        let word_vectors = runtime.take_axis(&self.word_embeddings, token_ids, 0)?;
        let position_vectors = runtime.take_axis(&self.position_embeddings, &position_ids, 0)?;
        let summed_embeddings = runtime.add(
            &runtime.add(&word_vectors, &position_vectors)?,
            &self.first_segment_embedding,
        )?;
        let normalized = self.embedding_norm.forward(runtime, &summed_embeddings)?;
        self.embedding_to_hidden.forward(runtime, &normalized)
    }
}

/// The full name of a tensor of the shared layer.
fn shared_layer_tensor(suffix: &str) -> String {
    format!("{SHARED_LAYER_PREFIX}.{suffix}")
}

impl SharedAlbertLayer {
    fn load(checkpoint: &CheckpointTensors<'_>) -> Result<Self, SpeechModelError> {
        Ok(Self {
            query: Linear::load(checkpoint, &shared_layer_tensor("attention.query"))?,
            key: Linear::load(checkpoint, &shared_layer_tensor("attention.key"))?,
            value: Linear::load(checkpoint, &shared_layer_tensor("attention.value"))?,
            attention_output: Linear::load(checkpoint, &shared_layer_tensor("attention.dense"))?,
            attention_norm: LayerNorm::load(
                checkpoint,
                &shared_layer_tensor("attention.LayerNorm"),
                ALBERT_LAYER_NORM_EPSILON,
            )?,
            feed_forward: Linear::load(checkpoint, &shared_layer_tensor("ffn"))?,
            feed_forward_output: Linear::load(checkpoint, &shared_layer_tensor("ffn_output"))?,
            layer_output_norm: LayerNorm::load(
                checkpoint,
                &shared_layer_tensor("full_layer_layer_norm"),
                ALBERT_LAYER_NORM_EPSILON,
            )?,
        })
    }
}

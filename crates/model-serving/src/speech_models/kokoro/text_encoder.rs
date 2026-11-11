//! Kokoro's acoustic text encoder: phoneme ids to the 512-wide states the
//! decoder turns into sound, through an embedding, three convolution stages
//! and a bidirectional LSTM.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::bidirectional_lstm::BidirectionalLstm;
use super::convolutions::{ConvolutionGeometry, KokoroConv1d};
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::activations::leaky_relu;
use crate::speech_models::mlx_layers::layer_norm::LayerNorm;
use crate::speech_models::speech_model_error::SpeechModelError;

const CONVOLUTION_STAGE_COUNT: usize = 3;
const CONVOLUTION_KERNEL_SIZE: i32 = 5;
const STAGE_NORM_EPSILON: f32 = 1e-5;
const STAGE_ACTIVATION_SLOPE: f32 = 0.2;

/// One convolution, its channel norm, then a leaky ReLU.
struct ConvolutionStage {
    convolution: KokoroConv1d,
    norm: LayerNorm,
}

pub(crate) struct AcousticTextEncoder {
    phoneme_embeddings: MlxArray,
    convolution_stages: Vec<ConvolutionStage>,
    recurrence: BidirectionalLstm,
}

impl AcousticTextEncoder {
    pub(crate) fn load(checkpoint: &CheckpointTensors<'_>) -> Result<Self, SpeechModelError> {
        let mut convolution_stages = Vec::with_capacity(CONVOLUTION_STAGE_COUNT);
        for stage_index in 0..CONVOLUTION_STAGE_COUNT {
            convolution_stages.push(ConvolutionStage {
                convolution: KokoroConv1d::load(
                    checkpoint,
                    &format!("text_encoder.cnn.{stage_index}.0"),
                    ConvolutionGeometry::same_length(CONVOLUTION_KERNEL_SIZE, 1),
                )?,
                norm: LayerNorm::load_gamma_beta(
                    checkpoint,
                    &format!("text_encoder.cnn.{stage_index}.1"),
                    STAGE_NORM_EPSILON,
                )?,
            });
        }
        Ok(Self {
            phoneme_embeddings: checkpoint.tensor("text_encoder.embedding.weight")?,
            convolution_stages,
            recurrence: BidirectionalLstm::load(checkpoint, "text_encoder.lstm")?,
        })
    }

    /// `[length]` token ids to `[length, 512]`.
    pub(crate) fn encode(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let mut encoded = runtime.take_axis(&self.phoneme_embeddings, token_ids, 0)?;
        for stage in &self.convolution_stages {
            let convolved = stage.convolution.forward(runtime, &encoded)?;
            let normalized = stage.norm.forward(runtime, &convolved)?;
            encoded = leaky_relu(runtime, &normalized, STAGE_ACTIVATION_SLOPE)?;
        }
        self.recurrence.forward(runtime, &encoded)
    }
}

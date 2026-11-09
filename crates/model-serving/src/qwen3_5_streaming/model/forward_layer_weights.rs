//! Where one forward finds each decoder layer's weights (issue #1985).
//!
//! A sparse model binds every decoder layer at load. A dense model the
//! streaming engine serves gets each layer from its streamer: a kept layer,
//! or one read from the SSD for this forward alone.

use super::Qwen3_5Model;
use super::dense_layer_streamer::DenseLayerWeights;
use super::layer_submission::{LayerGroupSubmission, LayerSubmission};
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

/// Where a forward finds one decoder layer.
pub(crate) enum ForwardLayerWeights<'model> {
    Bound(&'model Qwen3_5DecoderLayerWeights),
    Dense(DenseLayerWeights),
}

impl ForwardLayerWeights<'_> {
    pub(crate) fn decoder_layer_weights(&self) -> &Qwen3_5DecoderLayerWeights {
        match self {
            Self::Bound(bound_layer_weights) => bound_layer_weights,
            Self::Dense(dense_layer_weights) => dense_layer_weights.decoder_layer_weights(),
        }
    }

    /// Whether this layer was read for this forward alone, so its bytes
    /// free once it completes.
    pub(crate) const fn is_streamed(&self) -> bool {
        matches!(self, Self::Dense(dense_layer_weights) if dense_layer_weights.is_streamed())
    }

    /// How this layer reaches the GPU in a forward at `layer_group_submission`.
    pub(crate) const fn submission(
        &self,
        layer_group_submission: LayerGroupSubmission,
    ) -> LayerSubmission {
        LayerSubmission::for_layer(
            matches!(self, Self::Dense(_)),
            self.is_streamed(),
            layer_group_submission.forward_token_count,
        )
    }
}

impl Qwen3_5Model {
    /// The weights a forward uses for one decoder layer: the bound weights,
    /// or for a dense model the streamer's kept or freshly read layer.
    pub(crate) fn forward_layer_weights(
        &self,
        layer_index: usize,
    ) -> Result<ForwardLayerWeights<'_>, Qwen3_5ExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return self
                .weights
                .decoder_layer_weights
                .get(layer_index)
                .map(ForwardLayerWeights::Bound)
                .ok_or(Qwen3_5ExecutionError::MissingDecoderLayerWeights { layer_index });
        };
        Ok(ForwardLayerWeights::Dense(
            dense_layer_streamer.layer_weights(
                &self.runtime,
                &self.config,
                self.weights.raw_hugging_face_norm_offsets_apply(),
                layer_index,
            )?,
        ))
    }
}

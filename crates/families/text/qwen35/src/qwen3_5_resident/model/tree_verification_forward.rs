//! The resident model as a draft tree's target: every layer's weights are in
//! memory, so a layer's feed-forward over the tree's rows is the resident
//! feed-forward. The pass itself is the shared core's
//! (`draft_tree_verification`), the same one the streaming model runs.

use mlx_bridge::MlxArray;

use super::Qwen3_5ResidentModel;
use super::verification_forward::VerificationLogitsMode;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model::draft_tree_verification::{
    DraftTreeTarget, TreeLayerFeedForward, TreeLayerWeights,
};
use text_generation::execution_error::TextExecutionError;

pub use crate::qwen3_5_core::model::draft_tree_verification::Qwen3_5DraftTreeVerification;

impl DraftTreeTarget for Qwen3_5ResidentModel {
    fn draft_tree_model_base(&self) -> &Qwen3_5ModelBase {
        self
    }

    /// Every resident layer is bound in memory.
    fn tree_layer_weights(
        &self,
        layer_index: usize,
    ) -> Result<Box<dyn TreeLayerWeights + '_>, TextExecutionError> {
        let decoder_layer_weights = self
            .weights
            .decoder_layer_weights
            .get(layer_index)
            .ok_or(TextExecutionError::MissingDecoderLayerWeights { layer_index })?;
        Ok(Box::new(decoder_layer_weights))
    }

    fn tree_layer_feed_forward(
        &self,
        feed_forward: TreeLayerFeedForward<'_>,
    ) -> Result<MlxArray, TextExecutionError> {
        self.forward_decoder_layer_feed_forward(
            feed_forward.attention_output,
            feed_forward.row_count,
            feed_forward.layer_index,
            feed_forward.decoder_layer_weights,
            &mut PerformanceAttribution::disabled(),
        )
    }

    fn tree_per_row_logits(
        &self,
        hidden_states: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        self.verification_per_position_logits(
            hidden_states,
            VerificationLogitsMode::AllPositions,
            hidden_states.shape()[1],
        )?
        .ok_or(TextExecutionError::InvalidInput {
            description: "a draft tree must produce logits",
        })
    }

    /// Nothing in a resident pass waits for the host, so the GPU starts on
    /// each few layers while the next are built.
    fn starts_tree_layers_asynchronously(&self) -> bool {
        true
    }
}

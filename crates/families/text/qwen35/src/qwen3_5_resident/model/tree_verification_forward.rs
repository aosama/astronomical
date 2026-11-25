//! The resident model as a draft tree's target: every layer's weights are in
//! memory, so a layer's feed-forward over the tree's rows is the resident
//! feed-forward. The pass itself is the shared core's
//! (`draft_tree_verification`), the same one the streaming model runs.

use mlx_bridge::MlxArray;

use super::Qwen3_5ResidentModel;
use super::verification_forward::VerificationLogitsMode;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model::draft_tree_verification::{DraftTreeTarget, request_layer_state};
use crate::qwen3_5_core::model::tree_layer_attention::{TreeLayerRows, TreeRowInputs};
use crate::{PerformanceAttribution, RequestDecoderStateStack};
use text_generation::execution_error::TextExecutionError;

pub use crate::qwen3_5_core::model::draft_tree_verification::Qwen3_5DraftTreeVerification;

impl DraftTreeTarget for Qwen3_5ResidentModel {
    fn draft_tree_model_base(&self) -> &Qwen3_5ModelBase {
        self
    }

    fn forward_tree_layer(
        &self,
        hidden_states: &MlxArray,
        tree: &TreeRowInputs,
        layer_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<(MlxArray, TreeLayerRows), TextExecutionError> {
        let decoder_layer_weights = self
            .weights
            .decoder_layer_weights
            .get(layer_index)
            .ok_or(TextExecutionError::MissingDecoderLayerWeights { layer_index })?;
        let layer_model_state = request_layer_state(request_decoder_state, layer_index)?;
        let (attention_output, rows) = self.forward_tree_layer_attention(
            hidden_states,
            tree,
            layer_index,
            decoder_layer_weights,
            layer_model_state,
        )?;
        let layer_output = self.forward_decoder_layer_feed_forward(
            &attention_output,
            tree.row_count,
            layer_index,
            decoder_layer_weights,
            &mut PerformanceAttribution::disabled(),
        )?;
        Ok((layer_output, rows))
    }

    fn tree_per_row_logits(
        &self,
        hidden_states: &MlxArray,
        row_count: i32,
    ) -> Result<MlxArray, TextExecutionError> {
        self.verification_per_position_logits(
            hidden_states,
            VerificationLogitsMode::AllPositions,
            row_count,
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

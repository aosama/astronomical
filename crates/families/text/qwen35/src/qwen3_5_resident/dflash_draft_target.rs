//! The resident Qwen3.5 model as a DFlash2 drafter's target: the drafter
//! borrows its embedding rows and output head, so it has no vocabulary of its
//! own and its logits are the target's.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::InferenceEngineError;
use crate::gpu_token_sampling::sampling_distribution_rows;
use engine_core::speculative_decoding::dflash_drafter::DFlashDraftTarget;

use super::model::Qwen3_5ResidentModel;
use crate::qwen3_5_core::model::draft_tree_verification::DraftTreeTarget;

/// The resident model lends what its base lends; kept so a caller holding
/// the resident model can hand it to a drafter directly.
impl DFlashDraftTarget for Qwen3_5ResidentModel {
    fn embedded_rows(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.draft_tree_model_base()
            .embedded_rows(runtime, token_ids)
    }

    fn head_logits(
        &self,
        runtime: &MlxRuntime,
        hidden_rows: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        self.draft_tree_model_base()
            .head_logits(runtime, hidden_rows)
    }
}

impl Qwen3_5ResidentModel {
    /// The distribution this engine's sampler draws each row of `logits`
    /// [1, rows, vocabulary] from, as each row's `top_k` token ids and their
    /// probabilities (every other token has probability zero), for
    /// speculative verification.
    pub fn sampling_distribution_rows(
        &self,
        logits: &MlxArray,
        temperature_thousandths: u16,
        top_p_thousandths: u16,
        top_k: u16,
    ) -> Result<(MlxArray, MlxArray), InferenceEngineError> {
        sampling_distribution_rows(
            &self.runtime,
            logits,
            temperature_thousandths,
            top_p_thousandths,
            top_k,
        )
        .map_err(InferenceEngineError::from)
    }
}

impl crate::qwen3_5_core::draft_tree_decode::draft_round_target::DraftRoundTarget
    for Qwen3_5ResidentModel
{
    /// A resident pass stays lazy: the round's first readback evaluates it.
    fn verify_round_tree(
        &self,
        round_tree: crate::qwen3_5_core::draft_tree_decode::draft_round_target::RoundTree<'_>,
        request_decoder_state: &mut crate::RequestDecoderStateStack,
    ) -> Result<
        crate::qwen3_5_core::model::draft_tree_verification::Qwen3_5DraftTreeVerification,
        text_generation::execution_error::TextExecutionError,
    > {
        self.verify_draft_tree(
            round_tree.tree_rows,
            round_tree.anchor_position,
            request_decoder_state,
            round_tree.tap_layer_ids,
        )
    }

    /// Converts the dense projections into register order and declares the
    /// tree kernels, the row-exact tree attention among them where the
    /// target's attention shape fits it.
    ///
    /// Conversion only speeds up verification; the tree kernels are what
    /// make it possible. So a target with nothing to convert still drafts:
    /// an 8-bit or mixed build, or a mixture-of-experts model whose experts
    /// stay in MLX's layout for the gathered expert kernels and whose
    /// attention may be 8-bit.
    fn prepare_for_draft_trees(
        &mut self,
        performance_attribution: &mut crate::PerformanceAttribution,
    ) -> Result<Option<usize>, text_generation::execution_error::TextExecutionError> {
        let Some(converted_projection_count) =
            self.convert_projections_to_register_order(performance_attribution)?
        else {
            tracing::info!(
                "a paired DFlash drafter is installed, but this GPU cannot run the \
                 register-ordered kernels drafting needs; decoding stays classic"
            );
            return Ok(None);
        };
        self.enable_draft_tree_verification()?;
        self.enable_row_exact_tree_attention()?;
        Ok(Some(converted_projection_count))
    }
}

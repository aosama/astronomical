use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::{MtpDraftDepth, Qwen3_5MtpRequestState, Qwen3_5MtpRequestStateAllocationCheckpoint};

/// Evaluated bounded proposal IDs produced by one lazy autoregressive MTP chain.
pub(crate) struct MtpProposalChain {
    draft_token_ids: Vec<u32>,
    predictor_base_checkpoint: Qwen3_5MtpRequestStateAllocationCheckpoint,
}

impl MtpProposalChain {
    pub(crate) fn into_parts(self) -> (Vec<u32>, Qwen3_5MtpRequestStateAllocationCheckpoint) {
        (self.draft_token_ids, self.predictor_base_checkpoint)
    }
}

impl Qwen3_5Model {
    /// Reuses the stored MTP layer without synchronizing between draft depths.
    pub(crate) fn propose_mtp_chain_with_performance_attribution(
        &self,
        target_hidden_seed: &MlxArray,
        current_token_indices: &MlxArray,
        effective_depth: MtpDraftDepth,
        mtp_request_state: &mut Qwen3_5MtpRequestState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MtpProposalChain, Qwen3_5ExecutionError> {
        let mut draft_token_arrays = Vec::with_capacity(usize::from(effective_depth.get()));
        let mut post_normalization_hidden_rows =
            Vec::with_capacity(usize::from(effective_depth.get()));
        let mut predictor_base_checkpoint = None;
        performance_attribution.measure_operation(
            PerformanceOperation::MtpHeadForwardGraphConstruction,
            |performance_attribution| {
                for draft_position in 0..effective_depth.get() {
                    let hidden_states_for_fusion = if draft_position == 0 {
                        target_hidden_seed
                    } else {
                        post_normalization_hidden_rows.last().ok_or(
                            Qwen3_5ExecutionError::InvalidInput {
                                description: "MTP proposal chain lost its prior hidden row",
                            },
                        )?
                    };
                    let token_indices_for_fusion = if draft_position == 0 {
                        current_token_indices
                    } else {
                        draft_token_arrays
                            .last()
                            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                                description: "MTP proposal chain lost its prior draft token",
                            })?
                    };
                    let forward_output = self.build_mtp_draft_graph(
                        hidden_states_for_fusion,
                        token_indices_for_fusion,
                        mtp_request_state,
                        performance_attribution,
                    )?;
                    let draft_token_indices =
                        self.build_greedy_token(forward_output.draft_logits())?;
                    let (_, post_normalization_hidden_states) = forward_output.into_arrays();
                    draft_token_arrays.push(draft_token_indices);
                    post_normalization_hidden_rows.push(post_normalization_hidden_states);
                    if draft_position == 0 {
                        predictor_base_checkpoint =
                            Some(mtp_request_state.allocation_checkpoint()?);
                    }
                }
                Ok::<(), Qwen3_5ExecutionError>(())
            },
        )?;

        let draft_token_references = draft_token_arrays.iter().collect::<Vec<_>>();
        let draft_token_vector = self
            .runtime()
            .concatenate_axis(&draft_token_references, 1)?;
        let mut evaluation_roots = Vec::with_capacity(post_normalization_hidden_rows.len() + 1);
        evaluation_roots.push(&draft_token_vector);
        evaluation_roots.extend(post_normalization_hidden_rows.iter());
        self.evaluate_mtp_updated_state(
            mtp_request_state,
            &evaluation_roots,
            performance_attribution,
        )?;
        let draft_token_ids = draft_token_vector.to_vec_u32()?;
        let predictor_base_checkpoint = predictor_base_checkpoint.ok_or(
            Qwen3_5ExecutionError::InvalidInput {
                description: "MTP proposal chain did not produce its target-authoritative base checkpoint",
            },
        )?;
        Ok(MtpProposalChain {
            draft_token_ids,
            predictor_base_checkpoint,
        })
    }
}

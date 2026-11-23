//! The resident model's own attributed forwards: the terminal checkpointed
//! prefill and the plain forward. The prefill forwards both models run are
//! in `qwen3_5_core::model::prefill_forward`.

use crate::qwen3_5_core::model::prefill_forward::Qwen3_5PrefillForward;
use crate::qwen3_5_core::model::target_forward_graph::Qwen3_5TargetForwardGraph;
use mlx_bridge::MlxArray;

use crate::PerformanceAttribution;

use super::Qwen3_5ResidentModel;
use super::{Qwen3_5ExecutionError, Qwen3_5TargetForwardOutput, RequestDecoderStateStack};
use crate::qwen3_5_core::decoder::{
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpoint,
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector,
};

/// Outcome of a terminal prefill that produces both boundary checkpoints and logits.
///
/// Unlike intermediate checkpointed prefill, the terminal chunk needs the vocabulary
/// head to produce first-token logits. This single-forward outcome carries both the
/// `astro_ssd_prompt_cache` checkpoints and the logits needed for first-token seeding.
pub struct Qwen3_5TerminalCheckpointPrefillOutcome {
    pub target_forward_output: Qwen3_5TargetForwardOutput,
    pub boundary_checkpoints: Vec<Qwen3_5AstroSsdPromptCacheBoundaryCheckpoint>,
}

impl Qwen3_5ResidentModel {
    /// Terminal prefill with boundary checkpoints and vocabulary logits.
    ///
    /// Unlike intermediate checkpointed prefill, the terminal chunk needs the
    /// vocabulary head to produce first-token logits. This builds the full forward
    /// graph (with lm_head) while also collecting boundary checkpoints, so the
    /// terminal chunk is a single forward pass instead of a prefix+tail split.
    #[allow(clippy::too_many_arguments)]
    pub fn terminal_prefill_chunk_with_boundary_checkpoints_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        completed_prefill_chunk_tokens: Vec<usize>,
        checkpoint_interval_token_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TerminalCheckpointPrefillOutcome, Qwen3_5ExecutionError> {
        let mut boundary_checkpoint_collector =
            Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector::new(
                completed_prefill_chunk_tokens.clone(),
                self.astro_decoder_state_cache_layout
                    .boundary_tensor_count(),
                checkpoint_interval_token_count,
            )?;
        let target_forward_output = self.build_target_forward_graph(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            Some(&mut boundary_checkpoint_collector),
            (),
            performance_attribution,
        )?;
        self.evaluate_terminal_prefill_state_with_performance_attribution(
            target_forward_output.final_logits(),
            request_decoder_state,
            Some(&boundary_checkpoint_collector),
            performance_attribution,
        )?;
        Ok(Qwen3_5TerminalCheckpointPrefillOutcome {
            target_forward_output,
            boundary_checkpoints: boundary_checkpoint_collector.complete()?,
        })
    }

    pub fn forward_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let signed_token_ids = token_ids
            .iter()
            .map(|token_id| {
                i32::try_from(*token_id).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                    description: "token ID exceeds the MLX int32 range",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let token_indices = self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_ids.len() as i32])?;
        let final_logits = self.build_forward_graph(
            &token_indices,
            token_ids.len() as i32,
            starting_position_tokens,
            request_decoder_state,
            (),
            performance_attribution,
        )?;
        self.evaluate_forward_state_with_performance_attribution(
            &final_logits,
            request_decoder_state,
            performance_attribution,
        )?;
        Ok(final_logits)
    }
}

impl Qwen3_5PrefillForward for Qwen3_5ResidentModel {
    const PRODUCTION_FORWARD_MODE: () = ();

    fn evaluate_prefill_decoder_state(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.evaluate_decoder_state_with_performance_attribution(
            request_decoder_state,
            boundary_checkpoint_collector,
            performance_attribution,
        )
    }
}

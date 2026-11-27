//! The prefill forwards both Qwen3.5 text models run, with performance
//! attribution: plain, with visual embeddings, and with
//! `astro_ssd_prompt_cache` boundary checkpoints.
//!
//! The models differ only in how a prefill's decoder state is evaluated:
//! the streaming model also resolves the expert routes the forward deferred.
//! Each supplies that through
//! [`Qwen3_5PrefillForward::evaluate_prefill_decoder_state`].

use mlx_bridge::MlxArray;

use crate::PerformanceAttribution;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::forward_contract;
use crate::qwen3_5_core::vision::visual_embedding_injection;
use text_generation::execution_error::TextExecutionError;
use text_generation::{
    AstroSsdPromptCacheBoundaryCheckpoint, AstroSsdPromptCacheBoundaryCheckpointCollector,
    RequestDecoderStateStack,
};

use super::target_forward_graph::Qwen3_5TargetForwardGraph;

/// What a checkpointed prefill leaves: how many visual embedding rows it
/// read, and the `astro_ssd_prompt_cache` boundary checkpoints it captured.
pub struct Qwen3_5BoundaryCheckpointPrefillOutcome {
    pub consumed_visual_embedding_count: usize,
    pub boundary_checkpoints: Vec<AstroSsdPromptCacheBoundaryCheckpoint>,
}

/// A Qwen3.5 text model's prefill forwards. A model supplies how it
/// evaluates a prefill's decoder state and which forward mode serves
/// requests; the default methods run every prefill around them.
pub trait Qwen3_5PrefillForward: Qwen3_5TargetForwardGraph {
    /// The forward mode production requests run in.
    const PRODUCTION_FORWARD_MODE: Self::ForwardMode;

    /// Evaluates the decoder state a prefill forward built, with its
    /// boundary checkpoints when it captured any.
    fn evaluate_prefill_decoder_state(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<&AstroSsdPromptCacheBoundaryCheckpointCollector>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError>;

    // Visual-prefill inputs stay explicit rather than introducing a parameter facade.
    #[allow(clippy::too_many_arguments)]
    fn prefill_chunk_with_visual_embeddings_and_performance_attribution(
        &self,
        chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        image_pad_token_id: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<usize, TextExecutionError> {
        let consumed_visual_embedding_count = self.build_visual_prefill_graph(
            chunk_token_ids,
            starting_position_tokens,
            visual_embeddings,
            starting_visual_embedding_index,
            request_decoder_state,
            image_pad_token_id,
            None,
            performance_attribution,
        )?;
        self.evaluate_prefill_decoder_state(request_decoder_state, None, performance_attribution)?;
        Ok(consumed_visual_embedding_count)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_visual_prefill_graph(
        &self,
        chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        image_pad_token_id: u32,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<usize, TextExecutionError> {
        let token_count = forward_contract::validate_forward_input(
            chunk_token_ids,
            starting_position_tokens,
            None,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.vocabulary_size(),
            self.config.maximum_position_count(),
        )?;
        let token_indices = self.token_indices_without_validation(chunk_token_ids)?;
        let text_embeddings = self.context_free_token_meanings(&token_indices)?;
        let (injected_embeddings, consumed_visual_embedding_count) =
            visual_embedding_injection::qwen3_5_inject_visual_embeddings(
                &self.runtime,
                &text_embeddings,
                chunk_token_ids,
                visual_embeddings,
                starting_visual_embedding_index,
                image_pad_token_id,
            )?;
        drop(self.build_forward_graph_from_embeddings(
            injected_embeddings,
            token_count,
            starting_position_tokens,
            request_decoder_state,
            boundary_checkpoint_collector,
            Self::PRODUCTION_FORWARD_MODE,
            performance_attribution,
        )?);
        Ok(consumed_visual_embedding_count)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_chunk_with_visual_embeddings_and_boundary_checkpoints_with_performance_attribution(
        &self,
        chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        image_pad_token_id: u32,
        completed_prefill_chunk_tokens: Vec<usize>,
        checkpoint_interval_token_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5BoundaryCheckpointPrefillOutcome, TextExecutionError> {
        let mut boundary_checkpoint_collector =
            AstroSsdPromptCacheBoundaryCheckpointCollector::new(
                completed_prefill_chunk_tokens.clone(),
                self.astro_decoder_state_cache_layout
                    .boundary_tensor_count(),
                checkpoint_interval_token_count,
            )?;
        let consumed_visual_embedding_count = self.build_visual_prefill_graph(
            chunk_token_ids,
            starting_position_tokens,
            visual_embeddings,
            starting_visual_embedding_index,
            request_decoder_state,
            image_pad_token_id,
            Some(&mut boundary_checkpoint_collector),
            performance_attribution,
        )?;
        self.evaluate_prefill_decoder_state(
            request_decoder_state,
            Some(&boundary_checkpoint_collector),
            performance_attribution,
        )?;
        Ok(Qwen3_5BoundaryCheckpointPrefillOutcome {
            consumed_visual_embedding_count,
            boundary_checkpoints: boundary_checkpoint_collector.complete()?,
        })
    }

    fn prefill_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        let token_indices = self.token_indices_without_validation(token_ids)?;
        let graph_started_at = std::time::Instant::now();
        self.build_prefill_decoder_state_graph(
            &token_indices,
            token_ids.len() as i32,
            starting_position_tokens,
            request_decoder_state,
            None,
            Self::PRODUCTION_FORWARD_MODE,
            performance_attribution,
        )?;
        let graph_elapsed = graph_started_at.elapsed();
        let eval_started_at = std::time::Instant::now();
        self.evaluate_prefill_decoder_state(request_decoder_state, None, performance_attribution)?;
        let eval_elapsed = eval_started_at.elapsed();
        if graph_elapsed + eval_elapsed > std::time::Duration::from_secs(5) {
            tracing::info!(
                token_count = token_ids.len(),
                graph_elapsed_millis = graph_elapsed.as_millis(),
                eval_elapsed_millis = eval_elapsed.as_millis(),
                "slow prefill chunk"
            );
        }
        Ok(())
    }

    fn prefill_chunk_with_boundary_checkpoints_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        completed_prefill_chunk_tokens: Vec<usize>,
        checkpoint_interval_token_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5BoundaryCheckpointPrefillOutcome, TextExecutionError> {
        let mut boundary_checkpoint_collector =
            AstroSsdPromptCacheBoundaryCheckpointCollector::new(
                completed_prefill_chunk_tokens.clone(),
                self.astro_decoder_state_cache_layout
                    .boundary_tensor_count(),
                checkpoint_interval_token_count,
            )?;
        let token_indices = self.token_indices_without_validation(token_ids)?;
        self.build_prefill_decoder_state_graph(
            &token_indices,
            token_ids.len() as i32,
            starting_position_tokens,
            request_decoder_state,
            Some(&mut boundary_checkpoint_collector),
            Self::PRODUCTION_FORWARD_MODE,
            performance_attribution,
        )?;
        self.evaluate_prefill_decoder_state(
            request_decoder_state,
            Some(&boundary_checkpoint_collector),
            performance_attribution,
        )?;
        Ok(Qwen3_5BoundaryCheckpointPrefillOutcome {
            consumed_visual_embedding_count: 0,
            boundary_checkpoints: boundary_checkpoint_collector.complete()?,
        })
    }
}

impl Qwen3_5ModelBase {
    /// The `[1, token_count]` index array of a prefill's token ids; only the
    /// int32 range is checked here.
    pub fn token_indices_without_validation(
        &self,
        token_ids: &[u32],
    ) -> Result<MlxArray, TextExecutionError> {
        let mut signed_token_ids = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            signed_token_ids.push(i32::try_from(*token_id).map_err(|_| {
                TextExecutionError::InvalidInput {
                    description: "token ID exceeds the MLX int32 range",
                }
            })?);
        }
        Ok(self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_ids.len() as i32])?)
    }
}

//! The streaming model's part of the target forward graph: its decoder-layer
//! loop. The steps around it are shared with the resident model in
//! `qwen3_5_core::model::target_forward_graph`.

use mlx_bridge::MlxArray;

use crate::qwen3_5_core::model::target_forward_graph::{
    Qwen3_5ForwardPositions, Qwen3_5TargetForwardGraph, Qwen3_5TargetForwardOutput,
};
use crate::qwen3_5_streaming::Qwen3_5MoEPagedExecutionMode;
use crate::qwen4_exp::request_state::layer_and_qwen4_exp_state;
use crate::{AstroDecoderStateCacheState, PerformanceAttribution};
use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

use super::Qwen3_5Model;
use super::layer_submission::{CompletedLayerRoots, LayerGroupSubmission};
use super::{RequestDecoderStateStack, TextExecutionError};

impl Qwen3_5Model {
    /// Executes a target forward and retains pre-final-normalization rows for specialized consumers.
    pub fn forward_chunk_with_pre_final_normalization_hidden_states(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Qwen3_5TargetForwardOutput, TextExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        let target_forward_output = self.build_target_forward_graph(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            None,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            &mut disabled_performance_attribution,
        )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)?;
        self.runtime
            .evaluate_arrays(&[target_forward_output.pre_final_normalization_hidden_states()])?;
        Ok(target_forward_output)
    }

    /// Each forward owns one deferred missing-route collection. Clears any
    /// leftover roots from a cancelled or failed previous attempt.
    fn discard_previous_forward_leftovers(&self) {
        self.clear_paged_forward_missing_route_roots();
        self.route_observation
            .borrow_mut()
            .discard_pending_layer_route_arrays();
        self.deferred_expert_demand
            .borrow_mut()
            .discard_pending_layer_routes();
        self.discard_prefill_complete_layer_prefetch();
    }
}

impl Qwen3_5TargetForwardGraph for Qwen3_5Model {
    type ForwardMode = Qwen3_5MoEPagedExecutionMode;

    /// Runs every layer, reading a dense model's streamed layers and a sparse
    /// model's expert pages, and submits each finished layer itself.
    fn build_decoder_layers(
        &self,
        mut hidden_states: MlxArray,
        forward_positions: Qwen3_5ForwardPositions<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
        mut boundary_checkpoint_collector: Option<
            &mut AstroSsdPromptCacheBoundaryCheckpointCollector,
        >,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        self.discard_previous_forward_leftovers();
        let rope_offset_tokens = forward_positions.rope_offset_tokens()?;
        let token_count = forward_positions.token_count;
        let decoder_layer_count = self.config.layer_count() as usize;
        // A multi-token forward finishes each layer before the next
        // (`LayerSubmission::PromptLayer`), so the interval resolved here
        // only ever paces a one-token forward's command buffers.
        let pacing_token_count = self
            .paged_forward_missing_route_collector
            .pacing_token_count(token_count);
        let graph_submission_layer_interval =
            self.reply_graph_submission_layer_interval(pacing_token_count);
        let mut previous_layer_output = None;
        for layer_index in 0..decoder_layer_count {
            // A dense model's streamer supplies its layers: a kept layer, or
            // one read from the SSD for this forward alone.
            let forward_layer_weights = self.forward_layer_weights(layer_index)?;
            let decoder_layer_weights = forward_layer_weights.decoder_layer_weights();
            let (layer_model_state, qwen4_exp_state) =
                layer_and_qwen4_exp_state(request_decoder_state, layer_index)?;
            hidden_states = self.forward_decoder_layer(
                &hidden_states,
                token_count,
                rope_offset_tokens,
                layer_index,
                decoder_layer_weights,
                layer_model_state,
                forward_positions.token_position_offsets,
                boundary_checkpoint_collector.as_deref_mut(),
                paged_execution_mode,
                qwen4_exp_state,
                performance_attribution,
            )?;
            self.paged_forward_missing_route_collector
                .draft_tap_capture
                .keep_layer_rows(layer_index, rope_offset_tokens, &hidden_states);
            self.submit_completed_layer(
                &forward_layer_weights,
                completed_layer_roots(
                    &hidden_states,
                    layer_index,
                    request_decoder_state,
                    boundary_checkpoint_collector.as_deref(),
                ),
                &mut previous_layer_output,
                LayerGroupSubmission {
                    completed_layer_count: layer_index + 1,
                    decoder_layer_count,
                    graph_submission_layer_interval,
                    forward_token_count: pacing_token_count,
                },
                performance_attribution,
            )?;
            // Expert pages are lazy loads on MLX's own stream, so this read
            // overlaps the layers still computing instead of stalling graph
            // submission; its bytes are the plan's second prefill stream slot.
            if pacing_token_count > 1
                && matches!(
                    paged_execution_mode,
                    Qwen3_5MoEPagedExecutionMode::ProductionDefault
                )
            {
                self.read_next_chunk_layer_ahead(layer_index, token_count, performance_attribution);
            }
        }
        self.discard_prefill_complete_layer_prefetch();
        Ok(hidden_states)
    }
}

/// The arrays one completed layer leaves to evaluate: its output, plus the
/// convolution state and `astro_ssd_prompt_cache` boundary snapshots it does
/// not reach.
fn completed_layer_roots<'layer>(
    layer_output: &'layer MlxArray,
    layer_index: usize,
    request_decoder_state: &'layer RequestDecoderStateStack,
    boundary_checkpoint_collector: Option<&'layer AstroSsdPromptCacheBoundaryCheckpointCollector>,
) -> CompletedLayerRoots<'layer> {
    CompletedLayerRoots {
        layer_output,
        convolution_state: request_decoder_state
            .layer(layer_index)
            .and_then(AstroDecoderStateCacheState::convolution_state_array),
        boundary_checkpoint_states: boundary_checkpoint_collector
            .map(|collector| collector.layer_evaluation_arrays(layer_index))
            .unwrap_or_default(),
    }
}

//! The resident model's part of the target forward graph: its decoder-layer
//! loop. The steps around it are shared with the streaming model in
//! `qwen3_5_core::model::target_forward_graph`.

use mlx_bridge::{MlxArray, MlxDtype};

use crate::qwen3_5_core::model::target_forward_graph::{
    Qwen3_5ForwardPositions, Qwen3_5TargetForwardGraph, Qwen3_5TargetForwardOutput,
    validated_layer_decoder_state,
};
use crate::{PerformanceAttribution, PerformanceOperation};
use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

use super::Qwen3_5ResidentModel;
use super::{RequestDecoderStateStack, TextExecutionError};

impl Qwen3_5ResidentModel {
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
            (),
            &mut disabled_performance_attribution,
        )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)?;
        self.runtime
            .evaluate_arrays(&[target_forward_output.pre_final_normalization_hidden_states()])?;
        Ok(target_forward_output)
    }

    /// Vocabulary logits at every position of a forward, for the
    /// quantization probe's KL divergence, which averages over positions:
    /// the same final normalization and head as `final_vocabulary_logits`
    /// without its last-row slice.
    ///
    /// The head runs in Float32 and the result is stored as Float16.
    /// bf16 logits were measured to put a KL floor of about 5e-4 under every
    /// probe (at logit magnitudes of 16 to 32 bf16 resolves 0.125, and the
    /// reference and the probe round independently), which buried the
    /// 4-bit versus 8-bit signal. Float16 keeps three more mantissa bits,
    /// cutting that floor about 64-fold, at half the memory of Float32 for
    /// the reference logits the sweep holds across every module.
    pub fn all_position_vocabulary_logits(
        &self,
        pre_final_normalization_hidden_states: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        let normalized_states = self.runtime.rms_norm(
            pre_final_normalization_hidden_states,
            &self.weights.final_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let float32_states = self.runtime.astype(&normalized_states, MlxDtype::Float32)?;
        let float32_logits =
            self.quantized_linear(&float32_states, &self.weights.language_model_head_weights)?;
        Ok(self.runtime.astype(&float32_logits, MlxDtype::Float16)?)
    }
}

impl Qwen3_5TargetForwardGraph for Qwen3_5ResidentModel {
    /// The resident model runs every forward one way.
    type ForwardMode = ();

    /// Runs the layers this model holds. A multi-token prefill may submit
    /// completed layer groups so macOS can schedule other graphics work
    /// between command buffers.
    fn build_decoder_layers(
        &self,
        mut hidden_states: MlxArray,
        forward_positions: Qwen3_5ForwardPositions<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
        mut boundary_checkpoint_collector: Option<
            &mut AstroSsdPromptCacheBoundaryCheckpointCollector,
        >,
        _forward_mode: (),
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let rope_offset_tokens = forward_positions.rope_offset_tokens()?;
        let token_count = forward_positions.token_count;
        let decoder_layer_count = self.config.layer_count() as usize;
        let graph_submission_layer_interval =
            self.graph_submission_layer_interval(token_count, false);
        for layer_index in 0..decoder_layer_count {
            let decoder_layer_weights = self
                .weights
                .decoder_layer_weights
                .get(layer_index)
                .ok_or(TextExecutionError::MissingDecoderLayerWeights { layer_index })?;
            let layer_model_state =
                validated_layer_decoder_state(request_decoder_state, layer_index)?;
            hidden_states = self.forward_decoder_layer(
                &hidden_states,
                token_count,
                rope_offset_tokens,
                layer_index,
                decoder_layer_weights,
                layer_model_state,
                forward_positions.token_position_offsets,
                boundary_checkpoint_collector.as_deref_mut(),
                performance_attribution,
            )?;
            if graph_submission_layer_interval > 0
                && (layer_index + 1).is_multiple_of(graph_submission_layer_interval)
                && layer_index + 1 < decoder_layer_count
            {
                // Intermediate mlx_async_eval commits the completed layer group.
                // Do not submit after the final decoder layer: final
                // normalization and logits extend that same graph and the
                // caller owns the terminal evaluation.
                performance_attribution.measure_operation(
                    PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                    |_performance_attribution| self.runtime.async_eval_arrays(&[&hidden_states]),
                )?;
            }
            // Do not start a second complete-layer host read here. The positional
            // reader is synchronous on this thread, so a prefetch stalls GPU
            // submission and holds an unbudgeted extra complete layer.
        }
        Ok(hidden_states)
    }
}

//! The full-attention decoder layer of the straight forward.
//!
//! The normalized layer input is rotated once and feeds one fused packed
//! matmul whose output splits into the query, key, and value projections.
//! The rest mirrors the core's forward exactly:
//! gated-query split, q/k RMS norms, RoPE, append-only KV state, and the
//! shared attention step with the learned output gate.

use mlx_c_rust::MlxArray;

use crate::qwen3_5_core::model::full_attention::qwen3_5_full_attention_step;

use super::super::PrismHadamardModelError;
use super::super::weights::FullLayerWeights;
use super::StraightGenerationState;

impl super::super::PrismHadamardTextModel {
    #[expect(clippy::too_many_lines)]
    pub(super) fn forward_full_attention_layer(
        &self,
        normalized_input: &MlxArray,
        token_count: i32,
        layer: &FullLayerWeights,
        rms_norm_epsilon: f32,
        state_slot: usize,
        state: &mut StraightGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let text = self.pack.config().text();
        let query_head_count = text.num_attention_heads as i32;
        let key_value_head_count = text.num_key_value_heads as i32;
        let attention_head_dimension = text.head_dim as i32;
        let attention_projection_head_dimension = attention_head_dimension * 2;
        let rope_offset_tokens = state.sequence_position_offset;

        // One rotation of the layer input feeds the one fused launch that
        // produces all three rotated projections.
        let rotated_input = self
            .rotate(normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let [query_projection, keys, values] = layer
            .query_key_value
            .forward(&self.runtime, &rotated_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let query_projection = self
            .runtime
            .reshape(
                &query_projection,
                &[
                    1,
                    token_count,
                    query_head_count,
                    attention_projection_head_dimension,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let queries = super::super::weights::slice_last_dimension(
            &self.runtime,
            &query_projection,
            0,
            attention_head_dimension,
        )?;
        let output_gate = super::super::weights::slice_last_dimension(
            &self.runtime,
            &query_projection,
            attention_head_dimension,
            attention_projection_head_dimension,
        )?;
        let output_gate = self
            .runtime
            .reshape(
                &output_gate,
                &[1, token_count, query_head_count * attention_head_dimension],
            )
            .map_err(PrismHadamardModelError::Forward)?;

        let keys = self
            .runtime
            .reshape(
                &keys,
                &[
                    1,
                    token_count,
                    key_value_head_count,
                    attention_head_dimension,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let values = self
            .runtime
            .reshape(
                &values,
                &[
                    1,
                    token_count,
                    key_value_head_count,
                    attention_head_dimension,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;

        let normalized_queries = self
            .runtime
            .rms_norm(&queries, &layer.query_normalization, rms_norm_epsilon)
            .map_err(PrismHadamardModelError::Forward)?;
        let normalized_keys = self
            .runtime
            .rms_norm(&keys, &layer.key_normalization, rms_norm_epsilon)
            .map_err(PrismHadamardModelError::Forward)?;

        let transposed_queries = self
            .runtime
            .transpose_axes(&normalized_queries, &[0, 2, 1, 3])
            .map_err(PrismHadamardModelError::Forward)?;
        let transposed_keys = self
            .runtime
            .transpose_axes(&normalized_keys, &[0, 2, 1, 3])
            .map_err(PrismHadamardModelError::Forward)?;
        let transposed_values = self
            .runtime
            .transpose_axes(&values, &[0, 2, 1, 3])
            .map_err(PrismHadamardModelError::Forward)?;

        let rope_base = f32::from_bits(text.rope_theta_bits);
        let rotary_dimension =
            (text.head_dim as f32 * f32::from_bits(text.partial_rotary_factor_bits)) as i32;
        let rotated_queries = self
            .runtime
            .rope(
                &transposed_queries,
                rotary_dimension,
                rope_base,
                rope_offset_tokens,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let rotated_keys = self
            .runtime
            .rope(
                &transposed_keys,
                rotary_dimension,
                rope_base,
                rope_offset_tokens,
            )
            .map_err(PrismHadamardModelError::Forward)?;

        let key_value_state = &mut state.key_value_states[state_slot];
        let previous_storage_offset_tokens = key_value_state.offset_tokens();
        let (active_keys, active_values) = key_value_state
            .update_and_fetch(
                &self.runtime,
                &rotated_keys,
                &transposed_values,
                previous_storage_offset_tokens,
            )
            .map_err(PrismHadamardModelError::ForwardRuntime)?;

        let attention_scale = 1.0 / (text.head_dim as f32).sqrt();
        let gated_output = qwen3_5_full_attention_step(
            &self.runtime,
            &self.compiled_elementwise_graphs,
            &rotated_queries,
            &active_keys,
            &active_values,
            &output_gate,
            attention_scale,
        )
        .map_err(PrismHadamardModelError::ForwardRuntime)?;
        let rotated_gated_output = self
            .rotate(&gated_output)
            .map_err(PrismHadamardModelError::Forward)?;
        layer
            .output
            .forward(&self.runtime, &rotated_gated_output)
            .map_err(PrismHadamardModelError::Forward)
    }
}

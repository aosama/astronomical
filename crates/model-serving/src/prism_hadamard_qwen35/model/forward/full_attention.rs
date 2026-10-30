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
use super::super::fused_projection::FusedPackedProjection;
use super::super::fwht::RotatedActivations;
use super::super::weights::FullLayerWeights;
use super::StraightGenerationState;

/// Head geometry of one full-attention layer, in the i32 form the runtime
/// reshape calls take.
struct AttentionHeadDimensions {
    query_head_count: i32,
    key_value_head_count: i32,
    attention_head_dimension: i32,
    attention_projection_head_dimension: i32,
}

/// Derives the head geometry from the validated pack dimensions. The fused
/// query projection carries every query head interleaved with its learned
/// output gate, so its per-head width is exactly two head dimensions.
fn attention_head_dimensions(
    query_head_count: u32,
    key_value_head_count: u32,
    head_dimension: u32,
) -> AttentionHeadDimensions {
    let attention_head_dimension = head_dimension as i32;
    AttentionHeadDimensions {
        query_head_count: query_head_count as i32,
        key_value_head_count: key_value_head_count as i32,
        attention_head_dimension,
        attention_projection_head_dimension: attention_head_dimension * 2,
    }
}

/// The packed RoPE schedule shared by the query and key rotary embeddings.
struct RotaryPositionParameters {
    base_frequency: f32,
    rotary_dimension: i32,
}

/// Derives the RoPE schedule. Theta and the partial-rotary fraction ship as
/// raw bits so the packed config stays bit-identical to the validated core
/// contract; the rotary dimension covers only the fraction of each head the
/// pack declares.
fn rotary_position_parameters(
    rope_theta_bits: u32,
    partial_rotary_factor_bits: u32,
    head_dimension: u32,
) -> RotaryPositionParameters {
    RotaryPositionParameters {
        base_frequency: f32::from_bits(rope_theta_bits),
        rotary_dimension: (head_dimension as f32 * f32::from_bits(partial_rotary_factor_bits))
            as i32,
    }
}

/// The dot-product scale that keeps attention logits variance-stable: one
/// over the square root of the head dimension.
fn attention_scale(attention_head_dimension: i32) -> f32 {
    1.0 / (attention_head_dimension as f32).sqrt()
}

/// The fused query/key/value projection split into its attention inputs plus
/// the learned output gate carried in the query projection's upper half.
struct QueryKeyValueProjections {
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
    output_gate: MlxArray,
}

/// RMS-normalized projections with the heads moved to the leading axes the
/// RoPE and KV-append steps read.
struct TransposedProjections {
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
}

/// Position-encoded queries and keys; the values keep their transposed
/// layout because RoPE never touches them and the KV append consumes them
/// as-is.
struct RotatedProjections {
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
}

impl super::super::PrismHadamardTextModel {
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
        let head_dimensions = attention_head_dimensions(
            text.num_attention_heads,
            text.num_key_value_heads,
            text.head_dim,
        );
        let rope_offset_tokens = state.sequence_position_offset;

        // One rotation of the layer input feeds the one fused launch that
        // produces all three rotated projections.
        let rotated_input = self
            .rotate(normalized_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let projections = self.project_rotated_input(
            &rotated_input,
            &layer.query_key_value,
            token_count,
            &head_dimensions,
        )?;
        let transposed_projections =
            self.normalize_and_transpose_projections(&projections, layer, rms_norm_epsilon)?;
        let rotary_parameters = rotary_position_parameters(
            text.rope_theta_bits,
            text.partial_rotary_factor_bits,
            text.head_dim,
        );
        let rotated_projections = self.apply_rotary_position_encodings(
            transposed_projections,
            &rotary_parameters,
            rope_offset_tokens,
        )?;
        let (active_keys, active_values) =
            self.append_keys_and_fetch_active_values(state, state_slot, &rotated_projections)?;
        let attention_scale = attention_scale(head_dimensions.attention_head_dimension);
        self.attend_and_project_output(
            layer,
            &rotated_projections,
            &active_keys,
            &active_values,
            &projections.output_gate,
            attention_scale,
        )
    }

    /// Splits the one fused rotated projection into queries, keys, and
    /// values, and separates the learned output gate from the query heads,
    /// everything reshaped to head layout.
    fn project_rotated_input(
        &self,
        rotated_input: &RotatedActivations,
        query_key_value: &FusedPackedProjection<3>,
        token_count: i32,
        head_dimensions: &AttentionHeadDimensions,
    ) -> Result<QueryKeyValueProjections, PrismHadamardModelError> {
        let [query_projection, keys, values] = query_key_value
            .forward(&self.runtime, rotated_input)
            .map_err(PrismHadamardModelError::Forward)?;
        let query_projection = self
            .runtime
            .reshape(
                &query_projection,
                &[
                    1,
                    token_count,
                    head_dimensions.query_head_count,
                    head_dimensions.attention_projection_head_dimension,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let queries = super::super::weights::slice_last_dimension(
            &self.runtime,
            &query_projection,
            0,
            head_dimensions.attention_head_dimension,
        )?;
        let output_gate = super::super::weights::slice_last_dimension(
            &self.runtime,
            &query_projection,
            head_dimensions.attention_head_dimension,
            head_dimensions.attention_projection_head_dimension,
        )?;
        let output_gate = self
            .runtime
            .reshape(
                &output_gate,
                &[
                    1,
                    token_count,
                    head_dimensions.query_head_count * head_dimensions.attention_head_dimension,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let keys = self
            .runtime
            .reshape(
                &keys,
                &[
                    1,
                    token_count,
                    head_dimensions.key_value_head_count,
                    head_dimensions.attention_head_dimension,
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
                    head_dimensions.key_value_head_count,
                    head_dimensions.attention_head_dimension,
                ],
            )
            .map_err(PrismHadamardModelError::Forward)?;
        Ok(QueryKeyValueProjections {
            queries,
            keys,
            values,
            output_gate,
        })
    }

    /// RMS-normalizes the queries and keys against their layer norms and
    /// moves the head axis forward for every projection.
    fn normalize_and_transpose_projections(
        &self,
        projections: &QueryKeyValueProjections,
        layer: &FullLayerWeights,
        rms_norm_epsilon: f32,
    ) -> Result<TransposedProjections, PrismHadamardModelError> {
        let normalized_queries = self
            .runtime
            .rms_norm(
                &projections.queries,
                &layer.query_normalization,
                rms_norm_epsilon,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let normalized_keys = self
            .runtime
            .rms_norm(
                &projections.keys,
                &layer.key_normalization,
                rms_norm_epsilon,
            )
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
            .transpose_axes(&projections.values, &[0, 2, 1, 3])
            .map_err(PrismHadamardModelError::Forward)?;
        Ok(TransposedProjections {
            queries: transposed_queries,
            keys: transposed_keys,
            values: transposed_values,
        })
    }

    /// Applies the partial rotary embedding to the queries and keys; the
    /// values pass through because RoPE never touches them.
    fn apply_rotary_position_encodings(
        &self,
        transposed_projections: TransposedProjections,
        rotary_parameters: &RotaryPositionParameters,
        rope_offset_tokens: i32,
    ) -> Result<RotatedProjections, PrismHadamardModelError> {
        let TransposedProjections {
            queries,
            keys,
            values,
        } = transposed_projections;
        let rotated_queries = self
            .runtime
            .rope(
                &queries,
                rotary_parameters.rotary_dimension,
                rotary_parameters.base_frequency,
                rope_offset_tokens,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        let rotated_keys = self
            .runtime
            .rope(
                &keys,
                rotary_parameters.rotary_dimension,
                rotary_parameters.base_frequency,
                rope_offset_tokens,
            )
            .map_err(PrismHadamardModelError::Forward)?;
        Ok(RotatedProjections {
            queries: rotated_queries,
            keys: rotated_keys,
            values,
        })
    }

    /// Appends the rotated keys and transposed values to the slot's
    /// append-only KV state and fetches the active views covering every
    /// processed token.
    fn append_keys_and_fetch_active_values(
        &self,
        state: &mut StraightGenerationState,
        state_slot: usize,
        rotated_projections: &RotatedProjections,
    ) -> Result<(MlxArray, MlxArray), PrismHadamardModelError> {
        let key_value_state = &mut state.key_value_states[state_slot];
        let previous_storage_offset_tokens = key_value_state.offset_tokens();
        key_value_state
            .update_and_fetch(
                &self.runtime,
                &rotated_projections.keys,
                &rotated_projections.values,
                previous_storage_offset_tokens,
            )
            .map_err(PrismHadamardModelError::ForwardRuntime)
    }

    /// Runs the shared attention step over the active KV views with the
    /// learned output gate, then rotates and projects the gated result back
    /// to the residual width.
    fn attend_and_project_output(
        &self,
        layer: &FullLayerWeights,
        rotated_projections: &RotatedProjections,
        active_keys: &MlxArray,
        active_values: &MlxArray,
        output_gate: &MlxArray,
        attention_scale: f32,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let gated_output = qwen3_5_full_attention_step(
            &self.runtime,
            &self.compiled_elementwise_graphs,
            &rotated_projections.queries,
            active_keys,
            active_values,
            output_gate,
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

#[cfg(test)]
#[path = "full_attention_tests.rs"]
mod full_attention_tests;

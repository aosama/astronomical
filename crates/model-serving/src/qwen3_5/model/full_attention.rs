//! A beginner's guide to the full-attention path used by Qwen3.5.
//!
//! The foundational attention calculation is:
//!     attention(Q, K, V) = softmax((Q @ K-transpose) / sqrt(d)) @ V
//!
//! Here d is the number of features in one attention head. A head is one small,
//! separate attention calculation; the model runs many heads in parallel and
//! joins their outputs afterward.
//! Each new query token compares itself with each available key token. The
//! score of a query/key pair says how relevant that older token is to the
//! current token. Softmax converts all scores for one query into weights that
//! add up to one. Those weights then blend the value vectors into the output.
//!
//! If a prompt chunk has n query tokens and n available key/value tokens, this
//! conceptual comparison table has n x n entries. That is the O(n^2) work that
//! makes long prompt processing expensive. MLX executes the formula as one
//! fused operation. Its Metal implementation can process score tiles and consume them immediately, rather than requiring the whole n x n table to
//! exist in memory at once.
//!
//! During token generation, this file processes one new query token at a time.
//! Its comparison with the existing prefix is O(n) for that one step. The KV
//! state below preserves old keys and values, avoiding their recomputation.
//! Generating many tokens against a growing prefix still adds up to quadratic
//! work over the entire response.
//!
//! The tensors passed to MLX have these shapes:
//!
//! - queries: [batch, query_heads, query_tokens, features_per_head]
//! - keys: [batch, key_value_heads, active_tokens, features_per_head]
//! - values: [batch, key_value_heads, active_tokens, features_per_head]
//!
//! Qwen3.5 uses grouped-query attention: it has more query heads than
//! key/value heads, so one key/value head is shared by a group of query heads.
//! This reduces the KV memory needed for long contexts without changing the
//! basic attention calculation.

// MlxArray is an MLX tensor handle. These methods normally build a lazy MLX
// graph; actual graphics-processor evaluation happens at a later boundary.
use astronomical_runtime_integration::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxRuntime, MlxRuntimeError,
};

use super::Qwen3_5ExecutionError;
use super::attention_execution::sequential_causal_attention;
use super::decoder_layer_weights::Qwen3_5FullAttentionWeights;
use super::tensor_slicing::slice_last_dimension;
use super::{Qwen3_5AttentionCapture, model::Qwen3_5Model};
use crate::decoder_cache::FullAttentionKeyValueState;
use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;

const FULL_ATTENTION_OPERATION: &str = "apply one Qwen3.5 full-attention step";

/// Applies the post-projection Qwen3.5 full-attention formula over the
/// already-fetched active keys and values. The KV state owner is responsible
/// for capacity growth and concatenation; this function runs only the
/// scaled dot-product attention and the output gate.
///
/// `active_keys` and `active_values` are the views over `[0..offset]` returned
/// by `FullAttentionKeyValueState::update_and_fetch`. They already contain the
/// newly rotated keys and transposed values appended to the cached prefix.
/// `rotated_queries` are the post-RoPE queries ready for the attention call.
///
/// The MLX attention calls in this function execute the complete formula from
/// the module documentation: scaled Q @ K-transpose, causal masking when
/// needed, softmax, and weighted V. This function prepares neither the learned
/// Q/K/V projections nor the KV storage; its job is only the core comparison
/// calculation and Qwen's output gate.
///
/// A multi-token prompt chunk uses causal attention. A token must not learn
/// from a later token in the same prompt, because that later token would not
/// yet exist when the model normally generates text. One-token decode needs no
/// explicit mask: the active key/value view ends at the current token, so there
/// is no later token to hide.
pub fn qwen3_5_full_attention_step(
    runtime: &MlxRuntime,
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    rotated_queries: &MlxArray,
    active_keys: &MlxArray,
    active_values: &MlxArray,
    output_gate: &MlxArray,
    attention_scale: f32,
    paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
) -> Result<MlxArray, MlxRuntimeError> {
    // Fail early with a model-serving error if the tensors cannot represent the
    // matrix operations below. This is more informative than a native failure.
    let attention_shape = validate_full_attention_arguments(
        rotated_queries,
        active_keys,
        active_values,
        output_gate,
        attention_scale,
    )?;

    // More than one query token means this is prompt processing. The causal
    // mask lets each token attend only to itself and earlier positions.
    let is_causal = attention_shape.query_token_count > 1;
    let should_process_query_rows_sequentially = is_causal
        && paged_prefill_execution_mode
            == Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow;
    let attention_output = if should_process_query_rows_sequentially {
        sequential_causal_attention(
            runtime,
            rotated_queries,
            active_keys,
            active_values,
            attention_scale,
            attention_shape.query_token_count,
            attention_shape.active_key_value_token_count,
        )?
    } else if is_causal {
        // This is the O(query_tokens x active_tokens) relationship calculation.
        // MLX fuses scaling, causal masking, softmax, and V weighting so Rust
        // does not materialize intermediate score or probability tensors.
        runtime.causal_scaled_dot_product_attention(
            rotated_queries,
            active_keys,
            active_values,
            attention_scale,
        )?
    } else {
        // A generated token supplies one query against every cached key. That
        // is O(active_tokens) for this individual decode step. Reusing stored
        // K and V is the important optimization over recomputing the prefix.
        runtime.scaled_dot_product_attention(
            rotated_queries,
            active_keys,
            active_values,
            attention_scale,
        )?
    };

    // MLX returns [batch, heads, tokens, features_per_head]. The rest of this
    // transformer expects token-oriented vectors, so move tokens before heads.
    let attention_output = runtime.transpose_axes(&attention_output, &[0, 2, 1, 3])?;

    // Combine all per-head output vectors into one feature vector per token:
    // [batch, tokens, heads x features_per_head]. Conceptually, reshape changes
    // only how the head and feature dimensions are grouped, not their values.
    let attention_output = runtime.reshape(
        &attention_output,
        &[
            attention_shape.batch_size,
            attention_shape.query_token_count,
            attention_shape.output_dimension,
        ],
    )?;

    // Qwen's Q projection includes a separate output gate. Sigmoid turns its
    // numbers into weights between zero and one, then elementwise multiplication
    // controls how much of each attention-output feature continues onward.
    // Prefill uses a retained compiled MLX graph for the same sigmoid/multiply
    // calculation; decode keeps the small direct graph.
    if is_causal && !should_process_query_rows_sequentially {
        runtime.apply_compiled_attention_output_gate(
            compiled_elementwise_graphs,
            &attention_output,
            output_gate,
        )
    } else {
        let gate_weights = runtime.sigmoid(output_gate)?;
        runtime.multiply(&attention_output, &gate_weights)
    }
}
#[derive(Clone, Copy)]
struct FullAttentionShape {
    // Independent examples processed together. Text serving currently uses one
    // example, but preserving this axis makes the MLX attention contract clear.
    batch_size: i32,
    // Number of new query tokens in this forward: a prompt chunk length during
    // prefill, or one during normal autoregressive token generation.
    query_token_count: i32,
    // Number of key/value positions visible to the complete forward, including
    // the newly appended query positions.
    active_key_value_token_count: i32,
    // The width after all query-head vectors are placed side by side. It is
    // query_head_count x features_per_head and matches the output gate.
    output_dimension: i32,
}
fn validate_full_attention_arguments(
    rotated_queries: &MlxArray,
    active_keys: &MlxArray,
    active_values: &MlxArray,
    output_gate: &MlxArray,
    attention_scale: f32,
) -> Result<FullAttentionShape, MlxRuntimeError> {
    // Read the layouts once. Attention tensors use four axes in this order:
    // [batch, heads, token_positions, features_per_head].
    let query_shape = rotated_queries.shape();
    let key_shape = active_keys.shape();
    let value_shape = active_values.shape();
    // The attention primitive needs all three tensors to have those four axes.
    if query_shape.len() != 4 || key_shape.len() != 4 || value_shape.len() != 4 {
        return Err(full_attention_error(
            "rotated queries, active keys, and active values must have rank four",
        ));
    }
    // K describes which positions can be selected. V carries the content that
    // will be blended when a K position gets a high softmax weight. Therefore
    // they must have exactly the same positions, heads, and feature width.
    if key_shape != value_shape {
        return Err(full_attention_error(
            "full-attention active keys and values must have identical shapes",
        ));
    }
    let batch_size = query_shape[0];
    let query_head_count = query_shape[1];
    let query_token_count = query_shape[2];
    let head_dimension = query_shape[3];
    let key_value_head_count = key_shape[1];
    let active_key_value_token_count = key_shape[2];
    // Q @ K-transpose needs matching batches and per-head feature widths.
    // Grouped-query attention permits fewer K/V heads, but their count must
    // divide the Q-head count evenly so every Q head has a matching K/V group.
    // The active prefix must also include all newly supplied query positions.
    if batch_size <= 0
        || query_token_count <= 0
        || query_head_count <= 0
        || head_dimension <= 0
        || key_value_head_count <= 0
        || key_shape[0] != batch_size
        || key_shape[3] != head_dimension
        || query_head_count % key_value_head_count != 0
        || active_key_value_token_count < query_token_count
    {
        return Err(full_attention_error(
            "full-attention dimensions must be positive, grouped-query heads must divide evenly, \
             and the active KV view must cover the new query tokens",
        ));
    }
    // Once attention has produced one vector per query head, the decoder joins
    // those vectors into a single feature vector for each token. checked_mul
    // protects the shape arithmetic from malformed model metadata.
    let output_dimension = query_head_count
        .checked_mul(head_dimension)
        .ok_or_else(|| full_attention_error("full-attention output dimension overflowed"))?;
    // The gate is stored as one flattened vector per token, so it must line up
    // exactly with that joined attention output.
    if output_gate.shape() != [batch_size, query_token_count, output_dimension] {
        return Err(full_attention_error(
            "full-attention output gate shape is incompatible with query heads",
        ));
    }
    // Dividing QK scores by sqrt(features_per_head) keeps score magnitudes in a
    // range where softmax remains numerically useful. Invalid scale values
    // would make the resulting attention probabilities meaningless.
    if !attention_scale.is_finite() || attention_scale <= 0.0 {
        return Err(full_attention_error("full-attention scale is invalid"));
    }
    Ok(FullAttentionShape {
        batch_size,
        query_token_count,
        active_key_value_token_count,
        output_dimension,
    })
}
fn full_attention_error(description: &'static str) -> MlxRuntimeError {
    // Give all failures from this small owner a stable operation label while
    // retaining the specific explanation for the request-level error path.
    MlxRuntimeError::RuntimeOperation {
        operation: FULL_ATTENTION_OPERATION,
        description: description.to_owned(),
    }
}
impl Qwen3_5Model {
    /// Runs one full-attention layer, threading the in-memory KV state through
    /// the single `FullAttentionKeyValueState` owner. The owner grows capacity
    /// in 256-token steps, appends the rotated keys and transposed values, and
    /// returns active views for the attention call.
    ///
    /// `hidden_states` starts as one model vector per new token:
    /// [batch, token_count, hidden_size]. This method transforms that tensor
    /// into Q, K, and V; gives Q and K position information; adds K and V to
    /// the running context; applies attention; and projects the result back to
    /// the normal hidden-size vector expected by the rest of the decoder layer.
    pub(crate) fn forward_full_attention(
        &self,
        hidden_states: &MlxArray,
        token_count: i32,
        rope_offset_tokens: i32,
        full_attention_weights: &Qwen3_5FullAttentionWeights,
        kv_state: &mut FullAttentionKeyValueState,
        decoder_layer_index: usize,
        token_position_offsets: Option<&MlxArray>,
        attention_capture: Option<&mut Qwen3_5AttentionCapture>,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        // The model configuration supplies the dimensions for this layer. Qwen
        // has more Q heads than K/V heads, which is grouped-query attention.
        // Every head still uses the same number of features.
        let query_head_count = self.config.query_head_count() as i32;
        let key_value_head_count = self.config.key_value_head_count() as i32;
        let attention_head_dimension = self.config.head_dimension() as i32;

        // Qwen packs two same-sized vectors into one query projection: Q and a
        // learned output gate. This doubled final dimension is split below.
        let attention_projection_head_dimension = attention_head_dimension * 2;

        // RoPE rotates a configured prefix of each Q/K vector to encode token
        // positions. RMS normalization uses this small epsilon for stability.
        let rotary_dimension = self.config.rotary_dimension() as i32;
        let rms_norm_epsilon = f32::from_bits(self.config.rms_norm_epsilon_bits());

        // First learned projection: turn each hidden-state vector into the
        // combined query-and-gate features. Quantization changes how the matrix
        // is stored and evaluated, not the mathematical role of this projection.
        let query_projection = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &full_attention_weights.query_projection,
            paged_prefill_execution_mode,
        )?;

        // Expose the Q-head axis and the two packed halves:
        // [batch, tokens, query_heads, 2 x features_per_head].
        let query_projection = self.runtime.reshape(
            &query_projection,
            &[
                1,
                token_count,
                query_head_count,
                attention_projection_head_dimension,
            ],
        )?;

        // The first half of each packed vector is Q, the vector that asks what
        // information the current token should retrieve from the context.
        let queries = slice_last_dimension(
            &self.runtime,
            &query_projection,
            0,
            attention_head_dimension,
        )?;

        // The second half is the output gate. It is not part of the QK score;
        // it controls the attention output after the value vectors are blended.
        let output_gate = slice_last_dimension(
            &self.runtime,
            &query_projection,
            attention_head_dimension,
            attention_projection_head_dimension,
        )?;

        // Later attention output has all head features beside each other, so
        // flatten the gate from [heads, features_per_head] into that same shape.
        let output_gate = self.runtime.reshape(
            &output_gate,
            &[1, token_count, query_head_count * attention_head_dimension],
        )?;

        // Second learned projection: K describes what each token offers for
        // matching. There are fewer K heads because they are shared by groups
        // of Q heads in grouped-query attention.
        let keys = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &full_attention_weights.key_projection,
            paged_prefill_execution_mode,
        )?;

        // K becomes [batch, tokens, key_value_heads, features_per_head].
        let keys = self.runtime.reshape(
            &keys,
            &[
                1,
                token_count,
                key_value_head_count,
                attention_head_dimension,
            ],
        )?;

        // Third learned projection: V carries the information that will be
        // averaged together after softmax decides how relevant each key is.
        let values = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &full_attention_weights.value_projection,
            paged_prefill_execution_mode,
        )?;

        // V uses the same head layout as K so each key position has one matching
        // value position in the KV state.
        let values = self.runtime.reshape(
            &values,
            &[
                1,
                token_count,
                key_value_head_count,
                attention_head_dimension,
            ],
        )?;

        // RoPE and RMS-norm are applied to the new keys before concatenation,
        // exactly as the model expects. The owner then appends these rotated
        // tensors to the cached slab and returns views over the full prefix.
        // RMS normalization rescales each Q vector before it participates in a
        // dot product. It is a model-defined preparation step, not softmax.
        let normalized_queries = self.runtime.rms_norm(
            &queries,
            &full_attention_weights.query_normalization_weight,
            rms_norm_epsilon,
        )?;

        // The model applies a corresponding normalization to each K vector.
        let normalized_keys = self.runtime.rms_norm(
            &keys,
            &full_attention_weights.key_normalization_weight,
            rms_norm_epsilon,
        )?;

        // Linear projections naturally use [batch, tokens, heads, features].
        // MLX attention uses [batch, heads, tokens, features], so move the head
        // axis before the token axis for all three attention inputs.
        let transposed_queries = self
            .runtime
            .transpose_axes(&normalized_queries, &[0, 2, 1, 3])?;
        let transposed_keys = self
            .runtime
            .transpose_axes(&normalized_keys, &[0, 2, 1, 3])?;
        let transposed_values = self.runtime.transpose_axes(&values, &[0, 2, 1, 3])?;

        // Rotary positional embedding, or RoPE, rotates pairs of Q and K
        // features according to each token's position. Their dot product can
        // then reflect relative position as well as feature similarity. V is
        // intentionally not rotated. rope_offset_tokens tells RoPE where this
        // prompt chunk or generated token starts in the existing context.
        let rope_base = f32::from_bits(self.config.rope_theta_bits());
        let (rotated_queries, rotated_keys) =
            if let Some(token_position_offsets) = token_position_offsets {
                (
                    self.runtime.rope_with_token_position_offsets(
                        &transposed_queries,
                        token_position_offsets,
                        rotary_dimension,
                        rope_base,
                    )?,
                    self.runtime.rope_with_token_position_offsets(
                        &transposed_keys,
                        token_position_offsets,
                        rotary_dimension,
                        rope_base,
                    )?,
                )
            } else {
                (
                    self.runtime.rope(
                        &transposed_queries,
                        rotary_dimension,
                        rope_base,
                        rope_offset_tokens,
                    )?,
                    self.runtime.rope(
                        &transposed_keys,
                        rotary_dimension,
                        rope_base,
                        rope_offset_tokens,
                    )?,
                )
            };

        // Store only the new K and V tensors in the append-only KV state. The
        // owner returns views over the complete used prefix, including this
        // forward's new positions. It grows storage in fixed steps to avoid
        // repeatedly copying all prior context for each generated token.
        let previous_storage_offset_tokens = kv_state.offset_tokens();
        let (active_keys, active_values) = kv_state.update_and_fetch(
            &self.runtime,
            &rotated_keys,
            &transposed_values,
            previous_storage_offset_tokens,
        )?;
        if let Some(attention_capture) = attention_capture {
            attention_capture.record_full_attention_tensors(
                decoder_layer_index,
                &rotated_queries,
                &active_keys,
            )?;
        }

        // Execute the foundational attention formula against the active prefix.
        // 1 / sqrt(features_per_head) is the conventional score scale. The
        // helper uses causal mode for a multi-token prompt and unmasked mode for
        // one-token decode, then applies Qwen's learned output gate.
        let gated_output = qwen3_5_full_attention_step(
            &self.runtime,
            &self.compiled_elementwise_graphs,
            &rotated_queries,
            &active_keys,
            &active_values,
            &output_gate,
            (attention_head_dimension as f32).sqrt().recip(),
            paged_prefill_execution_mode,
        )?;

        // Final learned projection: mix the concatenated head outputs back into
        // the decoder's hidden-size space. The caller adds this result to the
        // residual stream before continuing with the rest of the layer.
        self.quantized_linear_for_paged_prefill_execution_mode(
            &gated_output,
            &full_attention_weights.output_projection,
            paged_prefill_execution_mode,
        )
    }
}

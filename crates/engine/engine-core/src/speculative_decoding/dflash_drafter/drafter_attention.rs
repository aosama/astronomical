//! One drafter layer's attention: block rows attend to the kept context of
//! fused target taps and to each other.
//!
//! Context keys are rotated at their own absolute positions as they arrive,
//! then kept; the block's queries and keys sit just after the newest context
//! row. Both follow z-lab's `DFlashAttention`.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::block_attention_mask::block_attention_visibility;
use super::drafter_configuration::DFlashDrafterConfiguration;
use super::drafter_context_state::DrafterLayerContextState;
use super::drafter_linear::DrafterLinear;
use super::drafter_weights::DrafterLayerWeights;

/// The rows one layer's attention reads this round.
pub struct BlockAttentionRows<'rows> {
    /// [1, L, H]: the block, already normalized (and convolved, for DFlash2).
    pub block_rows: &'rows MlxArray,
    /// [1, S, H]: this round's fused target context.
    pub fused_context_rows: &'rows MlxArray,
    /// Whether block rows see only earlier block rows in this layer.
    pub block_is_causal: bool,
}

/// Attends the block rows to the layer's kept context plus this round's
/// fused context rows.
pub fn attend_block_to_context(
    runtime: &MlxRuntime,
    layer: &DrafterLayerWeights,
    configuration: &DFlashDrafterConfiguration,
    context_state: &mut DrafterLayerContextState,
    attention_rows: BlockAttentionRows<'_>,
) -> Result<MlxArray, MlxRuntimeError> {
    let BlockAttentionRows {
        block_rows,
        fused_context_rows,
        block_is_causal,
    } = attention_rows;
    let block_row_count = block_rows.shape()[1];
    let incoming_row_count = fused_context_rows.shape()[1];
    let skipped_row_count = context_state.skip_rows_beyond_window(incoming_row_count);
    let new_context_rows = runtime.slice(
        fused_context_rows,
        &[0, skipped_row_count, 0],
        &[1, incoming_row_count, configuration.hidden_size],
        &[1, 1, 1],
    )?;
    let new_context_row_count = incoming_row_count - skipped_row_count;
    let context_position = context_state.next_context_position();
    let block_position = context_position + new_context_row_count;

    let head_shaper = HeadShaper {
        runtime,
        configuration,
    };
    let queries = head_shaper.rotated_heads(
        &layer.query_projection,
        &layer.query_head_normalization,
        block_rows,
        configuration.attention_head_count,
        block_position,
    )?;
    let block_keys = head_shaper.rotated_heads(
        &layer.key_projection,
        &layer.key_head_normalization,
        block_rows,
        configuration.key_value_head_count,
        block_position,
    )?;
    let block_values = head_shaper.value_heads(&layer.value_projection, block_rows)?;
    let context_keys = head_shaper.rotated_heads(
        &layer.key_projection,
        &layer.key_head_normalization,
        &new_context_rows,
        configuration.key_value_head_count,
        context_position,
    )?;
    let context_values = head_shaper.value_heads(&layer.value_projection, &new_context_rows)?;
    let attended_rows = context_state.attended_rows(
        runtime,
        (&context_keys, &context_values),
        (&block_keys, &block_values),
    )?;
    let kept_row_count = attended_rows.kept_row_count;
    let (keys, values) = (attended_rows.keys, attended_rows.values);
    let mask = additive_mask(
        runtime,
        configuration,
        context_state,
        MaskRows {
            kept_row_count,
            block_row_count,
            block_is_causal,
        },
        &queries,
    )?;
    let scale = (configuration.head_dimension as f32).powf(-0.5);
    let attended =
        runtime.masked_scaled_dot_product_attention(&queries, &keys, &values, scale, &mask)?;
    let merged_heads = runtime.reshape(
        &runtime.transpose_axes(&attended, &[0, 2, 1, 3])?,
        &[
            1,
            block_row_count,
            configuration.attention_head_count * configuration.head_dimension,
        ],
    )?;
    layer.output_projection.forward(runtime, &merged_heads)
}

/// The key rows a block mask covers, and whether the block is causal.
struct MaskRows {
    kept_row_count: i32,
    block_row_count: i32,
    block_is_causal: bool,
}

/// 0 where a block row may attend to a key, negative infinity where not, in
/// the queries' dtype so the fused attention keeps its low-precision path.
fn additive_mask(
    runtime: &MlxRuntime,
    configuration: &DFlashDrafterConfiguration,
    context_state: &DrafterLayerContextState,
    mask_rows: MaskRows,
    queries: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let MaskRows {
        kept_row_count,
        block_row_count,
        block_is_causal,
    } = mask_rows;
    let sliding_window = context_state
        .retained_row_limit()
        .map(|row_limit| row_limit + 1);
    let visibility = block_attention_visibility(
        kept_row_count,
        block_row_count,
        sliding_window,
        block_is_causal,
        configuration.block_size,
    );
    let mut additive_values = Vec::with_capacity(visibility.len());
    for is_visible in visibility {
        additive_values.push(if is_visible { 0.0 } else { f32::NEG_INFINITY });
    }
    let mask = runtime.array_from_f32(
        &additive_values,
        &[block_row_count, kept_row_count + block_row_count],
    )?;
    Ok(runtime.astype(&mask, queries.dtype())?)
}

/// Projects rows to attention heads: `[1, rows, H]` → `[1, heads, rows, head_dim]`.
struct HeadShaper<'shaping> {
    runtime: &'shaping MlxRuntime,
    configuration: &'shaping DFlashDrafterConfiguration,
}

impl HeadShaper<'_> {
    /// Projection, per-head RMS norm, then RoPE from `first_position` on.
    fn rotated_heads(
        &self,
        projection: &DrafterLinear,
        head_normalization: &MlxArray,
        rows: &MlxArray,
        head_count: i32,
        first_position: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let row_count = rows.shape()[1];
        let projected = self.runtime.reshape(
            &projection.forward(self.runtime, rows)?,
            &[1, row_count, head_count, self.configuration.head_dimension],
        )?;
        let normalized = self.runtime.rms_norm(
            &projected,
            head_normalization,
            self.configuration.rms_norm_epsilon,
        )?;
        let head_major = self.runtime.transpose_axes(&normalized, &[0, 2, 1, 3])?;
        Ok(self.runtime.rope(
            &head_major,
            self.configuration.head_dimension,
            self.configuration.rope_theta,
            first_position,
        )?)
    }

    fn value_heads(
        &self,
        projection: &DrafterLinear,
        rows: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let row_count = rows.shape()[1];
        let projected = self.runtime.reshape(
            &projection.forward(self.runtime, rows)?,
            &[
                1,
                row_count,
                self.configuration.key_value_head_count,
                self.configuration.head_dimension,
            ],
        )?;
        Ok(self.runtime.transpose_axes(&projected, &[0, 2, 1, 3])?)
    }
}

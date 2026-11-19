//! One drafter layer's attention: block rows attend to the kept context of
//! fused target taps and to each other.
//!
//! Context keys are rotated at their own absolute positions as they arrive,
//! then kept; the block's queries and keys sit just after the newest context
//! row. Both follow z-lab's `DFlashAttention`.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::block_attention_mask::block_attention_visibility;
use super::drafter_configuration::DFlash2DrafterConfiguration;
use super::drafter_context_state::DrafterLayerContextState;
use super::drafter_linear::DrafterLinear;
use super::drafter_weights::DrafterLayerWeights;

/// Attends `block_rows` [1, L, H] (already normalized and convolved) to the
/// layer's kept context plus this round's `fused_context_rows` [1, S, H].
pub fn attend_block_to_context(
    runtime: &MlxRuntime,
    layer: &DrafterLayerWeights,
    configuration: &DFlash2DrafterConfiguration,
    context_state: &mut DrafterLayerContextState,
    block_rows: &MlxArray,
    fused_context_rows: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
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
    let (kept_keys, kept_values) =
        context_state.append_and_fetch(runtime, context_keys, context_values)?;

    let kept_row_count = kept_keys.shape()[2];
    let keys = runtime.concatenate_axis(&[&kept_keys, &block_keys], 2)?;
    let values = runtime.concatenate_axis(&[&kept_values, &block_values], 2)?;
    let mask = additive_mask(
        runtime,
        configuration,
        context_state,
        kept_row_count,
        block_row_count,
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

/// 0 where a block row may attend to a key, negative infinity where not, in
/// the queries' dtype so the fused attention keeps its low-precision path.
fn additive_mask(
    runtime: &MlxRuntime,
    configuration: &DFlash2DrafterConfiguration,
    context_state: &DrafterLayerContextState,
    kept_row_count: i32,
    block_row_count: i32,
    queries: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let sliding_window = context_state
        .retained_row_limit()
        .map(|row_limit| row_limit + 1);
    let visibility = block_attention_visibility(
        kept_row_count,
        block_row_count,
        sliding_window,
        configuration.block_is_causal,
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
    configuration: &'shaping DFlash2DrafterConfiguration,
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

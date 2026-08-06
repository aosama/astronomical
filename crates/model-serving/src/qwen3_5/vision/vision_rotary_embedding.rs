//! Two-dimensional rotary-position graph for Qwen3.5 vision attention.
//!
//! Source lineage: Rust translation of MLX-VLM's Qwen3-VL vision rotary-position
//! routines (MIT License; see third-party license notices). Trigonometric tensor
//! work is delegated to MLX-C `mlx_cos` and `mlx_sin` from `mlx-c/mlx/c/ops.h`.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::{Qwen3_5ExecutionError, Qwen3_5VisionConfig, Qwen3_5VisionInputPlan};

const VISION_ROTARY_FREQUENCY_BASE: f32 = 10_000.0;

/// Builds the config-derived Qwen3.5 two-dimensional vision rotary embedding graph.
///
/// The reference builds a complete spatial frequency table, gathers its rows by
/// patch coordinate, then duplicates cosine and sine across the two rotate-half
/// segments. The resulting arrays have shape `[patch_count, 1, head_dimension]`.
pub(super) struct Qwen3_5VisionRotaryEmbedding;

impl Qwen3_5VisionRotaryEmbedding {
    pub(super) fn build(
        runtime: &MlxRuntime,
        vision_input_plan: &Qwen3_5VisionInputPlan,
        vision_config: &Qwen3_5VisionConfig,
    ) -> Result<(MlxArray, MlxArray), Qwen3_5ExecutionError> {
        let head_dimension = vision_config.hidden_size() / vision_config.head_count();
        let per_axis_rotary_dimension = head_dimension / 2;
        let patch_count = usize_to_i32(vision_input_plan.patch_count())?;
        // Coordinates are already in the same spatial-merge block order as the
        // patch tensor. Reordering either side rotates the wrong patch position.
        let rotary_position_coordinates = vision_input_plan.rotary_position_coordinates();
        let row_coordinates = rotary_position_coordinates
            .iter()
            .map(|position_coordinates| position_coordinates[0])
            .collect::<Vec<_>>();
        let column_coordinates = rotary_position_coordinates
            .iter()
            .map(|position_coordinates| position_coordinates[1])
            .collect::<Vec<_>>();
        let maximum_spatial_position_count = rotary_position_coordinates
            .iter()
            .flat_map(|position_coordinates| position_coordinates.iter().copied())
            .max()
            .and_then(|maximum_spatial_coordinate| maximum_spatial_coordinate.checked_add(1))
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "vision rotary coordinates must be nonempty and fit the MLX range",
            })?;
        let row_coordinate_indices = runtime.array_from_u32(&row_coordinates, &[patch_count])?;
        let column_coordinate_indices =
            runtime.array_from_u32(&column_coordinates, &[patch_count])?;

        // Keep phase arithmetic in Float32 even when attention activations are
        // BF16. The reference constructs both positions and inverse frequencies
        // as Float32, then casts only the final rotated attention output.
        let inverse_frequency_count = per_axis_rotary_dimension / 2;
        let inverse_frequency_positions =
            runtime.arange_f32(0.0, f64::from(per_axis_rotary_dimension), 2.0)?;
        let per_axis_rotary_dimension_scalar =
            runtime.array_from_f32(&[per_axis_rotary_dimension as f32], &[])?;
        let inverse_frequency_exponents = runtime.divide(
            &inverse_frequency_positions,
            &per_axis_rotary_dimension_scalar,
        )?;
        let frequency_base = runtime.array_from_f32(&[VISION_ROTARY_FREQUENCY_BASE], &[])?;
        let frequency_denominators =
            runtime.power(&frequency_base, &inverse_frequency_exponents)?;
        let inverse_frequency_numerator = runtime.array_from_f32(&[1.0], &[])?;
        let inverse_frequencies =
            runtime.divide(&inverse_frequency_numerator, &frequency_denominators)?;
        let row_shaped_inverse_frequencies = runtime.reshape(
            &inverse_frequencies,
            &[1, u32_to_i32(inverse_frequency_count)?],
        )?;
        let spatial_position_count = u32_to_i32(maximum_spatial_position_count)?;
        let spatial_positions = runtime.arange_f32(0.0, f64::from(spatial_position_count), 1.0)?;
        let column_shaped_spatial_positions =
            runtime.reshape(&spatial_positions, &[spatial_position_count, 1])?;
        let spatial_frequency_table = runtime.multiply(
            &column_shaped_spatial_positions,
            &row_shaped_inverse_frequencies,
        )?;
        let row_frequencies =
            runtime.take_axis(&spatial_frequency_table, &row_coordinate_indices, 0)?;
        let column_frequencies =
            runtime.take_axis(&spatial_frequency_table, &column_coordinate_indices, 0)?;
        let half_rotary_frequencies =
            runtime.concatenate_axis(&[&row_frequencies, &column_frequencies], 1)?;
        let half_rotary_cosines = runtime.cos(&half_rotary_frequencies)?;
        let half_rotary_sines = runtime.sin(&half_rotary_frequencies)?;
        let rotary_cosines =
            runtime.concatenate_axis(&[&half_rotary_cosines, &half_rotary_cosines], 1)?;
        let rotary_sines =
            runtime.concatenate_axis(&[&half_rotary_sines, &half_rotary_sines], 1)?;
        Ok((
            runtime.expand_dims(&rotary_cosines, 1)?,
            runtime.expand_dims(&rotary_sines, 1)?,
        ))
    }
}

fn usize_to_i32(dimension_size: usize) -> Result<i32, Qwen3_5ExecutionError> {
    i32::try_from(dimension_size).map_err(|_conversion_error| Qwen3_5ExecutionError::InvalidInput {
        description: "vision dimension exceeds the MLX integer range",
    })
}

fn u32_to_i32(dimension_size: u32) -> Result<i32, Qwen3_5ExecutionError> {
    i32::try_from(dimension_size).map_err(|_conversion_error| Qwen3_5ExecutionError::InvalidInput {
        description: "vision dimension exceeds the MLX integer range",
    })
}

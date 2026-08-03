//! Two-dimensional rotary-position graph for Qwen3.5 vision attention.
//!
//! Source lineage: Rust translation of MLX-VLM's Qwen3-VL vision rotary-position
//! routines (MIT License; see third-party license notices). Trigonometric tensor
//! work is delegated to MLX-C `mlx_cos` and `mlx_sin` from `mlx-c/mlx/c/ops.h`.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use super::{Qwen3_5ExecutionError, Qwen3_5VisionConfig, Qwen3_5VisionInputPlan};

// These are the Float32 results of
// `1 / 10000 ** (arange(0, 36, 2, float32) / 36)` produced by MLX 0.32.0.
// The per-axis rotary dimension is 36 because head_dimension=72 and height and
// width each consume half. Do not regenerate this table with Rust `powf`: tiny
// host-math differences propagate through sin/cos and measurably reduce parity.
const VISION_ROTARY_INVERSE_FREQUENCIES: [f32; 18] = [
    1.0,
    0.599_484_2,
    0.359_381_35,
    0.215_443_45,
    0.129_154_97,
    0.077_426_36,
    0.046_415_884,
    0.027_825_59,
    0.016_681_006,
    0.01,
    0.005_994_840_5,
    0.003_593_813_6,
    0.002_154_434,
    0.001_291_549_8,
    0.000_774_263_5,
    0.000_464_158_95,
    0.000_278_255_93,
    0.000_166_810_09,
];

/// Builds the fixed Qwen3.5 two-dimensional vision rotary embedding graph.
///
/// For patch coordinate `(row, column)` and inverse-frequency vector `f`, the
/// phase is `[row*f, column*f]`. Duplicating that half vector produces the full
/// head-width phase required by `rotate_half`, resulting in cosine and sine
/// arrays shaped `[patch_count, 1, head_dimension]` for broadcasting over heads.
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
        let flattened_position_coordinates = vision_input_plan
            .rotary_position_coordinates()
            .iter()
            .flat_map(|position_coordinates| position_coordinates.iter().copied())
            .collect::<Vec<_>>();
        let position_coordinates_u32 =
            runtime.array_from_u32(&flattened_position_coordinates, &[patch_count, 2])?;
        let position_coordinates = runtime.astype(&position_coordinates_u32, MlxDtype::Float32)?;
        let row_coordinates =
            runtime.slice(&position_coordinates, &[0, 0], &[patch_count, 1], &[1, 1])?;
        let column_coordinates =
            runtime.slice(&position_coordinates, &[0, 1], &[patch_count, 2], &[1, 1])?;
        // Keep phase arithmetic in Float32 even when attention activations are
        // BF16. The reference constructs both positions and inverse frequencies
        // as Float32, then casts only the final rotated attention output.
        let inverse_frequencies = runtime.array_from_f32(
            &VISION_ROTARY_INVERSE_FREQUENCIES,
            &[1, u32_to_i32(per_axis_rotary_dimension / 2)?],
        )?;
        let row_frequencies = runtime.multiply(&row_coordinates, &inverse_frequencies)?;
        let column_frequencies = runtime.multiply(&column_coordinates, &inverse_frequencies)?;
        // Height owns the first 18 frequencies and width owns the next 18.
        // Duplicating [height, width] yields 72 entries so both halves used by
        // rotate_half receive identical phases.
        let half_rotary_frequencies =
            runtime.concatenate_axis(&[&row_frequencies, &column_frequencies], 1)?;
        let rotary_frequencies =
            runtime.concatenate_axis(&[&half_rotary_frequencies, &half_rotary_frequencies], 1)?;
        let rotary_frequencies = runtime.expand_dims(&rotary_frequencies, 1)?;
        Ok((
            runtime.cos(&rotary_frequencies)?,
            runtime.sin(&rotary_frequencies)?,
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

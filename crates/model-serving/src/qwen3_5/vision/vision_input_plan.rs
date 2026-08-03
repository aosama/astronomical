//! Host-side geometry plan for one Qwen3.5 vision forward pass.
//!
//! This is intentionally CPU arithmetic: it computes small integer indices and
//! interpolation coefficients once per image, then MLX performs all large tensor
//! gathers and arithmetic. Source lineage: Rust translation of the MLX-VLM
//! Qwen3-VL rotary-position and position-interpolation routines (MIT License;
//! see third-party license notices).
//!
//! The critical invariant is *block-major patch order*:
//! `[merged_row, merged_column, intra_merge_row, intra_merge_column]`. The image
//! processor, absolute positions, rotary positions, and merger reshape must all
//! use this exact order or valid tensors will describe different patches.

use thiserror::Error;

use super::{Qwen3_5ImageGrid, Qwen3_5VisionConfig};

const BILINEAR_CORNER_COUNT: usize = 4;

/// CPU-computed, image-dependent coordinates consumed by the Qwen3.5 vision tower.
///
/// `attention_sequence_boundaries` stores cumulative per-segment patch counts.
/// The four corner vectors are a structure-of-arrays
/// representation that lets MLX gather each corner for every patch in one call.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3_5VisionInputPlan {
    patch_count: usize,
    merged_patch_count: usize,
    attention_sequence_boundaries: Vec<u32>,
    rotary_position_coordinates: Vec<[u32; 2]>,
    bilinear_corner_indices: [Vec<u32>; BILINEAR_CORNER_COUNT],
    bilinear_corner_weights: [Vec<f32>; BILINEAR_CORNER_COUNT],
}

impl Qwen3_5VisionInputPlan {
    /// Builds the data-dependent position plan used by the pinned vision architecture.
    pub fn new(
        image_grids: &[Qwen3_5ImageGrid],
        vision_config: &Qwen3_5VisionConfig,
    ) -> Result<Self, Qwen3_5VisionInputPlanError> {
        if image_grids.is_empty() {
            return Err(Qwen3_5VisionInputPlanError::MissingImageGrids);
        }

        let spatial_merge_size = vision_config.spatial_merge_size();
        // The checkpoint stores one flattened learned square position table.
        // Taking an exact square root recovers the table's row/column side.
        let position_embedding_grid_side =
            exact_square_side(vision_config.position_embedding_count())?;
        let mut attention_sequence_boundaries = vec![0_u32];
        let mut rotary_position_coordinates = Vec::new();
        let mut bilinear_corner_indices = std::array::from_fn(|_| Vec::new());
        let mut bilinear_corner_weights = std::array::from_fn(|_| Vec::new());
        let mut patch_count = 0_usize;

        for image_grid in image_grids {
            validate_image_grid(*image_grid, spatial_merge_size)?;
            append_image_grid_plan(
                *image_grid,
                spatial_merge_size,
                position_embedding_grid_side,
                &mut patch_count,
                &mut attention_sequence_boundaries,
                &mut rotary_position_coordinates,
                &mut bilinear_corner_indices,
                &mut bilinear_corner_weights,
            )?;
        }

        // Every merge_size x merge_size patch block becomes one text-width visual
        // token. Grid validation above guarantees exact divisibility.
        let spatial_merge_area = usize::try_from(
            spatial_merge_size
                .checked_mul(spatial_merge_size)
                .ok_or(Qwen3_5VisionInputPlanError::DimensionOverflow)?,
        )
        .map_err(|_conversion_error| Qwen3_5VisionInputPlanError::DimensionOverflow)?;

        Ok(Self {
            patch_count,
            merged_patch_count: patch_count / spatial_merge_area,
            attention_sequence_boundaries,
            rotary_position_coordinates,
            bilinear_corner_indices,
            bilinear_corner_weights,
        })
    }

    /// Returns the number of unmerged patch rows expected by patch embedding.
    #[must_use]
    pub const fn patch_count(&self) -> usize {
        self.patch_count
    }

    /// Returns the number of visual embeddings produced after spatial merging.
    #[must_use]
    pub const fn merged_patch_count(&self) -> usize {
        self.merged_patch_count
    }

    /// Returns cumulative per-image-frame patch boundaries for segmented attention.
    #[must_use]
    pub fn attention_sequence_boundaries(&self) -> &[u32] {
        &self.attention_sequence_boundaries
    }

    /// Returns block-major height and width coordinates for vision rotary embedding.
    #[must_use]
    pub fn rotary_position_coordinates(&self) -> &[[u32; 2]] {
        &self.rotary_position_coordinates
    }

    /// Returns four block-major lookup-index rows into the absolute-position table.
    #[must_use]
    pub fn bilinear_corner_indices(&self) -> &[Vec<u32>; BILINEAR_CORNER_COUNT] {
        &self.bilinear_corner_indices
    }

    /// Returns four interpolation-weight rows corresponding to the corner indices.
    #[must_use]
    pub fn bilinear_corner_weights(&self) -> &[Vec<f32>; BILINEAR_CORNER_COUNT] {
        &self.bilinear_corner_weights
    }
}

/// Invalid dynamic image geometry for the pinned Qwen3.5 vision architecture.
#[derive(Debug, Error, Clone, Copy, Eq, PartialEq)]
pub enum Qwen3_5VisionInputPlanError {
    #[error("at least one image grid is required for Qwen3.5 vision execution")]
    MissingImageGrids,
    #[error("image grid dimensions must all be positive")]
    NonPositiveImageGridDimension,
    #[error("image grid height and width must be divisible by the spatial merge size")]
    ImageGridNotDivisibleBySpatialMergeSize,
    #[error("vision position embedding count must be a perfect square")]
    PositionEmbeddingCountNotSquare,
    #[error("vision input dimensions exceed the supported integer range")]
    DimensionOverflow,
}

#[allow(clippy::too_many_arguments)]
fn append_image_grid_plan(
    image_grid: Qwen3_5ImageGrid,
    spatial_merge_size: u32,
    position_embedding_grid_side: u32,
    patch_count: &mut usize,
    attention_sequence_boundaries: &mut Vec<u32>,
    rotary_position_coordinates: &mut Vec<[u32; 2]>,
    bilinear_corner_indices: &mut [Vec<u32>; BILINEAR_CORNER_COUNT],
    bilinear_corner_weights: &mut [Vec<f32>; BILINEAR_CORNER_COUNT],
) -> Result<(), Qwen3_5VisionInputPlanError> {
    let patches_per_temporal_frame = image_grid
        .height_patch_count
        .checked_mul(image_grid.width_patch_count)
        .ok_or(Qwen3_5VisionInputPlanError::DimensionOverflow)?;

    // Attention is segmented per temporal frame, matching upstream `cu_seqlens`.
    // A still image has one frame; video-like inputs would append one boundary
    // for each temporal patch group and prevent cross-frame attention here.
    for _temporal_patch_index in 0..image_grid.temporal_patch_count {
        let preceding_sequence_boundary = *attention_sequence_boundaries
            .last()
            .ok_or(Qwen3_5VisionInputPlanError::DimensionOverflow)?;
        attention_sequence_boundaries.push(
            preceding_sequence_boundary
                .checked_add(patches_per_temporal_frame)
                .ok_or(Qwen3_5VisionInputPlanError::DimensionOverflow)?,
        );

        append_spatial_position_plan(
            image_grid.height_patch_count,
            image_grid.width_patch_count,
            spatial_merge_size,
            position_embedding_grid_side,
            rotary_position_coordinates,
            bilinear_corner_indices,
            bilinear_corner_weights,
        )?;
    }

    let image_patch_count = usize::try_from(patches_per_temporal_frame)
        .ok()
        .and_then(|patches_per_temporal_frame| {
            patches_per_temporal_frame.checked_mul(image_grid.temporal_patch_count as usize)
        })
        .ok_or(Qwen3_5VisionInputPlanError::DimensionOverflow)?;
    *patch_count = patch_count
        .checked_add(image_patch_count)
        .ok_or(Qwen3_5VisionInputPlanError::DimensionOverflow)?;
    Ok(())
}

fn append_spatial_position_plan(
    height_patch_count: u32,
    width_patch_count: u32,
    spatial_merge_size: u32,
    position_embedding_grid_side: u32,
    rotary_position_coordinates: &mut Vec<[u32; 2]>,
    bilinear_corner_indices: &mut [Vec<u32>; BILINEAR_CORNER_COUNT],
    bilinear_corner_weights: &mut [Vec<f32>; BILINEAR_CORNER_COUNT],
) -> Result<(), Qwen3_5VisionInputPlanError> {
    // This nesting is not interchangeable with ordinary row-major traversal.
    // Keeping all intra-block patches adjacent makes the final merger a reshape
    // rather than an expensive gather, and matches upstream reshape+transpose.
    for merged_patch_row_index in 0..height_patch_count / spatial_merge_size {
        for merged_patch_column_index in 0..width_patch_count / spatial_merge_size {
            for intra_merge_row_index in 0..spatial_merge_size {
                for intra_merge_column_index in 0..spatial_merge_size {
                    let patch_row_index =
                        merged_patch_row_index * spatial_merge_size + intra_merge_row_index;
                    let patch_column_index =
                        merged_patch_column_index * spatial_merge_size + intra_merge_column_index;
                    rotary_position_coordinates.push([patch_row_index, patch_column_index]);
                    append_bilinear_position(
                        patch_row_index,
                        patch_column_index,
                        height_patch_count,
                        width_patch_count,
                        position_embedding_grid_side,
                        bilinear_corner_indices,
                        bilinear_corner_weights,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_bilinear_position(
    patch_row_index: u32,
    patch_column_index: u32,
    height_patch_count: u32,
    width_patch_count: u32,
    position_embedding_grid_side: u32,
    bilinear_corner_indices: &mut [Vec<u32>; BILINEAR_CORNER_COUNT],
    bilinear_corner_weights: &mut [Vec<f32>; BILINEAR_CORNER_COUNT],
) -> Result<(), Qwen3_5VisionInputPlanError> {
    // Map runtime patch centers onto the learned square table's endpoint-aligned
    // coordinate system. This is equivalent to MLX `linspace(0, side-1, count)`.
    let interpolated_row = interpolated_position(
        patch_row_index,
        height_patch_count,
        position_embedding_grid_side,
    );
    let interpolated_column = interpolated_position(
        patch_column_index,
        width_patch_count,
        position_embedding_grid_side,
    );
    let row_floor = interpolated_row.floor() as u32;
    let column_floor = interpolated_column.floor() as u32;
    let row_ceiling = (row_floor + 1).min(position_embedding_grid_side - 1);
    let column_ceiling = (column_floor + 1).min(position_embedding_grid_side - 1);
    let row_fraction = interpolated_row - row_floor as f32;
    let column_fraction = interpolated_column - column_floor as f32;

    // Corner order must stay [top-left, top-right, bottom-left, bottom-right].
    // `vision_model.rs` adds the weighted corners in this same left-associative
    // order to preserve the translated Qwen3-VL BF16 operation order.
    let corner_rows = [row_floor, row_floor, row_ceiling, row_ceiling];
    let corner_columns = [column_floor, column_ceiling, column_floor, column_ceiling];
    let corner_weights = [
        (1.0 - row_fraction) * (1.0 - column_fraction),
        (1.0 - row_fraction) * column_fraction,
        row_fraction * (1.0 - column_fraction),
        row_fraction * column_fraction,
    ];

    for corner_index in 0..BILINEAR_CORNER_COUNT {
        let flattened_position_index = corner_rows[corner_index]
            .checked_mul(position_embedding_grid_side)
            .and_then(|row_offset| row_offset.checked_add(corner_columns[corner_index]))
            .ok_or(Qwen3_5VisionInputPlanError::DimensionOverflow)?;
        bilinear_corner_indices[corner_index].push(flattened_position_index);
        bilinear_corner_weights[corner_index].push(corner_weights[corner_index]);
    }
    Ok(())
}

fn interpolated_position(
    patch_index: u32,
    patch_count: u32,
    position_embedding_grid_side: u32,
) -> f32 {
    if patch_count == 1 {
        return 0.0;
    }
    // Preserve divide-then-multiply ordering. `index / (count-1) * (side-1)` is
    // what MLX linspace produced for the parity reference; reassociating it to
    // `index * (side-1) / (count-1)` can round differently in Float32.
    (patch_index as f32 / (patch_count - 1) as f32) * (position_embedding_grid_side - 1) as f32
}

fn validate_image_grid(
    image_grid: Qwen3_5ImageGrid,
    spatial_merge_size: u32,
) -> Result<(), Qwen3_5VisionInputPlanError> {
    if image_grid.temporal_patch_count == 0
        || image_grid.height_patch_count == 0
        || image_grid.width_patch_count == 0
    {
        return Err(Qwen3_5VisionInputPlanError::NonPositiveImageGridDimension);
    }
    if !image_grid
        .height_patch_count
        .is_multiple_of(spatial_merge_size)
        || !image_grid
            .width_patch_count
            .is_multiple_of(spatial_merge_size)
    {
        return Err(Qwen3_5VisionInputPlanError::ImageGridNotDivisibleBySpatialMergeSize);
    }
    Ok(())
}

fn exact_square_side(position_embedding_count: u32) -> Result<u32, Qwen3_5VisionInputPlanError> {
    let candidate_grid_side = (position_embedding_count as f64).sqrt() as u32;
    if candidate_grid_side.checked_mul(candidate_grid_side) != Some(position_embedding_count) {
        return Err(Qwen3_5VisionInputPlanError::PositionEmbeddingCountNotSquare);
    }
    Ok(candidate_grid_side)
}

//! The blockwise 32-wide Walsh-Hadamard rotation used by jangtq2 bundles.
//!
//! The rotation is declared per module (`"rotation": "hadamard32"` in the
//! quantization override). Weights are stored as `W · Rᵀ` where `R` is the
//! blockwise normalized Walsh-Hadamard over 32-wide input blocks, and the
//! decode path applies `R` to activations instead (`y = W_stored · R x`):
//! the transform is its own inverse, so activating it on the input side is
//! equivalent to un-rotating every weight row — while leaving the stored
//! codes in the basis the encoder optimized them for.
//!
//! This module defines the transform and provides an exact CPU reference
//! implementation used by tests and verification tooling. The serving decode
//! applies it through MLX (`mx.hadamard_transform` over a
//! `(..., inputs/32, 32)` view) — the fused production path lives with the
//! decode integration, not here.
//!
//! `H[i][j] = (-1)^popcount(i & j) / sqrt(32)` — symmetric, orthonormal,
//! self-inverse. No random signs: v2 chose the signless blockwise transform
//! measured to fix the per-row error tails at unchanged byte cost.

/// The transform block width. Input widths must be a whole multiple of it.
pub const HADAMARD_BLOCK_WIDTH: usize = 32;

/// Returns whether the transform applies to an input width.
#[must_use]
pub const fn is_applicable_to_input_width(input_width: usize) -> bool {
    input_width % HADAMARD_BLOCK_WIDTH == 0
}

/// Applies the CPU reference transform to one vector in place over its
/// 32-wide blocks.
///
/// Computation follows the reference (`h32` in `jangh/format.py`): cast to
/// `f32`, transform each block, keep `f32`. The direct per-block dot product
/// is used deliberately — clarity over butterfly ordering. Reordered
/// floating-point summation can differ from MLX's butterfly by a few ulps,
/// which is why golden comparisons against the reference tolerate a small
/// relative epsilon instead of demanding bitwise equality.
///
/// Returns `None` when the input width is not a multiple of the block width.
pub fn reference_transform_vector(input_values: &[f32]) -> Option<Vec<f32>> {
    if !is_applicable_to_input_width(input_values.len()) {
        return None;
    }
    let hadamard_matrix = normalized_block_matrix();
    let mut transformed_values = vec![0_f32; input_values.len()];
    for (input_block, transformed_block) in input_values
        .chunks_exact(HADAMARD_BLOCK_WIDTH)
        .zip(transformed_values.chunks_exact_mut(HADAMARD_BLOCK_WIDTH))
    {
        for (row_index, transformed_slot) in transformed_block.iter_mut().enumerate() {
            let mut dot_product = 0_f32;
            for (column_index, &input_value) in input_block.iter().enumerate() {
                dot_product += hadamard_matrix[row_index][column_index] * input_value;
            }
            *transformed_slot = dot_product;
        }
    }
    Some(transformed_values)
}

/// Builds the normalized 32×32 Walsh-Hadamard matrix once per call; callers
/// in tests care about clarity, not allocation.
#[must_use]
fn normalized_block_matrix() -> [[f32; HADAMARD_BLOCK_WIDTH]; HADAMARD_BLOCK_WIDTH] {
    let inverse_sqrt_width = 1.0 / (HADAMARD_BLOCK_WIDTH as f32).sqrt();
    let mut matrix = [[0_f32; HADAMARD_BLOCK_WIDTH]; HADAMARD_BLOCK_WIDTH];
    for (row_index, row) in matrix.iter_mut().enumerate() {
        for (column_index, slot) in row.iter_mut().enumerate() {
            let sign_bit_count = (row_index & column_index).count_ones();
            *slot = if sign_bit_count % 2 == 0 {
                inverse_sqrt_width
            } else {
                -inverse_sqrt_width
            };
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_return_none_for_an_input_width_that_is_not_blockable() {
        assert_eq!(reference_transform_vector(&[0_f32; 30]), None);
        assert_eq!(reference_transform_vector(&[0_f32; 33]), None);
        assert!(!is_applicable_to_input_width(30));
        assert!(!is_applicable_to_input_width(33));
    }

    #[test]
    fn should_transform_one_block_to_its_scaled_first_axis_sum() {
        // The all-ones vector under a normalized Hadamard maps to its norm
        // concentrated on the first axis: a 32-length ones vector has norm
        // sqrt(32), so the transform is exactly [sqrt(32), 0, ...].
        let ones_vector = [1_f32; HADAMARD_BLOCK_WIDTH];
        let transformed_vector =
            reference_transform_vector(&ones_vector).expect("one full block is blockable");
        let expected_first_axis_value = (HADAMARD_BLOCK_WIDTH as f32).sqrt();
        assert!((transformed_vector[0] - expected_first_axis_value).abs() < 1e-6);
        assert!(
            transformed_vector[1..]
                .iter()
                .all(|&transformed_value| transformed_value.abs() < 1e-6)
        );
    }
}

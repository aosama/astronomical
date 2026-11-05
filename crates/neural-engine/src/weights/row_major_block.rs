//! Column blocks of row-major matrices.
//!
//! A contraction split gives each part the matching input-channel columns of
//! every weight row, and the same cut applies to fp16 weights, 4-bit codes,
//! and per-group scales alike, so the slicing lives here once.

/// Columns `first_column .. first_column + block_width` of every row of a
/// row-major matrix whose rows are `row_width` wide.
#[must_use]
pub fn row_block<T: Copy>(
    row_major_values: &[T],
    row_width: usize,
    first_column: usize,
    block_width: usize,
) -> Vec<T> {
    assert!(
        first_column + block_width <= row_width,
        "the block must lie inside the row"
    );
    // `chunks_exact` would silently drop a ragged final row.
    assert!(
        row_width > 0 && row_major_values.len().is_multiple_of(row_width),
        "the values must form whole rows"
    );
    let mut block = Vec::with_capacity(row_major_values.len() / row_width * block_width);
    for row in row_major_values.chunks_exact(row_width) {
        block.extend_from_slice(&row[first_column..first_column + block_width]);
    }
    block
}

#[cfg(test)]
#[path = "row_major_block_tests.rs"]
mod tests;

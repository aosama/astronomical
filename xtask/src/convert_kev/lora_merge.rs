//! Folds one LoRA module into its base weight: `W + scale · B · A`.
//!
//! The arithmetic is done in float32, as PEFT's own merge does on a float32
//! model, and the result is rounded once to the base's storage type. The
//! rows of `W` are independent, so they are split across every core; at
//! Kev-9B's size the merge is about a hundred billion multiply-adds.

use std::thread;

use crate::model_files::bfloat16::{bfloat16_bits_to_f32, f32_to_bfloat16_bits};
use crate::model_files::safetensors_file::TensorElementType;

/// The shapes of one merge: `W` is `[output_count, input_count]`, `A` is
/// `[rank, input_count]` and `B` is `[output_count, rank]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MergeShape {
    pub(crate) output_count: usize,
    pub(crate) input_count: usize,
    pub(crate) rank: usize,
}

/// One merge's inputs: the base weight's bytes, both factors as float32
/// values, and the adapter's scale.
pub(crate) struct LoraMergeInputs<'merge> {
    pub(crate) base_weight_bytes: &'merge [u8],
    pub(crate) base_element_type: TensorElementType,
    pub(crate) down_projection: &'merge [f32],
    pub(crate) up_projection: &'merge [f32],
    pub(crate) scale: f32,
    pub(crate) merge_shape: MergeShape,
}

/// The merged weight's bytes, in the base weight's storage type.
pub(crate) fn merged_weight_bytes(merge_inputs: &LoraMergeInputs<'_>) -> Result<Vec<u8>, String> {
    let merge_shape = merge_inputs.merge_shape;
    let bytes_per_element = match merge_inputs.base_element_type {
        TensorElementType::BFloat16 => 2,
        TensorElementType::Float32 => 4,
        _ => return Err("only bfloat16 and float32 base weights are merged".to_owned()),
    };
    if merge_inputs.base_weight_bytes.len()
        != merge_shape.output_count * merge_shape.input_count * bytes_per_element
        || merge_inputs.down_projection.len() != merge_shape.rank * merge_shape.input_count
        || merge_inputs.up_projection.len() != merge_shape.output_count * merge_shape.rank
    {
        return Err("the base weight and the adapter's factors do not fit together".to_owned());
    }
    let row_bytes = merge_shape.input_count * bytes_per_element;
    let mut merged_bytes = vec![0_u8; merge_inputs.base_weight_bytes.len()];
    let worker_count = thread::available_parallelism().map_or(1, usize::from);
    let rows_per_worker = merge_shape.output_count.div_ceil(worker_count).max(1);
    thread::scope(|merge_scope| {
        for (worker_index, merged_rows) in merged_bytes
            .chunks_mut(rows_per_worker * row_bytes)
            .enumerate()
        {
            let first_row = worker_index * rows_per_worker;
            merge_scope.spawn(move || merge_rows(merge_inputs, first_row, merged_rows));
        }
    });
    Ok(merged_bytes)
}

/// Merges the rows starting at `first_row` into `merged_rows`.
fn merge_rows(merge_inputs: &LoraMergeInputs<'_>, first_row: usize, merged_rows: &mut [u8]) {
    let merge_shape = merge_inputs.merge_shape;
    let is_bfloat16 = merge_inputs.base_element_type == TensorElementType::BFloat16;
    let bytes_per_element = if is_bfloat16 { 2 } else { 4 };
    let row_bytes = merge_shape.input_count * bytes_per_element;
    let mut delta_row = vec![0.0_f32; merge_shape.input_count];
    for (row_offset, merged_row) in merged_rows.chunks_mut(row_bytes).enumerate() {
        let output_row = first_row + row_offset;
        delta_row.fill(0.0);
        // Rank-major accumulation keeps the inner loop a contiguous axpy
        // over `A`'s row, which the compiler vectorizes; the reverse order
        // would stride across `A` and be several times slower.
        for rank_index in 0..merge_shape.rank {
            let up_value = merge_inputs.up_projection[output_row * merge_shape.rank + rank_index];
            let down_row = &merge_inputs.down_projection
                [rank_index * merge_shape.input_count..(rank_index + 1) * merge_shape.input_count];
            for (delta_value, down_value) in delta_row.iter_mut().zip(down_row) {
                *delta_value += up_value * down_value;
            }
        }
        let base_row =
            &merge_inputs.base_weight_bytes[output_row * row_bytes..(output_row + 1) * row_bytes];
        for (input_index, delta_value) in delta_row.iter().enumerate() {
            let element_start = input_index * bytes_per_element;
            let base_value = if is_bfloat16 {
                bfloat16_bits_to_f32(u16::from_le_bytes([
                    base_row[element_start],
                    base_row[element_start + 1],
                ]))
            } else {
                f32::from_le_bytes(
                    base_row[element_start..element_start + 4]
                        .try_into()
                        .unwrap_or([0; 4]),
                )
            };
            let merged_value = base_value + merge_inputs.scale * delta_value;
            if is_bfloat16 {
                merged_row[element_start..element_start + 2]
                    .copy_from_slice(&f32_to_bfloat16_bits(merged_value).to_le_bytes());
            } else {
                merged_row[element_start..element_start + 4]
                    .copy_from_slice(&merged_value.to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
#[path = "lora_merge_tests.rs"]
mod tests;

//! Writes one streamed expert row into a retained slot table.
//!
//! `slice_update` donates the destination buffer when the row is uniquely held,
//! so only that expert row is copied instead of the whole packed page.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_moe::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;

/// Convenience trait for cloning expert weights without importing the runtime.
pub(super) trait RetainedReferenceOk {
    fn retained_reference_ok(&self) -> Self;
}

impl RetainedReferenceOk for Qwen3_5PagedExpertWeights {
    fn retained_reference_ok(&self) -> Self {
        Qwen3_5PagedExpertWeights {
            gate_projection: self.gate_projection.retained_reference().unwrap(),
            up_projection: self.up_projection.retained_reference().unwrap(),
            down_projection: self.down_projection.retained_reference().unwrap(),
        }
    }
}

/// Writes one expert row from `streamed_weights` into `slot` of the table's
/// preallocated weights. Each projection's expert axis is 0, so the slice spans
/// `[slot, 0.., 0..]` to `[slot+1, end, end]`. `slice_update` donates the
/// destination buffer when it is row-contiguous and uniquely held, so only the
/// one expert row is written — no whole-tensor copy.
pub(super) fn write_expert_into_slot(
    runtime: &MlxRuntime,
    table_weights: &mut Qwen3_5PagedExpertWeights,
    streamed_weights: &Qwen3_5PagedExpertWeights,
    expert_row: usize,
    slot: usize,
) -> Result<(), MlxRuntimeError> {
    let slot_i32 = i32::try_from(slot).map_err(|_| MlxRuntimeError::RuntimeOperation {
        operation: "write expert into slot",
        description: "slot index exceeds the MLX shape range".to_owned(),
    })?;
    let row_i32 = i32::try_from(expert_row).map_err(|_| MlxRuntimeError::RuntimeOperation {
        operation: "write expert into slot",
        description: "expert row index exceeds the MLX shape range".to_owned(),
    })?;
    table_weights.gate_projection = write_projection_into_slot(
        runtime,
        &table_weights.gate_projection,
        &streamed_weights.gate_projection,
        row_i32,
        slot_i32,
    )?;
    table_weights.up_projection = write_projection_into_slot(
        runtime,
        &table_weights.up_projection,
        &streamed_weights.up_projection,
        row_i32,
        slot_i32,
    )?;
    table_weights.down_projection = write_projection_into_slot(
        runtime,
        &table_weights.down_projection,
        &streamed_weights.down_projection,
        row_i32,
        slot_i32,
    )?;
    Ok(())
}

fn write_projection_into_slot(
    runtime: &MlxRuntime,
    destination: &Qwen3_5AffineWeights,
    source: &Qwen3_5AffineWeights,
    source_row: i32,
    slot: i32,
) -> Result<Qwen3_5AffineWeights, MlxRuntimeError> {
    match (destination, source) {
        (
            Qwen3_5AffineWeights::NativeBfloat16 {
                weight: dest_weight,
            },
            Qwen3_5AffineWeights::NativeBfloat16 {
                weight: source_weight,
            },
        ) => {
            let updated =
                write_row_into_slot(runtime, dest_weight, source_weight, source_row, slot)?;
            Ok(Qwen3_5AffineWeights::NativeBfloat16 { weight: updated })
        }
        (
            Qwen3_5AffineWeights::Quantized {
                packed_weight: dest_packed,
                quantization_scales: dest_scales,
                quantization_biases: dest_biases,
                quantization_bits,
                quantization_group_size,
            },
            Qwen3_5AffineWeights::Quantized {
                packed_weight: source_packed,
                quantization_scales: source_scales,
                quantization_biases: source_biases,
                ..
            },
        ) => {
            let packed =
                write_row_into_slot(runtime, dest_packed, source_packed, source_row, slot)?;
            let scales =
                write_row_into_slot(runtime, dest_scales, source_scales, source_row, slot)?;
            let biases =
                write_row_into_slot(runtime, dest_biases, source_biases, source_row, slot)?;
            Ok(Qwen3_5AffineWeights::Quantized {
                packed_weight: packed,
                quantization_scales: scales,
                quantization_biases: biases,
                quantization_bits: *quantization_bits,
                quantization_group_size: *quantization_group_size,
            })
        }
        _ => Err(MlxRuntimeError::RuntimeOperation {
            operation: "write expert into slot",
            description: "streamed and slot table projections have mismatched quantization"
                .to_owned(),
        }),
    }
}

/// Slices one expert row from `source` (axis 0, index `source_row`) and writes
/// it into `slot` of `destination` (axis 0). Uses `slice_update` which donates
/// the destination buffer when row-contiguous and uniquely held, so the write
/// copies only the one expert row instead of the whole tensor.
fn write_row_into_slot(
    runtime: &MlxRuntime,
    destination: &MlxArray,
    source: &MlxArray,
    source_row: i32,
    slot: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let destination_shape = destination.shape();
    if destination_shape.len() < 2 {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "write expert row into slot",
            description: "expert weight arrays must have at least two dimensions".to_owned(),
        });
    }
    // Slice the source expert row: [source_row, 0.., 0..] .. [source_row+1, end, end]
    let mut source_starts = vec![0_i32; destination_shape.len()];
    let mut source_stops = destination_shape.clone();
    let source_strides = vec![1_i32; destination_shape.len()];
    source_starts[0] = source_row;
    source_stops[0] = source_row + 1;
    let expert_row = runtime.slice(source, &source_starts, &source_stops, &source_strides)?;
    // Write that row into `slot` of the destination.
    let mut slot_starts = vec![0_i32; destination_shape.len()];
    let mut slot_stops = destination_shape.clone();
    let slot_strides = vec![1_i32; destination_shape.len()];
    slot_starts[0] = slot;
    slot_stops[0] = slot + 1;
    let updated = runtime.slice_update(
        destination,
        &expert_row,
        &slot_starts,
        &slot_stops,
        &slot_strides,
    )?;
    Ok(updated)
}

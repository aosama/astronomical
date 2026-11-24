//! A dense projection's weight: as bound from the artifact, or converted at
//! load into the register order the register-ordered matmul reads.
//!
//! Only dense projections (attention, the dense MLP, the output head) can be
//! converted, so the conversion lives in this type rather than in
//! `Qwen3_5AffineWeights`, which expert stacks share: code that only ever
//! sees expert stacks never has to know a converted weight exists.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::row_batched_matmul::{RegisterOrderedWeight, RestoredMlxRows};

/// The register-ordered matmul reads 4-bit codes in groups of 64 only.
const REGISTER_ORDERED_BITS: i32 = 4;
const REGISTER_ORDERED_GROUP_SIZE: i32 = 64;

/// A dense projection: its weights, and the module name the artifact
/// ships them under (`language_model.model.layers.3.linear_attn.in_proj_qkv`).
/// The name is what a decision model's add-on for this projection is filed
/// under, so the projection itself can find its add-on in the forward pass.
#[derive(Debug)]
pub struct Qwen3_5DenseProjection {
    pub module_name: String,
    pub storage: Qwen3_5ProjectionStorage,
}

/// How a dense projection's weights are held.
#[derive(Debug)]
pub enum Qwen3_5ProjectionStorage {
    /// The module as bound: bf16, MLX's affine packing, or a codebook.
    Bound(Qwen3_5AffineWeights),
    /// A 4-bit, group-64 module converted into register order. It replaces
    /// the bound arrays, so every row count of it multiplies through the
    /// register-ordered matmul, prompt chunks included.
    RegisterOrdered(RegisterOrderedWeight),
}

impl Qwen3_5DenseProjection {
    /// The module `module_name` as bound from the artifact.
    pub fn named(module_name: &str, bound_weights: Qwen3_5AffineWeights) -> Self {
        Self {
            module_name: module_name.to_owned(),
            storage: Qwen3_5ProjectionStorage::Bound(bound_weights),
        }
    }

    /// The bound weights, unless the projection was converted.
    pub fn bound(&self) -> Option<&Qwen3_5AffineWeights> {
        match &self.storage {
            Qwen3_5ProjectionStorage::Bound(bound_weights) => Some(bound_weights),
            Qwen3_5ProjectionStorage::RegisterOrdered(_) => None,
        }
    }

    /// The bound weights to swap in place, unless the projection was
    /// converted (the quantization probe only ever swaps bound modules).
    pub fn bound_mut(&mut self) -> Option<&mut Qwen3_5AffineWeights> {
        match &mut self.storage {
            Qwen3_5ProjectionStorage::Bound(bound_weights) => Some(bound_weights),
            Qwen3_5ProjectionStorage::RegisterOrdered(_) => None,
        }
    }

    pub fn append_array_references<'weights>(&'weights self, arrays: &mut Vec<&'weights MlxArray>) {
        match &self.storage {
            Qwen3_5ProjectionStorage::Bound(bound_weights) => {
                bound_weights.append_array_references(arrays)
            }
            Qwen3_5ProjectionStorage::RegisterOrdered(converted) => {
                arrays.push(&converted.codes);
                arrays.push(&converted.scale_bias_words);
            }
        }
    }

    /// The whole weight dequantized, `[output, input]`, in its parameter
    /// dtype (bf16 for a converted projection). None for a codebook.
    pub fn dequantized(&self, runtime: &MlxRuntime) -> Option<Result<MlxArray, MlxRuntimeError>> {
        match &self.storage {
            Qwen3_5ProjectionStorage::Bound(bound_weights) => {
                bound_weights_dequantized(runtime, bound_weights)
            }
            Qwen3_5ProjectionStorage::RegisterOrdered(converted) => {
                Some(converted_dequantized(runtime, converted))
            }
        }
    }

    /// The output rows `row_ids`, dequantized, `[row_ids.len(), input]`.
    /// None for a codebook.
    pub fn dequantized_rows(
        &self,
        runtime: &MlxRuntime,
        row_ids: &[i32],
    ) -> Option<Result<MlxArray, MlxRuntimeError>> {
        match &self.storage {
            Qwen3_5ProjectionStorage::Bound(bound_weights) => {
                bound_rows_dequantized(runtime, bound_weights, row_ids)
            }
            Qwen3_5ProjectionStorage::RegisterOrdered(converted) => {
                Some(converted_rows_dequantized(runtime, converted, row_ids))
            }
        }
    }
}

fn bound_weights_dequantized(
    runtime: &MlxRuntime,
    bound_weights: &Qwen3_5AffineWeights,
) -> Option<Result<MlxArray, MlxRuntimeError>> {
    match bound_weights {
        Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
            Some(runtime.copy(weight).map_err(MlxRuntimeError::from))
        }
        Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            quantization_group_size,
            quantization_bits,
        } => Some(
            runtime
                .dequantize_affine(
                    packed_weight,
                    quantization_scales,
                    quantization_biases,
                    *quantization_group_size,
                    *quantization_bits,
                )
                .map_err(MlxRuntimeError::from),
        ),
        Qwen3_5AffineWeights::Jangtq2Codebook { .. } => None,
    }
}

fn bound_rows_dequantized(
    runtime: &MlxRuntime,
    bound_weights: &Qwen3_5AffineWeights,
    row_ids: &[i32],
) -> Option<Result<MlxArray, MlxRuntimeError>> {
    match bound_weights {
        Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
            Some(taken_rows(runtime, weight, row_ids))
        }
        Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            quantization_group_size,
            quantization_bits,
        } => Some(taken_rows_dequantized(
            runtime,
            [packed_weight, quantization_scales, quantization_biases],
            row_ids,
            *quantization_group_size,
            *quantization_bits,
        )),
        Qwen3_5AffineWeights::Jangtq2Codebook { .. } => None,
    }
}

fn taken_rows(
    runtime: &MlxRuntime,
    weight: &MlxArray,
    row_ids: &[i32],
) -> Result<MlxArray, MlxRuntimeError> {
    let row_id_array = runtime.array_from_i32(row_ids, &[row_ids.len() as i32])?;
    Ok(runtime.take_axis(weight, &row_id_array, 0)?)
}

fn taken_rows_dequantized(
    runtime: &MlxRuntime,
    [packed_weight, scales, biases]: [&MlxArray; 3],
    row_ids: &[i32],
    group_size: i32,
    bits: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let row_id_array = runtime.array_from_i32(row_ids, &[row_ids.len() as i32])?;
    Ok(runtime.dequantize_affine(
        &runtime.take_axis(packed_weight, &row_id_array, 0)?,
        &runtime.take_axis(scales, &row_id_array, 0)?,
        &runtime.take_axis(biases, &row_id_array, 0)?,
        group_size,
        bits,
    )?)
}

fn converted_dequantized(
    runtime: &MlxRuntime,
    converted: &RegisterOrderedWeight,
) -> Result<MlxArray, MlxRuntimeError> {
    let restored = converted
        .restored_mlx_layout(runtime)
        .map_err(restoration_failure)?;
    dequantize_restored(runtime, &restored)
}

fn converted_rows_dequantized(
    runtime: &MlxRuntime,
    converted: &RegisterOrderedWeight,
    row_ids: &[i32],
) -> Result<MlxArray, MlxRuntimeError> {
    let restored = converted
        .restored_rows(runtime, row_ids)
        .map_err(restoration_failure)?;
    dequantize_restored(runtime, &restored)
}

fn dequantize_restored(
    runtime: &MlxRuntime,
    restored: &RestoredMlxRows,
) -> Result<MlxArray, MlxRuntimeError> {
    Ok(runtime.dequantize_affine(
        &restored.packed_weight,
        &restored.scales,
        &restored.biases,
        REGISTER_ORDERED_GROUP_SIZE,
        REGISTER_ORDERED_BITS,
    )?)
}

fn restoration_failure(
    restoration_error: crate::row_batched_matmul::RowBatchedProjectionError,
) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "restore MLX's layout from a register-ordered projection",
        description: restoration_error.to_string(),
    }
}

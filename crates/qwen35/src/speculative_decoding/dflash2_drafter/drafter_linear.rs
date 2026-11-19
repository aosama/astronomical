//! One drafter linear layer, kept either as the checkpoint stores it (bf16)
//! or quantized to 4 bits at load.
//!
//! The drafter streams all its weights every round, so its precision is a
//! direct cost: the 27B's drafter is 3.6 GB in bf16, 14.7 ms a round on an M5
//! Pro, against about 1 GB at 4 bits. A quantized linear is converted into
//! register order when this GPU has proven the register-ordered matmul, as
//! the target's projections are.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::MlxArray;

use crate::row_batched_matmul::{
    RegisterOrderedWeight, RowBatchedMatmulKernels, RowBatchedProjectionError,
    register_ordered_matmul_any_rows,
};

const QUANTIZATION_GROUP_SIZE: i32 = 64;
const QUANTIZATION_BITS: i32 = 4;

pub enum DrafterLinear {
    /// The checkpoint's bf16 weight as the transposed view MLX's matmul reads
    /// directly: `input @ weightᵀ`, with no copy of the stored `[out, in]`.
    FullPrecision { transposed_weight: MlxArray },
    /// MLX's affine 4-bit packing of the same weight, group 64.
    Quantized {
        packed_weight: MlxArray,
        scales: MlxArray,
        biases: MlxArray,
    },
    /// The same 4-bit weight in register order. The kernel handles are cheap
    /// host objects; MLX compiles each kernel once per process.
    RegisterOrdered {
        weight: RegisterOrderedWeight,
        kernels: RowBatchedMatmulKernels,
    },
}

impl DrafterLinear {
    pub fn full_precision(
        runtime: &MlxRuntime,
        weight: &MlxArray,
    ) -> Result<Self, MlxRuntimeError> {
        Ok(Self::FullPrecision {
            transposed_weight: runtime.transpose_axes(weight, &[1, 0])?,
        })
    }

    pub fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        match self {
            Self::FullPrecision { transposed_weight } => {
                Ok(runtime.matmul(input, transposed_weight)?)
            }
            Self::Quantized {
                packed_weight,
                scales,
                biases,
            } => Ok(runtime.quantized_matmul_affine(
                input,
                packed_weight,
                scales,
                biases,
                true,
                QUANTIZATION_GROUP_SIZE,
                QUANTIZATION_BITS,
            )?),
            Self::RegisterOrdered { weight, kernels } => {
                register_ordered_matmul_any_rows(runtime, kernels, input, weight)
                    .map_err(projection_failure)
            }
        }
    }

    /// Quantizes a full-precision linear to 4 bits in place and materializes
    /// it, so the bf16 view can be released; with `use_register_order` it is
    /// then converted into register order unless its shape cannot be tiled.
    /// A linear already quantized is left alone.
    pub fn quantize_in_place(
        &mut self,
        runtime: &MlxRuntime,
        use_register_order: bool,
    ) -> Result<(), MlxRuntimeError> {
        let Self::FullPrecision { transposed_weight } = self else {
            return Ok(());
        };
        let weight = runtime.transpose_axes(transposed_weight, &[1, 0])?;
        let (packed_weight, scales, biases) =
            runtime.quantize_affine(&weight, QUANTIZATION_GROUP_SIZE, QUANTIZATION_BITS)?;
        runtime.evaluate_arrays(&[&packed_weight, &scales, &biases])?;
        if use_register_order {
            let kernels = RowBatchedMatmulKernels::declare()?;
            match RegisterOrderedWeight::convert(
                runtime,
                &kernels,
                &packed_weight,
                &scales,
                &biases,
            ) {
                Ok(converted) => {
                    runtime.evaluate_arrays(&[&converted.codes, &converted.scale_bias_words])?;
                    *self = Self::RegisterOrdered {
                        weight: converted,
                        kernels,
                    };
                    return Ok(());
                }
                Err(RowBatchedProjectionError::Refused(_)) => {}
                Err(conversion_error) => return Err(projection_failure(conversion_error)),
            }
        }
        *self = Self::Quantized {
            packed_weight,
            scales,
            biases,
        };
        Ok(())
    }
}

fn projection_failure(projection_error: RowBatchedProjectionError) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "project drafter rows through the register-ordered matmul",
        description: projection_error.to_string(),
    }
}

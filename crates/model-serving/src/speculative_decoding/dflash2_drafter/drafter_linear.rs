//! One drafter linear layer, kept either as the checkpoint stores it (bf16)
//! or quantized to 4 bits at load.
//!
//! The drafter streams all its weights every round, so its precision is a
//! direct cost: the 27B's drafter is 3.6 GB in bf16, 14.7 ms a round on an M5
//! Pro, against about 1 GB at 4 bits. A quantized linear over a block's 8 rows
//! uses the row-batched matmul when this GPU has proven it.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::MlxArray;

use crate::row_batched_matmul::{
    RowBatchedMatmulKernels, row_batched_quantized_matmul, row_batched_quantized_matmul_accepts,
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
        /// Present when the row-batched matmul is proven on this GPU. The
        /// handles are cheap host objects; MLX compiles the kernel once.
        row_batched_matmul_kernels: Option<RowBatchedMatmulKernels>,
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
                row_batched_matmul_kernels,
            } => {
                if let Some(kernels) = row_batched_matmul_kernels
                    && row_batched_quantized_matmul_accepts(input, packed_weight, scales, biases)
                {
                    return row_batched_quantized_matmul(
                        runtime,
                        kernels,
                        input,
                        packed_weight,
                        scales,
                        biases,
                    )
                    .map_err(|projection_error| {
                        MlxRuntimeError::RuntimeOperation {
                            operation: "project drafter rows through the row-batched matmul",
                            description: projection_error.to_string(),
                        }
                    });
                }
                Ok(runtime.quantized_matmul_affine(
                    input,
                    packed_weight,
                    scales,
                    biases,
                    true,
                    QUANTIZATION_GROUP_SIZE,
                    QUANTIZATION_BITS,
                )?)
            }
        }
    }

    /// Quantizes a full-precision linear to 4 bits in place and materializes
    /// it, so the bf16 view can be released; a quantized one is left alone.
    pub fn quantize_in_place(
        &mut self,
        runtime: &MlxRuntime,
        use_row_batched_matmul: bool,
    ) -> Result<(), MlxRuntimeError> {
        let Self::FullPrecision { transposed_weight } = self else {
            return Ok(());
        };
        let weight = runtime.transpose_axes(transposed_weight, &[1, 0])?;
        let (packed_weight, scales, biases) =
            runtime.quantize_affine(&weight, QUANTIZATION_GROUP_SIZE, QUANTIZATION_BITS)?;
        runtime.evaluate_arrays(&[&packed_weight, &scales, &biases])?;
        let row_batched_matmul_kernels = if use_row_batched_matmul {
            Some(RowBatchedMatmulKernels::declare()?)
        } else {
            None
        };
        *self = Self::Quantized {
            packed_weight,
            scales,
            biases,
            row_batched_matmul_kernels,
        };
        Ok(())
    }
}

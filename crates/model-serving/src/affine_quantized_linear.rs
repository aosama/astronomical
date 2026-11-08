//! One MLX affine 4-bit quantized projection, shared by every engine whose
//! artifact stores its linears in MLX's packed layout (Qwen-Image 2.1, and the
//! 4-bit FLUX.2 Klein build).
//!
//! The layout is `weight` `[out, in / 8]` uint32 (eight 4-bit values per word)
//! plus `scales` and `biases` `[out, in / 64]`. MLX consumes exactly this layout
//! through `quantized_matmul_affine` with `transpose_weights = true`, so the
//! tensors load without repacking. The affine `biases` are dequantization
//! offsets, not layer biases.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_c_rust::MlxArray;
use thiserror::Error;

pub(crate) use crate::affine_quantization_scheme::{
    AFFINE_QUANTIZATION_BITS, AFFINE_QUANTIZATION_GROUP_SIZE, AFFINE_QUANTIZED_VALUES_PER_WORD,
};

/// Why a quantized projection could not be bound or applied.
#[derive(Debug, Error)]
pub enum AffineQuantizedLinearError {
    #[error(
        "quantized tensor '{tensor_name}' has shape {actual_shape:?}, expected {expected_shape:?}"
    )]
    TensorShape {
        tensor_name: String,
        actual_shape: Vec<i32>,
        expected_shape: Vec<usize>,
    },
    #[error(
        "quantized projection expected trailing width {expected_width}, received {actual_shape:?}"
    )]
    InputWidth {
        expected_width: usize,
        actual_shape: Vec<i32>,
    },
    #[error(transparent)]
    Mlx(#[from] MlxRuntimeError),
}

impl From<mlx_c_rust::MlxCError> for AffineQuantizedLinearError {
    fn from(captured_error: mlx_c_rust::MlxCError) -> Self {
        Self::Mlx(MlxRuntimeError::from(captured_error))
    }
}

/// One 4-bit affine-quantized projection with its logical widths.
#[derive(Debug)]
pub(crate) struct AffineQuantizedLinear {
    input_width: usize,
    weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

impl AffineQuantizedLinear {
    /// Binds the three tensors of the projection named `tensor_prefix`, checking
    /// they have the packed shapes the quantization scheme implies for a
    /// `[output_width, input_width]` linear. A mismatch means the artifact is
    /// not the reviewed one, so it is refused rather than reinterpreted.
    pub(crate) fn bind(
        tensor_prefix: &str,
        tensors: [MlxArray; 3],
        input_width: usize,
        output_width: usize,
    ) -> Result<Self, AffineQuantizedLinearError> {
        let [weight, scales, biases] = tensors;
        let group_count = input_width / AFFINE_QUANTIZATION_GROUP_SIZE;
        expect_shape(
            tensor_prefix,
            "weight",
            &weight,
            &[output_width, input_width / AFFINE_QUANTIZED_VALUES_PER_WORD],
        )?;
        expect_shape(
            tensor_prefix,
            "scales",
            &scales,
            &[output_width, group_count],
        )?;
        expect_shape(
            tensor_prefix,
            "biases",
            &biases,
            &[output_width, group_count],
        )?;
        Ok(Self {
            input_width,
            weight,
            scales,
            biases,
        })
    }

    /// `activations @ weightᵀ` computed on the packed weights.
    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
    ) -> Result<MlxArray, AffineQuantizedLinearError> {
        let activation_shape = activations.shape();
        let trailing_width = activation_shape.last().copied();
        if activation_shape.len() < 2 || trailing_width != i32::try_from(self.input_width).ok() {
            return Err(AffineQuantizedLinearError::InputWidth {
                expected_width: self.input_width,
                actual_shape: activation_shape,
            });
        }
        Ok(runtime.quantized_matmul_affine(
            activations,
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            AFFINE_QUANTIZATION_GROUP_SIZE as i32,
            AFFINE_QUANTIZATION_BITS,
        )?)
    }

    /// The three packed arrays, for callers that materialize retained weights.
    pub(crate) fn arrays(&self) -> Vec<&MlxArray> {
        vec![&self.weight, &self.scales, &self.biases]
    }
}

fn expect_shape(
    tensor_prefix: &str,
    tensor_role: &str,
    tensor: &MlxArray,
    expected_shape: &[usize],
) -> Result<(), AffineQuantizedLinearError> {
    let shape_matches = tensor.shape().len() == expected_shape.len()
        && tensor.shape().iter().zip(expected_shape).all(
            |(actual_dimension, expected_dimension)| {
                usize::try_from(*actual_dimension).ok() == Some(*expected_dimension)
            },
        );
    if shape_matches {
        return Ok(());
    }
    Err(AffineQuantizedLinearError::TensorShape {
        tensor_name: format!("{tensor_prefix}.{tensor_role}"),
        actual_shape: tensor.shape(),
        expected_shape: expected_shape.to_vec(),
    })
}

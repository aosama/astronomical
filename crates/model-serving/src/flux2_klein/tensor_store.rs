//! Where a FLUX.2 Klein component's weights come from, looked up by their
//! Diffusers names whatever the artifact's on-disk layout.
//!
//! Two layouts exist: the BF16 Diffusers build, and the 4-bit MLX build whose
//! names, packing and convolution layout differ (issue #1904). Components ask
//! this store for tensors by Diffusers name and never see the layout, so the
//! layout-specific knowledge stays in `mlx_4bit_layout` and this file.

use std::sync::Arc;

use astronomical_runtime_integration::{
    MlxRuntime, MlxRuntimeError, MlxSafetensors, PositionalFileReadMetrics,
};
use mlx_c_rust::MlxArray;
use thiserror::Error;

use crate::affine_quantized_linear::{
    AFFINE_QUANTIZATION_BITS, AFFINE_QUANTIZATION_GROUP_SIZE, AffineQuantizedLinear,
    AffineQuantizedLinearError,
};

use super::mlx_4bit_layout::mlx_4bit_tensor_name;
pub(crate) use super::weight_layout::{
    Flux2KleinComponent, Flux2KleinComponentWeightFiles, Flux2KleinWeightLayout,
};

#[derive(Debug, Error)]
pub enum Flux2KleinTensorStoreError {
    #[error("FLUX.2 Klein {component:?} tensor '{tensor_name}' is missing")]
    MissingTensor {
        component: Flux2KleinComponent,
        tensor_name: String,
    },
    #[error("failed to clone a FLUX.2 Klein {component:?} weight descriptor")]
    DescriptorClone {
        component: Flux2KleinComponent,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    QuantizedLinear(#[from] AffineQuantizedLinearError),
    #[error(transparent)]
    Mlx(#[from] MlxRuntimeError),
}

impl From<mlx_c_rust::MlxCError> for Flux2KleinTensorStoreError {
    fn from(captured_error: mlx_c_rust::MlxCError) -> Self {
        Self::Mlx(MlxRuntimeError::from(captured_error))
    }
}

/// One stored weight: a dense array, or a projection kept in its 4-bit form.
#[derive(Debug)]
pub(crate) enum Flux2KleinWeight {
    Dense(MlxArray),
    Affine4Bit(AffineQuantizedLinear),
}

impl Flux2KleinWeight {
    /// `input @ weightᵀ`, on the packed weights when the projection is 4-bit.
    pub(crate) fn project(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, Flux2KleinTensorStoreError> {
        match self {
            Self::Dense(weight) => {
                let transposed_weight = runtime.transpose_axes(weight, &[1, 0])?;
                Ok(runtime.matmul(input, &transposed_weight)?)
            }
            Self::Affine4Bit(quantized_linear) => Ok(quantized_linear.forward(runtime, input)?),
        }
    }

    /// The dense array, for tensors that are never quantized (norm scales).
    #[must_use]
    pub(crate) const fn dense(&self) -> Option<&MlxArray> {
        match self {
            Self::Dense(weight) => Some(weight),
            Self::Affine4Bit(_) => None,
        }
    }

    /// Ensures the stored arrays are materialized, so a retained weight is
    /// resident before the first request rather than during it.
    pub(crate) fn arrays(&self) -> Vec<&MlxArray> {
        match self {
            Self::Dense(weight) => vec![weight],
            Self::Affine4Bit(quantized_linear) => quantized_linear.arrays(),
        }
    }
}

/// Maps every shard of `weight_files` through cloned descriptors, leaving the
/// files available for the next mapping (streamed blocks map per operation).
pub(crate) fn map_component_weight_files(
    weight_files: &Flux2KleinComponentWeightFiles,
    runtime: &MlxRuntime,
    positional_file_read_metrics: Option<&Arc<PositionalFileReadMetrics>>,
) -> Result<Flux2KleinTensorStore, Flux2KleinTensorStoreError> {
    let mut shards = Vec::new();
    for cloned_file in weight_files.clone_shard_files().map_err(|source| {
        Flux2KleinTensorStoreError::DescriptorClone {
            component: weight_files.component(),
            source,
        }
    })? {
        shards.push(runtime.load_safetensors(cloned_file, positional_file_read_metrics.cloned())?);
    }
    Ok(Flux2KleinTensorStore::new(
        weight_files.layout(),
        weight_files.component(),
        shards,
    ))
}

/// One component's mapped weight shards in a known layout.
pub(crate) struct Flux2KleinTensorStore {
    layout: Flux2KleinWeightLayout,
    component: Flux2KleinComponent,
    shards: Vec<MlxSafetensors>,
}

impl Flux2KleinTensorStore {
    #[must_use]
    pub(crate) const fn new(
        layout: Flux2KleinWeightLayout,
        component: Flux2KleinComponent,
        shards: Vec<MlxSafetensors>,
    ) -> Self {
        Self {
            layout,
            component,
            shards,
        }
    }

    /// The weight named `diffusers_tensor_name` as stored. A 4-bit projection
    /// stays packed and is checked against its logical `[out, in]` shape.
    pub(crate) fn weight(
        &self,
        diffusers_tensor_name: &str,
        logical_shape: &[usize],
    ) -> Result<Flux2KleinWeight, Flux2KleinTensorStoreError> {
        if self.layout == Flux2KleinWeightLayout::DiffusersBf16 {
            return Ok(Flux2KleinWeight::Dense(self.stored(diffusers_tensor_name)?));
        }
        let stored_name = mlx_4bit_tensor_name(self.component, diffusers_tensor_name);
        match (self.quantized_triple(&stored_name), logical_shape) {
            (Some(packed_tensors), [output_width, input_width]) => {
                Ok(Flux2KleinWeight::Affine4Bit(AffineQuantizedLinear::bind(
                    diffusers_tensor_name,
                    packed_tensors,
                    *input_width,
                    *output_width,
                )?))
            }
            _ => Ok(Flux2KleinWeight::Dense(self.stored(&stored_name)?)),
        }
    }

    /// The tensor named `diffusers_tensor_name` as a dense array in the
    /// Diffusers layout. In the 4-bit build a packed projection is dequantized
    /// and an MLX-layout convolution kernel `[out, kh, kw, in]` is returned as
    /// `[out, in, kh, kw]`, so components with dense arithmetic need no change.
    /// The kernel layout is applied by component, never inferred from shape:
    /// `encoder.conv_in` has the same shape in both layouts.
    pub(crate) fn dense(
        &self,
        runtime: &MlxRuntime,
        diffusers_tensor_name: &str,
    ) -> Result<MlxArray, Flux2KleinTensorStoreError> {
        if self.layout == Flux2KleinWeightLayout::DiffusersBf16 {
            return self.stored(diffusers_tensor_name);
        }
        let stored_name = mlx_4bit_tensor_name(self.component, diffusers_tensor_name);
        if let Some([weight, scales, biases]) = self.quantized_triple(&stored_name) {
            return Ok(runtime.dequantize_affine(
                &weight,
                &scales,
                &biases,
                AFFINE_QUANTIZATION_GROUP_SIZE as i32,
                AFFINE_QUANTIZATION_BITS,
            )?);
        }
        let stored_tensor = self.stored(&stored_name)?;
        if self.component == Flux2KleinComponent::Vae && stored_tensor.shape().len() == 4 {
            return Ok(runtime.transpose_axes(&stored_tensor, &[0, 3, 1, 2])?);
        }
        Ok(stored_tensor)
    }

    /// The three packed tensors of the projection whose weight is
    /// `stored_weight_name`, when it is stored 4-bit.
    fn quantized_triple(&self, stored_weight_name: &str) -> Option<[MlxArray; 3]> {
        let stem = stored_weight_name.strip_suffix(".weight")?;
        let scales = self.lookup(&format!("{stem}.scales"))?;
        let biases = self.lookup(&format!("{stem}.biases"))?;
        let weight = self.lookup(stored_weight_name)?;
        Some([weight, scales, biases])
    }

    fn stored(&self, stored_name: &str) -> Result<MlxArray, Flux2KleinTensorStoreError> {
        self.lookup(stored_name)
            .ok_or_else(|| Flux2KleinTensorStoreError::MissingTensor {
                component: self.component,
                tensor_name: stored_name.to_owned(),
            })
    }

    fn lookup(&self, stored_name: &str) -> Option<MlxArray> {
        self.shards
            .iter()
            .find_map(|shard| shard.tensor(stored_name).ok())
    }
}

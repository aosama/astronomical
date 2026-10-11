//! The shared Qwen3.5 model base: one loaded artifact's runtime, configuration,
//! typed weights, vision tower, retained Metal kernels, and compiled graphs.
//!
//! Issue #1132: everything here is identical for the fully-resident engine and
//! the SSD-streaming engine. The residency fork lives in the engine wrappers
//! (`Qwen3_5StreamingModel` and, after the fork, the resident model), which
//! deref to this base for every shared field and method.

use astronomical_mlx_c_rust::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel,
};
use astronomical_runtime_integration::MlxRuntime;

use std::cell::RefCell;

use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::weights::Qwen3_5Weights;
use crate::qwen3_5_core::vision::Qwen3_5VisionModel;
use crate::{DecoderCacheLayout, MlxRamBudget};

use super::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;

/// One resident native Qwen3.5 text model, optional vision tower, and its direct MLX runtime.
#[derive(Debug)]
pub struct Qwen3_5ModelBase {
    pub(crate) runtime: MlxRuntime,
    pub(crate) config: Qwen3_5Config,
    pub(crate) decoder_cache_layout: DecoderCacheLayout,
    pub(crate) weights: Qwen3_5Weights,
    pub(crate) vision_model: Option<Qwen3_5VisionModel>,
    /// Single-source MLX RAM split for context, activations, streaming, and experts.
    pub(crate) mlx_ram_budget: RefCell<MlxRamBudget>,

    pub(crate) gated_delta_kernel: Option<MlxMetalKernel>,
    pub(crate) gated_delta_checkpoint_kernel: Option<MlxMetalKernel>,
    /// Fused decode prework kernel; None when demoted or environment-disabled.
    pub(crate) gdn_decode_prework_kernel: Option<MlxMetalKernel>,
    pub(crate) sorted_expert_weighted_sum_kernel: Option<MlxMetalKernel>,
    pub(crate) compiled_swiglu: MlxCompiledSwiGlu,
    pub(crate) compiled_elementwise_graphs: MlxCompiledElementwiseGraphs,
    pub(crate) chunking: Qwen3_5ModelChunkingConfiguration,
    /// Model-owned BF16 scalar for the query normalization scale in every
    /// linear-attention forward pass.
    pub(crate) inverse_linear_head_dimension_scale: MlxArray,
    /// Model-owned BF16 scalar for the key normalization scale in every
    /// linear-attention forward pass.
    pub(crate) inverse_square_root_linear_head_dimension_scale: MlxArray,
    /// Model-owned BF16 per-channel weight folding the query normalization
    /// scale into one `fast_rms_norm` launch on the prefill composed path
    /// (issue #915 item 5); per-channel values equal the scalar scale exactly.
    pub(crate) query_normalization_scale_weight: MlxArray,
    /// Model-owned BF16 per-channel weight folding the key normalization
    /// scale into one `fast_rms_norm` launch on the prefill composed path
    /// (issue #915 item 5); per-channel values equal the scalar scale exactly.
    pub(crate) key_normalization_scale_weight: MlxArray,
}

impl Qwen3_5ModelBase {
    /// Returns the MLX runtime used by this model.
    #[must_use]
    pub fn runtime(&self) -> &MlxRuntime {
        &self.runtime
    }

    /// Returns the single-source MLX RAM budget owner.
    #[must_use]
    pub fn mlx_ram_budget(&self) -> std::cell::Ref<'_, MlxRamBudget> {
        self.mlx_ram_budget.borrow()
    }

    /// Returns the mutable single-source MLX RAM budget owner.
    pub fn mlx_ram_budget_mut(&self) -> std::cell::RefMut<'_, MlxRamBudget> {
        self.mlx_ram_budget.borrow_mut()
    }

    pub(crate) fn sorted_expert_weighted_sum_kernel(
        &self,
    ) -> Result<&MlxMetalKernel, Qwen3_5ExecutionError> {
        self.sorted_expert_weighted_sum_kernel
            .as_ref()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "sparse Qwen3.5 execution requires a sorted expert output kernel",
            })
    }

    /// Returns the validated text configuration bound to this loaded model.
    #[must_use]
    pub(crate) const fn config(&self) -> &Qwen3_5Config {
        &self.config
    }

    /// Returns the validated decoder-cache layout bound to this model artifact.
    #[must_use]
    pub(crate) const fn decoder_cache_layout(&self) -> &DecoderCacheLayout {
        &self.decoder_cache_layout
    }

    /// Returns the optional vision tower loaded beside the language model.
    #[must_use]
    pub fn vision_model(&self) -> Option<&Qwen3_5VisionModel> {
        self.vision_model.as_ref()
    }

    /// Returns the resident target payload bytes wired after materialization.
    #[must_use]
    pub(crate) fn resident_model_payload_byte_count(&self) -> u64 {
        self.weights.total_payload_bytes()
    }

    pub(crate) fn materialize_target_weights(&self) -> Result<(), Qwen3_5ExecutionError> {
        self.weights.materialize(&self.runtime)?;
        self.runtime.evaluate_arrays(&[
            &self.inverse_linear_head_dimension_scale,
            &self.inverse_square_root_linear_head_dimension_scale,
            &self.query_normalization_scale_weight,
            &self.key_normalization_scale_weight,
        ])?;
        Ok(())
    }

    pub(crate) fn embedding_lookup(
        &self,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        match &self.weights.embedding_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                Ok(self.runtime.take_axis(weight, token_indices, 0)?)
            }
            Qwen3_5AffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => {
                let selected_weights = self.runtime.take_axis(packed_weight, token_indices, 0)?;
                let selected_scales =
                    self.runtime
                        .take_axis(quantization_scales, token_indices, 0)?;
                let selected_biases =
                    self.runtime
                        .take_axis(quantization_biases, token_indices, 0)?;
                Ok(self.runtime.dequantize_affine(
                    &selected_weights,
                    &selected_scales,
                    &selected_biases,
                    *quantization_group_size,
                    *quantization_bits,
                )?)
            }
        }
    }

    pub(crate) fn quantized_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &Qwen3_5AffineWeights,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        match affine_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                let transposed_weight = self.runtime.transpose_axes(weight, &[1, 0])?;
                Ok(self.runtime.matmul(activations, &transposed_weight)?)
            }
            Qwen3_5AffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => Ok(self.runtime.quantized_matmul_affine(
                activations,
                packed_weight,
                quantization_scales,
                quantization_biases,
                true,
                *quantization_group_size,
                *quantization_bits,
            )?),
        }
    }
}

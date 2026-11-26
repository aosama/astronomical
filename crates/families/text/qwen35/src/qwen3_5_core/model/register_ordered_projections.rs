//! Converts a resident model's dense 4-bit projections into register order
//! and multiplies them through the register-ordered matmul.
//!
//! Measured over the 27B's 64 layers on an M5 Pro, the converted projections
//! cost 59 ms for an 8-row verify pass against MLX's 133 ms, and 57 ms for
//! one row against MLX's 52 ms (`cargo measure-k-row-verify-cost`). A model
//! that drafts verifies every step, so it converts; one that decodes a row a
//! step keeps MLX's layout. This is the layout TensorFold's engine serves the
//! Qwen 27B from on chips with matrix units.

use mlx_bridge::MlxArray;

use super::Qwen3_5ModelBase;
use super::base::Qwen3_5LinearWeights;
use crate::kernel_capability::worker_process_kernel_capabilities;
use crate::performance_attribution::PerformanceAttribution;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
};
use crate::qwen3_5_core::model_math::dense_projection::{
    Qwen3_5DenseProjection, Qwen3_5ProjectionStorage,
};
use crate::row_batched_matmul::{
    RegisterOrderedWeight, RowBatchedMatmulKernels, RowBatchedProjectionError,
    register_ordered_matmul_any_rows,
};
use runtime_integration::MlxRuntimeError;
use text_generation::execution_error::TextExecutionError;

impl Qwen3_5ModelBase {
    /// Whether this GPU runs the row-batched matrix-unit kernels: the ones a
    /// register-ordered projection multiplies with, and the ones a draft
    /// tree's row-exact attention is built on.
    pub fn runs_row_batched_matrix_unit_kernels(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> bool {
        worker_process_kernel_capabilities(&self.runtime, performance_attribution)
            .is_custom_kernel_supported(crate::kernel_capability::ROW_BATCHED_MATMUL)
    }

    /// Declares the register-ordered kernels without converting any of the
    /// model's own projections, for a model that hosts weights already in
    /// register order (a drafter's) beside projections it keeps as they are.
    pub fn declare_register_ordered_kernels(&mut self) -> Result<(), TextExecutionError> {
        self.optional_accelerators.register_ordered_matmul_kernels =
            Some(RowBatchedMatmulKernels::declare()?);
        Ok(())
    }

    /// Converts every dense 4-bit, group-64 projection with bf16 scales, and
    /// an output head not tied to the embedding, into register order, once
    /// this worker's capability probe has run the kernel on this GPU. Each
    /// projection's MLX arrays are released as soon as its converted copy
    /// exists, so the load never holds more than one extra projection. Call
    /// it only on a resident model whose weights are all materialized: it
    /// drops the loaded shard maps, which would otherwise keep every original
    /// projection alive. Returns how many projections were converted, which
    /// is 0 for a model with none of that packing (an 8-bit or mixed build,
    /// or a mixture-of-experts model whose attention is not 4-bit), or `None`
    /// on a GPU without M5 matrix units, where nothing can convert.
    pub fn convert_projections_to_register_order(
        &mut self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<usize>, TextExecutionError> {
        if !self.runs_row_batched_matrix_unit_kernels(performance_attribution) {
            return Ok(None);
        }
        let kernels = RowBatchedMatmulKernels::declare()?;
        self.weights.release_model_shard_maps();
        let mut converted_count = 0;
        for layer_weights in &mut self.weights.decoder_layer_weights {
            for projection in layer_projections(
                &mut layer_weights.attention_weights,
                &mut layer_weights.mlp_weights,
            ) {
                converted_count += convert_one(&self.runtime, &kernels, projection)?;
            }
        }
        if !self.weights.has_tied_embeddings() {
            let head = &mut self.weights.language_model_head_weights;
            converted_count += convert_one(&self.runtime, &kernels, head)?;
        }
        self.optional_accelerators.register_ordered_matmul_kernels = Some(kernels);
        Ok(Some(converted_count))
    }

    /// `activations` [..., input] through a register-ordered projection. More
    /// rows than one dispatch takes (a prompt chunk) go 128 at a time.
    pub fn register_ordered_projection(
        &self,
        activations: &MlxArray,
        weight: &RegisterOrderedWeight,
    ) -> Result<MlxArray, TextExecutionError> {
        let kernels = self
            .optional_accelerators
            .register_ordered_matmul_kernels
            .as_ref()
            .ok_or(TextExecutionError::InvalidInput {
                description: "a register-ordered projection needs the kernels that converted it",
            })?;
        register_ordered_matmul_any_rows(&self.runtime, kernels, activations, weight)
            .map_err(projection_failure)
    }
}

/// Every dense projection of one decoder layer; a mixture-of-experts layer
/// gives only its attention projections, since expert stacks stay in MLX's
/// layout for the gathered expert kernels.
fn layer_projections<'layer>(
    attention_weights: &'layer mut Qwen3_5AttentionWeights,
    feed_forward_weights: &'layer mut Qwen3_5DecoderFeedForwardWeights,
) -> Vec<&'layer mut Qwen3_5DenseProjection> {
    let mut projections: Vec<&mut Qwen3_5DenseProjection> = match attention_weights {
        Qwen3_5AttentionWeights::Linear(linear) => vec![
            &mut linear.input_queries_keys_values_projection,
            &mut linear.output_gate_projection,
            &mut linear.update_rate_projection,
            &mut linear.decay_interval_projection,
            &mut linear.output_projection,
        ],
        Qwen3_5AttentionWeights::Full(full) => vec![
            &mut full.query_projection,
            &mut full.key_projection,
            &mut full.value_projection,
            &mut full.output_projection,
        ],
    };
    if let Qwen3_5DecoderFeedForwardWeights::Dense(mlp) = feed_forward_weights {
        projections.push(&mut mlp.gate_projection);
        projections.push(&mut mlp.up_projection);
        projections.push(&mut mlp.down_projection);
    }
    projections
}

/// Replaces `projection` with its register-ordered copy when the kernel
/// reads its packing and shape: 1 when converted, 0 when left as it was (the
/// 48-wide DeltaNet gates, 8-bit or group-32 modules, bf16 weights).
fn convert_one(
    runtime: &runtime_integration::MlxRuntime,
    kernels: &RowBatchedMatmulKernels,
    projection: &mut Qwen3_5DenseProjection,
) -> Result<usize, TextExecutionError> {
    let Some(Qwen3_5AffineWeights::Quantized {
        packed_weight,
        quantization_scales,
        quantization_biases,
        quantization_bits: 4,
        quantization_group_size: 64,
    }) = projection.bound()
    else {
        return Ok(0);
    };
    let converted = match RegisterOrderedWeight::convert(
        runtime,
        kernels,
        packed_weight,
        quantization_scales,
        quantization_biases,
    ) {
        Ok(converted) => converted,
        Err(RowBatchedProjectionError::Refused(_))
        | Err(RowBatchedProjectionError::UnsupportedQuantization { .. }) => return Ok(0),
        Err(conversion_error) => return Err(projection_failure(conversion_error)),
    };
    runtime.evaluate_arrays(&[&converted.codes, &converted.scale_bias_words])?;
    projection.storage = Qwen3_5ProjectionStorage::RegisterOrdered(converted);
    Ok(1)
}

pub(super) fn projection_failure(
    projection_error: RowBatchedProjectionError,
) -> TextExecutionError {
    TextExecutionError::from(MlxRuntimeError::RuntimeOperation {
        operation: "project rows through the register-ordered matmul",
        description: projection_error.to_string(),
    })
}

impl Qwen3_5LinearWeights for Qwen3_5DenseProjection {
    fn project_rows(
        &self,
        model: &Qwen3_5ModelBase,
        activations: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        let base_output = match &self.storage {
            Qwen3_5ProjectionStorage::Bound(bound_weights) => {
                model.bound_linear(activations, bound_weights)?
            }
            Qwen3_5ProjectionStorage::RegisterOrdered(converted) => {
                model.register_ordered_projection(activations, converted)?
            }
        };
        model.with_layer_add_on(&self.module_name, activations, base_output)
    }
}

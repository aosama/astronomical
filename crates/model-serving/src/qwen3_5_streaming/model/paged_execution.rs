//! MLX gathered execution against one Rust-streamed expert layer.

use astronomical_mlx_c_rust::MlxArray;
use astronomical_runtime_integration::MlxRuntime;

use crate::expert_paging::QuantizedExpertPageManifest;
use crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::sparse_experts::{
    ExpertAssignmentOrder, StackedExpertProjection, gather_expert_projection,
};
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

use super::super::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;
use super::cached_plus_streamed_page_route;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::output_combination;
use super::routing::{
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_unsorted_expert_weighted_sum,
};
use crate::sparse_experts;

/// Per-assignment routed expert rows before any weighted reduction.
pub(super) enum RoutedExpertAssignmentOutputs {
    /// Sorted gather rows plus the inverse permutation the sorted reduction needs.
    Sorted {
        sorted_outputs: MlxArray,
        inverse_order: MlxArray,
    },
    /// Original-order assignment rows.
    Unsorted { assignment_outputs: MlxArray },
}

impl Qwen3_5Model {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_paged_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        page_manifest: &QuantizedExpertPageManifest,
        selected_indices: &MlxArray,
        sorted_unique_expert_ids: &[usize],
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let paged_output = performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |performance_attribution| {
                let page_slot_indices =
                    cached_plus_streamed_page_route::qwen3_5_moe_remap_expert_page_slots(
                        &self.runtime,
                        selected_indices,
                        sorted_unique_expert_ids,
                        page_manifest,
                    )?;
                let sparse_output = self.forward_moe_with_streamed_weights(
                    hidden_states,
                    paged_expert_weights,
                    &page_slot_indices,
                    selected_scores,
                    performance_attribution,
                )?;
                self.combine_paged_sparse_and_shared_outputs(
                    hidden_states,
                    mixture_of_experts_weights,
                    &sparse_output,
                    should_use_compiled_elementwise_graphs,
                )
            },
        )?;
        performance_attribution.record_counter(
            PerformanceCounter::RustStreamedExpertProjectionGraphCount,
            3,
        );
        Ok(paged_output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_moe_with_streamed_weights(
        &self,
        hidden_states: &MlxArray,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_expert_indices: &MlxArray,
        selected_scores: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let selected_outputs = self.streamed_expert_assignment_outputs(
            hidden_states,
            paged_expert_weights,
            selected_expert_indices,
            performance_attribution,
        )?;
        let sparse_output = match selected_outputs {
            RoutedExpertAssignmentOutputs::Sorted {
                sorted_outputs,
                inverse_order,
            } => qwen3_5_moe_sorted_expert_weighted_sum(
                &self.runtime,
                self.sorted_expert_weighted_sum_kernel()?,
                &sorted_outputs,
                &inverse_order,
                selected_scores,
            )?,
            RoutedExpertAssignmentOutputs::Unsorted { assignment_outputs } => {
                qwen3_5_moe_unsorted_expert_weighted_sum(
                    &self.runtime,
                    &assignment_outputs,
                    selected_scores,
                )?
            }
        };
        Ok(sparse_output)
    }

    /// Runs the routed expert chain through one page and returns one row per
    /// assignment, before any weighted reduction.
    ///
    /// Issue #373: mixed serving gathers the covered assignments from the warm
    /// table and the missing assignments from the streamed page, then restores
    /// the original assignment order so ONE weighted reduction consumes the
    /// same values in the same order as the single-page path. Returning rows
    /// instead of a reduced output is what makes that possible.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn streamed_expert_assignment_outputs(
        &self,
        hidden_states: &MlxArray,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_expert_indices: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<RoutedExpertAssignmentOutputs, Qwen3_5ExecutionError> {
        let expanded_states = self.runtime.expand_dims(hidden_states, -2)?;
        let expanded_states = self.runtime.expand_dims(&expanded_states, -3)?;
        let sorted_assignments = if sparse_experts::should_use_sorted_expert_reduction(
            selected_expert_indices.element_count(),
            self.sorted_expert_weighted_sum_kernel.is_some(),
        ) {
            Some(qwen3_5_moe_sort_expert_assignments(
                &self.runtime,
                &expanded_states,
                selected_expert_indices,
            )?)
        } else {
            None
        };
        let (expert_inputs, expert_indices, indices_are_sorted) = match sorted_assignments.as_ref()
        {
            Some((sorted_states, sorted_indices, _)) => (sorted_states, sorted_indices, true),
            None => (&expanded_states, selected_expert_indices, false),
        };
        let gate_activation_rotation = paged_expert_weights.gate_activation_rotation;
        let up_activation_rotation = paged_expert_weights.up_activation_rotation;
        let shared_gate_up_activations = if gate_activation_rotation
            == Jangtq2CodebookRotation::Hadamard32
            && up_activation_rotation == Jangtq2CodebookRotation::Hadamard32
        {
            // Both projections use the same input and transform, so build
            // one rotated tensor and share it instead of repeating the
            // blockwise Hadamard work for gate and up independently.
            Some(
                crate::jangtq2_qwen35::mlx_decode::rotate_activations_hadamard32(
                    &self.runtime,
                    expert_inputs,
                )?,
            )
        } else {
            None
        };
        let gate_up_activations = shared_gate_up_activations.as_ref().unwrap_or(expert_inputs);
        let gate_projection_rotation = if shared_gate_up_activations.is_some() {
            Jangtq2CodebookRotation::NoRotation
        } else {
            gate_activation_rotation
        };
        let up_projection_rotation = if shared_gate_up_activations.is_some() {
            Jangtq2CodebookRotation::NoRotation
        } else {
            up_activation_rotation
        };
        let selected_up = self.streamed_expert_linear(
            gate_up_activations,
            &paged_expert_weights.up_projection,
            up_projection_rotation,
            expert_indices,
            indices_are_sorted,
            performance_attribution,
        )?;
        let selected_gate = self.streamed_expert_linear(
            gate_up_activations,
            &paged_expert_weights.gate_projection,
            gate_projection_rotation,
            expert_indices,
            indices_are_sorted,
            performance_attribution,
        )?;
        let selected_activated = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &selected_gate,
            &selected_up,
        )?;
        let selected_outputs = self.streamed_expert_linear(
            &selected_activated,
            &paged_expert_weights.down_projection,
            paged_expert_weights.down_activation_rotation,
            expert_indices,
            indices_are_sorted,
            performance_attribution,
        )?;
        if indices_are_sorted {
            let Some((_, _, inverse_order)) = sorted_assignments else {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "sorted routed assignments lost their inverse order",
                });
            };
            Ok(RoutedExpertAssignmentOutputs::Sorted {
                sorted_outputs: selected_outputs,
                inverse_order,
            })
        } else {
            let assignment_outputs = self.runtime.squeeze_axis(&selected_outputs, -2)?;
            Ok(RoutedExpertAssignmentOutputs::Unsorted { assignment_outputs })
        }
    }

    fn streamed_expert_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights,
        activation_rotation: crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation,
        selected_expert_indices: &MlxArray,
        are_expert_indices_sorted: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
        // Streamed pages and resident layers intentionally converge here on the
        // same canonical operation. Paging decides which arrays are alive; this
        // adapter only describes their matrix layout and assignment ordering.
        let assignment_order = if are_expert_indices_sorted {
            ExpertAssignmentOrder::SortedByExpert
        } else {
            ExpertAssignmentOrder::Original
        };
        let rotated_activation_owner = match activation_rotation {
            crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation => None,
            crate::jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32 => Some(
                crate::jangtq2_qwen35::mlx_decode::rotate_activations_hadamard32(
                    &self.runtime,
                    activations,
                )?,
            ),
        };
        let projection_activations = rotated_activation_owner.as_ref().unwrap_or(activations);
        match affine_weights {
            Qwen3_5AffineWeights::Jangtq2Codebook {
                packed_code_words,
                row_scales,
                codebook_bit_width,
            } => {
                if let Some(fused_kernel) = self.fused_jangtq2_gathered_projection_kernel.as_ref() {
                    if let Some(activation_rows) = decode_token_activation_rows(
                        &self.runtime,
                        projection_activations,
                        selected_expert_indices,
                        usize::try_from(self.config.experts_per_token()).unwrap_or(0),
                    )? {
                        // One decode token: multiply straight from the packed
                        // codes. The kernel returns `[assignments, rows]`;
                        // restore gather_mm's `[..., assignments, 1, rows]`
                        // layout so the SwiGLU and reduction stay unchanged.
                        let projected = performance_attribution.measure_operation(
                            PerformanceOperation::DecodeFusedJangtq2GatheredProjection,
                            |_| {
                                let flat_expert_indices = self.runtime.reshape(
                                    selected_expert_indices,
                                    &[selected_expert_indices.element_count() as i32],
                                )?;
                                fused_kernel.apply(
                                    &self.runtime,
                                    &activation_rows,
                                    &flat_expert_indices,
                                    packed_code_words,
                                    row_scales,
                                    *codebook_bit_width,
                                )
                            },
                        )?;
                        let mut gathered_shape = selected_expert_indices.shape();
                        gathered_shape.push(1);
                        gathered_shape.push(packed_code_words.shape()[1]);
                        return Ok(self.runtime.reshape(&projected, &gathered_shape)?);
                    }
                }
                // Multi-token prefill (or a worker without the fused kernel):
                // decode a transient bfloat16 copy and gather through
                // gather_mm, whose matrix multiply amortizes the decode over
                // the chunk's tokens. The copy dies with this forward.
                let decoded_weight = performance_attribution.measure_operation(
                    PerformanceOperation::Jangtq2CodebookPageDecode,
                    |_| {
                        crate::jangtq2_qwen35::mlx_decode::decode_expert_weight_rows(
                            &self.runtime,
                            packed_code_words,
                            row_scales,
                            *codebook_bit_width,
                        )
                    },
                )?;
                let transposed_weights =
                    self.runtime.transpose_axes(&decoded_weight, &[0, 2, 1])?;
                Ok(gather_expert_projection(
                    &self.runtime,
                    projection_activations,
                    StackedExpertProjection::Dense {
                        transposed_weights: &transposed_weights,
                    },
                    selected_expert_indices,
                    assignment_order,
                    performance_attribution,
                )?)
            }
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                // The transpose is a lazy view from checkpoint linear layout to
                // the `[expert, input, output]` layout expected by gather_mm.
                let transposed_weights = self.runtime.transpose_axes(weight, &[0, 2, 1])?;
                Ok(gather_expert_projection(
                    &self.runtime,
                    projection_activations,
                    StackedExpertProjection::Dense {
                        transposed_weights: &transposed_weights,
                    },
                    selected_expert_indices,
                    assignment_order,
                    performance_attribution,
                )?)
            }
            Qwen3_5AffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => Ok(gather_expert_projection(
                &self.runtime,
                projection_activations,
                // Pass the streamed packed page directly to gather_qmm. Taking
                // selected matrices first would recreate the memory expansion
                // that gathered execution exists to avoid.
                StackedExpertProjection::Affine {
                    packed_weights: packed_weight,
                    scales: quantization_scales,
                    biases: quantization_biases,
                    group_size: *quantization_group_size,
                    bits: *quantization_bits,
                },
                selected_expert_indices,
                assignment_order,
                performance_attribution,
            )?),
        }
    }

    pub(super) fn combine_paged_sparse_and_shared_outputs(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        sparse_expert_output: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let shared_gate = self.quantized_linear(
            hidden_states,
            &mixture_of_experts_weights.shared_expert_gate_projection,
        )?;
        let shared_up = self.quantized_linear(
            hidden_states,
            &mixture_of_experts_weights.shared_expert_up_projection,
        )?;
        let shared_activated =
            self.runtime
                .apply_compiled_swiglu(&self.compiled_swiglu, &shared_gate, &shared_up)?;
        let shared_output = self.quantized_linear(
            &shared_activated,
            &mixture_of_experts_weights.shared_expert_down_projection,
        )?;
        let shared_gate_logits = self.quantized_linear(
            hidden_states,
            &mixture_of_experts_weights.shared_expert_output_gate_projection,
        )?;
        if should_use_compiled_elementwise_graphs {
            return Ok(self
                .runtime
                .apply_compiled_sparse_shared_expert_combination(
                    &self.compiled_elementwise_graphs,
                    &sparse_expert_output,
                    &shared_output,
                    &shared_gate_logits,
                )?);
        }
        Ok(output_combination::combine_sparse_and_shared_experts(
            &self.runtime,
            sparse_expert_output,
            &shared_output,
            &shared_gate_logits,
        )?)
    }
}

/// Returns the activation rows a single decode token presents to the fused
/// gathered projection kernel, or `None` for a multi-token forward.
///
/// The unsorted decode path carries one activation row shared by every
/// assignment (`[1, 1, 1, width]`); the sorted path carries one row per
/// assignment (`[assignments, 1, width]`). Both flatten to `[rows, width]`
/// with `rows` equal to one or to the assignment count. A chunk of several
/// tokens has more assignments than one token routes, and keeps the
/// materializing path: the kernel is a matrix-vector product and the
/// chunk's matrix multiply belongs on `gather_mm`.
pub(super) fn decode_token_activation_rows(
    runtime: &MlxRuntime,
    projection_activations: &MlxArray,
    selected_expert_indices: &MlxArray,
    experts_per_token: usize,
) -> Result<Option<MlxArray>, Qwen3_5ExecutionError> {
    let activation_shape = projection_activations.shape();
    let Some(&input_width) = activation_shape.last() else {
        return Ok(None);
    };
    if input_width <= 0 {
        return Ok(None);
    }
    let activation_row_count = projection_activations.element_count() / input_width as usize;
    let assignment_count = selected_expert_indices.element_count();
    if assignment_count == 0 || assignment_count > experts_per_token {
        return Ok(None);
    }
    if activation_row_count != 1 && activation_row_count != assignment_count {
        return Ok(None);
    }
    Ok(Some(runtime.reshape(
        projection_activations,
        &[activation_row_count as i32, input_width],
    )?))
}

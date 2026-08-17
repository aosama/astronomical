//! MLX gathered execution against one Rust-streamed expert layer.

use astronomical_runtime_integration::MlxArray;

use crate::expert_paging::{ExpertPageRoutePartition, QuantizedExpertPageManifest};
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::sparse_experts::{
    ExpertAssignmentOrder, StackedExpertProjection, gather_expert_projection,
};
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

use super::super::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::output_combination::combine_sparse_and_shared_experts;
use super::routing::{
    MINIMUM_SORTED_EXPERT_ASSIGNMENTS, qwen3_5_moe_sort_expert_assignments,
    qwen3_5_moe_sorted_expert_weighted_sum, qwen3_5_moe_unsorted_expert_weighted_sum,
};
use super::split_page_route::{Qwen3_5MoESplitPageRoute, qwen3_5_moe_remap_expert_page_slots};

impl Qwen3_5Model {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_paged_target_verification_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        page_manifest: &QuantizedExpertPageManifest,
        selected_indices: &MlxArray,
        sorted_unique_expert_ids: &[usize],
        selected_scores: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |performance_attribution| {
                let page_slot_indices = qwen3_5_moe_remap_expert_page_slots(
                    &self.runtime,
                    selected_indices,
                    sorted_unique_expert_ids,
                    page_manifest,
                )?;
                let hidden_shape = hidden_states.shape();
                let batch_size = hidden_shape[0];
                let token_count = hidden_shape[1];
                let hidden_dimension = hidden_shape[2];
                let experts_per_token = page_slot_indices.shape()[2];
                let flattened_states = self
                    .runtime
                    .reshape(hidden_states, &[batch_size * token_count, hidden_dimension])?;
                let flattened_indices = self.runtime.reshape(
                    &page_slot_indices,
                    &[batch_size * token_count, experts_per_token],
                )?;
                let expanded_states = self.runtime.expand_dims(&flattened_states, -2)?;
                let expanded_states = self.runtime.expand_dims(&expanded_states, -3)?;
                let selected_up = self.streamed_expert_linear(
                    &expanded_states,
                    &paged_expert_weights.up_projection,
                    &flattened_indices,
                    false,
                    performance_attribution,
                )?;
                let selected_gate = self.streamed_expert_linear(
                    &expanded_states,
                    &paged_expert_weights.gate_projection,
                    &flattened_indices,
                    false,
                    performance_attribution,
                )?;
                let activated = self.runtime.apply_compiled_swiglu(
                    &self.compiled_swiglu,
                    &selected_gate,
                    &selected_up,
                )?;
                let selected_outputs = self.streamed_expert_linear(
                    &activated,
                    &paged_expert_weights.down_projection,
                    &flattened_indices,
                    false,
                    performance_attribution,
                )?;
                let selected_outputs = self.runtime.squeeze_axis(&selected_outputs, -2)?;
                let selected_outputs = self.runtime.reshape(
                    &selected_outputs,
                    &[batch_size, token_count, experts_per_token, hidden_dimension],
                )?;
                let sparse_output = qwen3_5_moe_unsorted_expert_weighted_sum(
                    &self.runtime,
                    &selected_outputs,
                    selected_scores,
                )?;
                self.combine_paged_sparse_and_shared_outputs(
                    hidden_states,
                    mixture_of_experts_weights,
                    &sparse_output,
                    false,
                )
            },
        )
    }

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
                let page_slot_indices = qwen3_5_moe_remap_expert_page_slots(
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

    /// Executes one route from a retained page plus an operation-local miss page.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_split_paged_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        retained_expert_weights: &Qwen3_5PagedExpertWeights,
        retained_page_manifest: &QuantizedExpertPageManifest,
        missing_expert_weights: &Qwen3_5PagedExpertWeights,
        missing_page_manifest: &QuantizedExpertPageManifest,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        route_partition: &ExpertPageRoutePartition,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let paged_output = performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |performance_attribution| {
                let split_page_route = Qwen3_5MoESplitPageRoute::build(
                    &self.runtime,
                    selected_indices,
                    selected_scores,
                    route_partition,
                    retained_page_manifest,
                    missing_page_manifest,
                )?;
                let retained_sparse_output = self.forward_moe_with_streamed_weights(
                    hidden_states,
                    retained_expert_weights,
                    &split_page_route.retained_page_slot_indices,
                    &split_page_route.retained_scores,
                    performance_attribution,
                )?;
                let missing_sparse_output = self.forward_moe_with_streamed_weights(
                    hidden_states,
                    missing_expert_weights,
                    &split_page_route.missing_page_slot_indices,
                    &split_page_route.missing_scores,
                    performance_attribution,
                )?;
                let sparse_output = self
                    .runtime
                    .add(&retained_sparse_output, &missing_sparse_output)?;
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
            6,
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
        let expanded_states = self.runtime.expand_dims(hidden_states, -2)?;
        let expanded_states = self.runtime.expand_dims(&expanded_states, -3)?;
        let sorted_assignments =
            if selected_expert_indices.element_count() >= MINIMUM_SORTED_EXPERT_ASSIGNMENTS {
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
        let selected_up = self.streamed_expert_linear(
            expert_inputs,
            &paged_expert_weights.up_projection,
            expert_indices,
            indices_are_sorted,
            performance_attribution,
        )?;
        let selected_gate = self.streamed_expert_linear(
            expert_inputs,
            &paged_expert_weights.gate_projection,
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
            expert_indices,
            indices_are_sorted,
            performance_attribution,
        )?;
        let sparse_output = match sorted_assignments.as_ref() {
            Some((_, _, inverse_order)) => qwen3_5_moe_sorted_expert_weighted_sum(
                &self.runtime,
                self.sorted_expert_weighted_sum_kernel()?,
                &selected_outputs,
                inverse_order,
                selected_scores,
            )?,
            None => {
                let selected_outputs = self.runtime.squeeze_axis(&selected_outputs, -2)?;
                qwen3_5_moe_unsorted_expert_weighted_sum(
                    &self.runtime,
                    &selected_outputs,
                    selected_scores,
                )?
            }
        };
        Ok(sparse_output)
    }

    fn streamed_expert_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights,
        selected_expert_indices: &MlxArray,
        are_expert_indices_sorted: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;
        // Streamed pages and resident layers intentionally converge here on the
        // same canonical operation. Paging decides which arrays are alive; this
        // adapter only describes their matrix layout and assignment ordering.
        let assignment_order = if are_expert_indices_sorted {
            ExpertAssignmentOrder::SortedByExpert
        } else {
            ExpertAssignmentOrder::Original
        };
        match affine_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                // The transpose is a lazy view from checkpoint linear layout to
                // the `[expert, input, output]` layout expected by gather_mm.
                let transposed_weights = self.runtime.transpose_axes(weight, &[0, 2, 1])?;
                Ok(gather_expert_projection(
                    &self.runtime,
                    activations,
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
                activations,
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
        Ok(combine_sparse_and_shared_experts(
            &self.runtime,
            sparse_expert_output,
            &shared_output,
            &shared_gate_logits,
        )?)
    }
}

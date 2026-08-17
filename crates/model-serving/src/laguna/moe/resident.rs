//! Resident Laguna Mixture-of-Experts: gathered SwiGLU plus shared expert.

use astronomical_runtime_integration::{MlxArray, MlxCompiledSwiGlu, MlxMetalKernel, MlxRuntime};

use crate::laguna::artifacts::{LagunaExpertProjection, LagunaLayerTensorRole};
use crate::laguna::normalization::LagunaMoeDescriptor;
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};
use crate::sparse_experts::{
    router_weighted_expert_inputs, sort_expert_assignments, sorted_expert_weighted_sum,
    unsorted_expert_weighted_sum,
};

use crate::laguna::LagunaNativeWeights;

use super::router::route_laguna_experts;
use crate::laguna::model::LagunaExecutionError;

const MINIMUM_SORTED_EXPERT_ASSIGNMENTS: usize = 64;

/// Routes, executes stacked experts, scales the reduction, and adds the shared expert.
#[allow(clippy::too_many_arguments)]
pub(in crate::laguna) fn forward_resident_mixture_of_experts(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    moe_descriptor: &LagunaMoeDescriptor,
    layer_index: usize,
    router_logit_softcap: f64,
    sorted_expert_reduction_kernel: &MlxMetalKernel,
    compiled_swiglu: &MlxCompiledSwiGlu,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    performance_attribution.measure_operation(PerformanceOperation::MlpForwardSpan, |attribution| {
        attribution.measure_operation(
            PerformanceOperation::ResidentMoeGraphConstruction,
            |attribution| {
                forward_resident_mixture_of_experts_inner(
                    runtime,
                    hidden_states,
                    weights,
                    moe_descriptor,
                    layer_index,
                    router_logit_softcap,
                    sorted_expert_reduction_kernel,
                    compiled_swiglu,
                    attribution,
                )
            },
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn forward_resident_mixture_of_experts_inner(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    moe_descriptor: &LagunaMoeDescriptor,
    layer_index: usize,
    router_logit_softcap: f64,
    sorted_expert_reduction_kernel: &MlxMetalKernel,
    compiled_swiglu: &MlxCompiledSwiGlu,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    let router_logits = weights
        .linear(layer_index, LagunaLayerTensorRole::Router)?
        .project(runtime, hidden_states)?;
    let correction_bias =
        weights.optional_layer(layer_index, LagunaLayerTensorRole::RouterCorrectionBias);
    let (selected_indices, selected_scores) = route_laguna_experts(
        runtime,
        &router_logits,
        correction_bias,
        moe_descriptor,
        router_logit_softcap,
        performance_attribution,
    )?;

    let routed_output = gathered_routed_swiglu(
        runtime,
        hidden_states,
        weights,
        layer_index,
        &selected_indices,
        &selected_scores,
        moe_descriptor.applies_router_weight_on_input(),
        sorted_expert_reduction_kernel,
        compiled_swiglu,
        performance_attribution,
    )?;
    let scaled_routed_output = runtime.multiply_scalar(
        &routed_output,
        moe_descriptor.routed_scaling_factor() as f32,
    )?;

    if moe_descriptor.shared_expert_intermediate_size() == 0 {
        return Ok(scaled_routed_output);
    }
    let shared_output = performance_attribution.measure_operation(
        PerformanceOperation::SharedExpertExecution,
        |_| {
            shared_expert_swiglu(
                runtime,
                hidden_states,
                weights,
                layer_index,
                Some(compiled_swiglu),
            )
        },
    )?;
    Ok(runtime.add(&scaled_routed_output, &shared_output)?)
}

#[allow(clippy::too_many_arguments)]
fn gathered_routed_swiglu(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    layer_index: usize,
    selected_indices: &MlxArray,
    selected_scores: &MlxArray,
    applies_router_weight_on_input: bool,
    sorted_expert_reduction_kernel: &MlxMetalKernel,
    compiled_swiglu: &MlxCompiledSwiGlu,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    // Input-side weighting needs one hidden row per assignment, so skip the
    // 64-assignment sort that expects a shared hidden row per token.
    let should_sort = !applies_router_weight_on_input
        && selected_indices.element_count() >= MINIMUM_SORTED_EXPERT_ASSIGNMENTS;
    let gather_states = if applies_router_weight_on_input {
        router_weighted_expert_inputs(runtime, hidden_states, selected_scores)?
    } else {
        expand_for_gather(runtime, hidden_states)?
    };
    let sorted_assignments = if should_sort {
        Some(sort_expert_assignments(
            runtime,
            &gather_states,
            selected_indices,
            performance_attribution,
        )?)
    } else {
        None
    };
    let (expert_input_states, expert_indices, are_indices_sorted) = match &sorted_assignments {
        Some(sorted) => (&sorted.sorted_states, &sorted.sorted_indices, true),
        None => (&gather_states, selected_indices, false),
    };

    // Each project_gathered call attributes its own actual matrix operation.
    // Do not wrap this whole SwiGLU block in the same attribution operation: that
    // would double-count nested projection spans and obscure which graph was built.
    let (gate, up) = if let Some(fused_gate_up) = weights.fused_routed_gate_up(layer_index) {
        let fused_output = fused_gate_up.project_gathered(
            runtime,
            expert_input_states,
            expert_indices,
            are_indices_sorted,
            performance_attribution,
        )?;
        crate::laguna::model::LagunaBoundLinear::split_fused_gate_up(runtime, &fused_output)?
    } else {
        let gate = weights
            .linear(
                layer_index,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
            )?
            .project_gathered(
                runtime,
                expert_input_states,
                expert_indices,
                are_indices_sorted,
                performance_attribution,
            )?;
        let up = weights
            .linear(
                layer_index,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Up),
            )?
            .project_gathered(
                runtime,
                expert_input_states,
                expert_indices,
                are_indices_sorted,
                performance_attribution,
            )?;
        (gate, up)
    };
    let hidden_product = runtime.apply_compiled_swiglu(compiled_swiglu, &gate, &up)?;
    let selected_outputs = weights
        .linear(
            layer_index,
            LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Down),
        )?
        .project_gathered(
            runtime,
            &hidden_product,
            expert_indices,
            are_indices_sorted,
            performance_attribution,
        )?;

    if let Some(sorted) = sorted_assignments {
        return Ok(sorted_expert_weighted_sum(
            runtime,
            sorted_expert_reduction_kernel,
            &selected_outputs,
            &sorted.inverse_order,
            selected_scores,
            performance_attribution,
        )?);
    }
    let selected_outputs = runtime.squeeze_axis(&selected_outputs, -2)?;
    if applies_router_weight_on_input {
        // Input-side scores still culminate in an expert-combination reduction.
        // Attribute that boundary even though no second score multiply is needed.
        return Ok(performance_attribution
            .measure_operation(PerformanceOperation::ExpertWeightedReduction, |_| {
                runtime.sum_axis(&selected_outputs, -2, false)
            })?);
    }
    Ok(unsorted_expert_weighted_sum(
        runtime,
        &selected_outputs,
        selected_scores,
        performance_attribution,
    )?)
}

pub(in crate::laguna) fn shared_expert_swiglu(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    layer_index: usize,
    compiled_swiglu: Option<&MlxCompiledSwiGlu>,
) -> Result<MlxArray, LagunaExecutionError> {
    let (gate, up) = if let Some(fused_gate_up) = weights.fused_shared_gate_up(layer_index) {
        let fused_output = fused_gate_up.project(runtime, hidden_states)?;
        crate::laguna::model::LagunaBoundLinear::split_fused_gate_up(runtime, &fused_output)?
    } else {
        let gate = weights
            .linear(
                layer_index,
                LagunaLayerTensorRole::SharedExpert(LagunaExpertProjection::Gate),
            )?
            .project(runtime, hidden_states)?;
        let up = weights
            .linear(
                layer_index,
                LagunaLayerTensorRole::SharedExpert(LagunaExpertProjection::Up),
            )?
            .project(runtime, hidden_states)?;
        (gate, up)
    };
    let hidden_product = if let Some(compiled_swiglu) = compiled_swiglu {
        runtime.apply_compiled_swiglu(compiled_swiglu, &gate, &up)?
    } else {
        let activated_gate = runtime.silu(&gate)?;
        runtime.multiply(&activated_gate, &up)?
    };
    weights
        .linear(
            layer_index,
            LagunaLayerTensorRole::SharedExpert(LagunaExpertProjection::Down),
        )?
        .project(runtime, &hidden_product)
}

fn expand_for_gather(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
) -> Result<MlxArray, LagunaExecutionError> {
    let expanded = runtime.expand_dims(hidden_states, -2)?;
    Ok(runtime.expand_dims(&expanded, -3)?)
}

//! Gathered SwiGLU over one streamed Laguna expert page.

use astronomical_runtime_integration::{MlxArray, MlxMetalKernel, MlxRuntime};

use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};
use crate::sparse_experts::{
    router_weighted_expert_inputs, sort_expert_assignments, sorted_expert_weighted_sum,
    unsorted_expert_weighted_sum,
};

use crate::laguna::model::LagunaBoundLinear;

use super::error::LagunaPagingError;
use super::weight_page::LagunaExpertWeightPage;

const MINIMUM_SORTED_EXPERT_ASSIGNMENTS: usize = 64;

/// Executes routed SwiGLU on one streamed page after remapping global expert IDs.
#[allow(clippy::too_many_arguments)]
pub fn forward_paged_routed_swiglu(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    expert_page: &LagunaExpertWeightPage,
    selected_indices: &MlxArray,
    selected_scores: &MlxArray,
    applies_router_weight_on_input: bool,
    sorted_expert_reduction_kernel: &MlxMetalKernel,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaPagingError> {
    performance_attribution.measure_operation(
        PerformanceOperation::PagedMoeGraphConstruction,
        |performance_attribution| {
            forward_paged_routed_swiglu_inner(
                runtime,
                hidden_states,
                expert_page,
                selected_indices,
                selected_scores,
                applies_router_weight_on_input,
                sorted_expert_reduction_kernel,
                performance_attribution,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_paged_routed_swiglu_inner(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    expert_page: &LagunaExpertWeightPage,
    selected_indices: &MlxArray,
    selected_scores: &MlxArray,
    applies_router_weight_on_input: bool,
    sorted_expert_reduction_kernel: &MlxMetalKernel,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaPagingError> {
    let page_slot_indices =
        remap_global_expert_ids_to_page_slots(runtime, expert_page, selected_indices)?;
    let should_sort = !applies_router_weight_on_input
        && selected_indices.element_count() >= MINIMUM_SORTED_EXPERT_ASSIGNMENTS;
    let gather_states = if applies_router_weight_on_input {
        router_weighted_expert_inputs(runtime, hidden_states, selected_scores)?
    } else {
        let expanded = runtime.expand_dims(hidden_states, -2)?;
        runtime.expand_dims(&expanded, -3)?
    };
    let sorted_assignments = if should_sort {
        Some(sort_expert_assignments(
            runtime,
            &gather_states,
            &page_slot_indices,
            performance_attribution,
        )?)
    } else {
        None
    };
    let (expert_input_states, expert_indices, are_indices_sorted) = match &sorted_assignments {
        Some(sorted) => (&sorted.sorted_states, &sorted.sorted_indices, true),
        None => (&gather_states, &page_slot_indices, false),
    };
    let selected_outputs = performance_attribution.measure_operation(
        PerformanceOperation::GatheredExpertExecution,
        |_| {
            gathered_page_swiglu(
                runtime,
                expert_page,
                expert_input_states,
                expert_indices,
                are_indices_sorted,
            )
        },
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
        return Ok(runtime.sum_axis(&selected_outputs, -2, false)?);
    }
    Ok(unsorted_expert_weighted_sum(
        runtime,
        &selected_outputs,
        selected_scores,
        performance_attribution,
    )?)
}

fn remap_global_expert_ids_to_page_slots(
    runtime: &MlxRuntime,
    expert_page: &LagunaExpertWeightPage,
    selected_indices: &MlxArray,
) -> Result<MlxArray, LagunaPagingError> {
    let page_slot_lookup = runtime.array_from_u32(
        &expert_page.manifest().page_slot_by_global_expert_id,
        &[
            i32::try_from(expert_page.manifest().page_slot_by_global_expert_id.len()).map_err(
                |_| LagunaPagingError::PageExecution {
                    description: "page slot lookup length exceeds MLX shape",
                },
            )?,
        ],
    )?;
    Ok(runtime.take_axis(&page_slot_lookup, selected_indices, 0)?)
}

fn gathered_page_swiglu(
    runtime: &MlxRuntime,
    expert_page: &LagunaExpertWeightPage,
    expert_input_states: &MlxArray,
    expert_indices: &MlxArray,
    are_indices_sorted: bool,
) -> Result<MlxArray, LagunaPagingError> {
    let (gate, up) = if let Some(fused_gate_up) = expert_page.fused_gate_up() {
        let fused_output = fused_gate_up.project_gathered(
            runtime,
            expert_input_states,
            expert_indices,
            are_indices_sorted,
        )?;
        LagunaBoundLinear::split_fused_gate_up(runtime, &fused_output)?
    } else {
        let (gate_projection, up_projection) =
            expert_page
                .split_gate_up()
                .ok_or(LagunaPagingError::PageExecution {
                    description: "Laguna expert page owns neither fused nor split gate/up storage",
                })?;
        (
            gate_projection.project_gathered(
                runtime,
                expert_input_states,
                expert_indices,
                are_indices_sorted,
            )?,
            up_projection.project_gathered(
                runtime,
                expert_input_states,
                expert_indices,
                are_indices_sorted,
            )?,
        )
    };
    let activated_gate = runtime.silu(&gate)?;
    let hidden_product = runtime.multiply(&activated_gate, &up)?;
    Ok(expert_page.down().project_gathered(
        runtime,
        &hidden_product,
        expert_indices,
        are_indices_sorted,
    )?)
}

impl From<astronomical_runtime_integration::MlxRuntimeError> for LagunaPagingError {
    fn from(_error: astronomical_runtime_integration::MlxRuntimeError) -> Self {
        Self::PageExecution {
            description: "an MLX runtime operation failed during Laguna page execution",
        }
    }
}

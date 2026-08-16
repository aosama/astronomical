//! Operation-local Laguna Mixture-of-Experts: stream a page, then gathered SwiGLU.

use astronomical_runtime_integration::{MlxArray, MlxMetalKernel, MlxRuntime};

use crate::expert_paging::ExpertWeightPage;
use crate::laguna::LagunaNativeWeights;
use crate::laguna::artifacts::LagunaLayerTensorRole;
use crate::laguna::model::LagunaExecutionError;
use crate::laguna::model::LagunaLastExpertForward;
use crate::laguna::normalization::LagunaMoeDescriptor;
use crate::laguna::paging::{
    LagunaSparseLayerPagingPlan, forward_paged_routed_swiglu, load_laguna_expert_page,
};
use crate::performance_attribution::PerformanceAttribution;

use super::resident::shared_expert_swiglu;
use super::router::route_laguna_experts;

/// Routes with the resident router, streams one page, scales, and adds the shared expert.
#[allow(clippy::too_many_arguments)]
pub(in crate::laguna) fn forward_paged_mixture_of_experts(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    moe_descriptor: &LagunaMoeDescriptor,
    layer_index: usize,
    sparse_layer_plan: &LagunaSparseLayerPagingPlan,
    router_logit_softcap: f64,
    sorted_expert_reduction_kernel: &MlxMetalKernel,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<
    (
        MlxArray,
        LagunaLastExpertForward,
        crate::laguna::paging::LagunaExpertWeightPage,
    ),
    LagunaExecutionError,
> {
    let (selected_indices, selected_scores) = route_laguna_layer_experts(
        runtime,
        weights,
        moe_descriptor,
        layer_index,
        hidden_states,
        router_logit_softcap,
        performance_attribution,
    )?;
    let token_count = token_count_from_hidden_states(hidden_states)?;
    let page_expert_ids = if token_count > 1 {
        (0..sparse_layer_plan.expert_capacity()).collect::<Vec<_>>()
    } else {
        sorted_unique_expert_ids(&selected_indices)?
    };
    if page_expert_ids.is_empty() {
        return Err(LagunaExecutionError::invalid_geometry(
            "a paged Laguna layer produced no routed expert identifiers",
        ));
    }
    let expert_page = load_laguna_expert_page(
        runtime,
        sparse_layer_plan,
        &page_expert_ids,
        performance_attribution,
    )?;
    let output = execute_paged_mixture_on_page(
        runtime,
        hidden_states,
        weights,
        moe_descriptor,
        layer_index,
        &expert_page,
        &selected_indices,
        &selected_scores,
        sorted_expert_reduction_kernel,
        performance_attribution,
    )?;
    // MLX graphs retain every input array until evaluation. Materialize the
    // operation-local output before this function returns so a non-retained page
    // can be released before the next sparse layer is loaded from SSD.
    performance_attribution.measure_operation(
        crate::performance_attribution::PerformanceOperation::PagedMoeOutputMaterializationSynchronizationWait,
        |_| runtime.evaluate_arrays(&[&output]),
    )?;
    let last_forward = if token_count > 1 {
        LagunaLastExpertForward::StreamedCompleteLayer {
            layer_count: 1,
            payload_bytes: expert_page.resident_payload_byte_count(),
        }
    } else {
        LagunaLastExpertForward::StreamedRoutedPage {
            layer_count: 1,
            payload_bytes: expert_page.resident_payload_byte_count(),
        }
    };
    Ok((output, last_forward, expert_page))
}

/// Executes routed SwiGLU plus the shared expert on an already-owned page.
#[allow(clippy::too_many_arguments)]
pub(in crate::laguna) fn execute_paged_mixture_on_page(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    moe_descriptor: &LagunaMoeDescriptor,
    layer_index: usize,
    expert_page: &crate::laguna::paging::LagunaExpertWeightPage,
    selected_indices: &MlxArray,
    selected_scores: &MlxArray,
    sorted_expert_reduction_kernel: &MlxMetalKernel,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    let routed_output = forward_paged_routed_swiglu(
        runtime,
        hidden_states,
        expert_page,
        selected_indices,
        selected_scores,
        moe_descriptor.applies_router_weight_on_input(),
        sorted_expert_reduction_kernel,
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
        crate::performance_attribution::PerformanceOperation::SharedExpertExecution,
        |_| shared_expert_swiglu(runtime, hidden_states, weights, layer_index, None),
    )?;
    Ok(runtime.add(&scaled_routed_output, &shared_output)?)
}

/// Routes one batch so a retained complete layer can execute without a disk read.
#[allow(clippy::too_many_arguments)]
pub(in crate::laguna) fn forward_retained_complete_mixture_of_experts(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &LagunaNativeWeights,
    moe_descriptor: &LagunaMoeDescriptor,
    layer_index: usize,
    expert_page: &crate::laguna::paging::LagunaExpertWeightPage,
    router_logit_softcap: f64,
    sorted_expert_reduction_kernel: &MlxMetalKernel,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    let (selected_indices, selected_scores) = route_laguna_layer_experts(
        runtime,
        weights,
        moe_descriptor,
        layer_index,
        hidden_states,
        router_logit_softcap,
        performance_attribution,
    )?;
    execute_paged_mixture_on_page(
        runtime,
        hidden_states,
        weights,
        moe_descriptor,
        layer_index,
        expert_page,
        &selected_indices,
        &selected_scores,
        sorted_expert_reduction_kernel,
        performance_attribution,
    )
}

/// Routes one sparse layer with the resident router.
pub(in crate::laguna) fn route_laguna_layer_experts(
    runtime: &MlxRuntime,
    weights: &LagunaNativeWeights,
    moe_descriptor: &LagunaMoeDescriptor,
    layer_index: usize,
    hidden_states: &MlxArray,
    router_logit_softcap: f64,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(MlxArray, MlxArray), LagunaExecutionError> {
    let router_logits = weights
        .linear(layer_index, LagunaLayerTensorRole::Router)?
        .project(runtime, hidden_states)?;
    let correction_bias =
        weights.optional_layer(layer_index, LagunaLayerTensorRole::RouterCorrectionBias);
    route_laguna_experts(
        runtime,
        &router_logits,
        correction_bias,
        moe_descriptor,
        router_logit_softcap,
        performance_attribution,
    )
}

pub(in crate::laguna) fn unique_routed_expert_ids(
    selected_indices: &MlxArray,
) -> Result<Vec<usize>, LagunaExecutionError> {
    sorted_unique_expert_ids(selected_indices)
}

fn token_count_from_hidden_states(hidden_states: &MlxArray) -> Result<usize, LagunaExecutionError> {
    let hidden_shape = hidden_states.shape();
    let token_axis = if hidden_shape.len() >= 2 {
        hidden_shape[hidden_shape.len() - 2]
    } else {
        hidden_shape.first().copied().unwrap_or(0)
    };
    usize::try_from(token_axis).map_err(|_| {
        LagunaExecutionError::invalid_geometry("hidden token count exceeds the usize range")
    })
}

fn sorted_unique_expert_ids(
    selected_indices: &MlxArray,
) -> Result<Vec<usize>, LagunaExecutionError> {
    let mut expert_ids = selected_indices
        .to_vec_u32()?
        .into_iter()
        .map(|expert_id| expert_id as usize)
        .collect::<Vec<_>>();
    expert_ids.sort_unstable();
    expert_ids.dedup();
    Ok(expert_ids)
}

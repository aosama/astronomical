use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxRuntime, MlxRuntimeError,
};

use crate::PerformanceAttribution;
use crate::sparse_experts::{
    restore_expert_assignment_order, sort_expert_assignments, sorted_expert_weighted_sum,
    sorted_expert_weighted_sum_kernel, unsorted_expert_weighted_sum,
};

const ROUTE_EXPERTS_OPERATION: &str = "route Qwen3.5-MoE sparse experts";
const COMBINE_EXPERTS_OPERATION: &str = "combine Qwen3.5-MoE sparse and shared experts";
pub(super) const MINIMUM_SORTED_EXPERT_ASSIGNMENTS: usize = 64;

/// Selects and optionally renormalizes the highest-probability sparse experts.
pub fn qwen3_5_moe_route_experts(
    runtime: &MlxRuntime,
    router_logits: &MlxArray,
    expert_count_per_token: i32,
    should_normalize_scores: bool,
) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
    let router_shape = router_logits.shape();
    validate_router_arguments(router_logits, &router_shape, expert_count_per_token)?;
    let total_expert_count = *router_shape
        .last()
        .ok_or_else(|| route_experts_error("router logits must not be scalar"))?;
    let first_selected_expert = total_expert_count - expert_count_per_token;
    let probabilities = runtime.softmax_axis(router_logits, -1)?;
    let partitioned_indices =
        runtime.argpartition_axis(&probabilities, first_selected_expert, -1)?;
    let mut slice_starts = vec![0; router_shape.len()];
    let slice_stops = router_shape;
    let slice_strides = vec![1; slice_starts.len()];
    let expert_axis = slice_starts.len() - 1;
    slice_starts[expert_axis] = first_selected_expert;
    let selected_indices = runtime.slice(
        &partitioned_indices,
        &slice_starts,
        &slice_stops,
        &slice_strides,
    )?;
    let mut selected_scores = runtime.take_along_axis(&probabilities, &selected_indices, -1)?;
    if should_normalize_scores {
        let selected_score_sums = runtime.sum_axis(&selected_scores, -1, true)?;
        selected_scores = runtime.divide(&selected_scores, &selected_score_sums)?;
    }
    Ok((selected_indices, selected_scores))
}

/// Sorts many token-to-expert assignments so gather-QMM reads expert weights contiguously.
pub fn qwen3_5_moe_sort_expert_assignments(
    runtime: &MlxRuntime,
    expanded_states: &MlxArray,
    selected_indices: &MlxArray,
) -> Result<(MlxArray, MlxArray, MlxArray), MlxRuntimeError> {
    let mut performance_attribution = PerformanceAttribution::disabled();
    let sorted_assignments = sort_expert_assignments(
        runtime,
        expanded_states,
        selected_indices,
        &mut performance_attribution,
    )?;
    Ok((
        sorted_assignments.sorted_states,
        sorted_assignments.sorted_indices,
        sorted_assignments.inverse_order,
    ))
}

/// Restores sorted gather-QMM outputs to their original token and top-k assignment order.
pub fn qwen3_5_moe_restore_expert_assignment_order(
    runtime: &MlxRuntime,
    sorted_expert_outputs: &MlxArray,
    inverse_order: &MlxArray,
    selected_index_shape: &[i32],
) -> Result<MlxArray, MlxRuntimeError> {
    restore_expert_assignment_order(
        runtime,
        sorted_expert_outputs,
        inverse_order,
        selected_index_shape,
    )
}

/// Combines sorted routed-expert outputs directly without materializing `[B, T, K, D]`.
pub fn qwen3_5_moe_sorted_expert_weighted_sum(
    runtime: &MlxRuntime,
    weighted_sum_kernel: &MlxMetalKernel,
    sorted_expert_outputs: &MlxArray,
    inverse_order: &MlxArray,
    selected_scores: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let mut performance_attribution = PerformanceAttribution::disabled();
    sorted_expert_weighted_sum(
        runtime,
        weighted_sum_kernel,
        sorted_expert_outputs,
        inverse_order,
        selected_scores,
        &mut performance_attribution,
    )
}

/// Reduces unsorted Qwen expert outputs while restoring their activation dtype.
pub fn qwen3_5_moe_unsorted_expert_weighted_sum(
    runtime: &MlxRuntime,
    selected_expert_outputs: &MlxArray,
    selected_scores: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let mut performance_attribution = PerformanceAttribution::disabled();
    unsorted_expert_weighted_sum(
        runtime,
        selected_expert_outputs,
        selected_scores,
        &mut performance_attribution,
    )
}

pub fn qwen3_5_moe_sorted_expert_weighted_sum_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    sorted_expert_weighted_sum_kernel()
}

fn validate_router_arguments(
    router_logits: &MlxArray,
    router_shape: &[i32],
    expert_count_per_token: i32,
) -> Result<(), MlxRuntimeError> {
    if router_shape.len() < 2 {
        return Err(route_experts_error(
            "router logits must include token and expert dimensions",
        ));
    }
    let total_expert_count = *router_shape
        .last()
        .ok_or_else(|| route_experts_error("router logits must not be scalar"))?;
    if total_expert_count <= 0
        || expert_count_per_token <= 0
        || expert_count_per_token > total_expert_count
    {
        return Err(route_experts_error(
            "selected expert count must be positive and no larger than the router dimension",
        ));
    }
    if !matches!(
        router_logits.dtype(),
        MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
    ) {
        return Err(route_experts_error(
            "router logits must use float16, bfloat16, or float32",
        ));
    }
    Ok(())
}

fn route_experts_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: ROUTE_EXPERTS_OPERATION,
        description: description.to_owned(),
    }
}

pub(super) fn combine_experts_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: COMBINE_EXPERTS_OPERATION,
        description: description.to_owned(),
    }
}

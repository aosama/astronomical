use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    MlxRuntime, MlxRuntimeError,
};

const ROUTE_EXPERTS_OPERATION: &str = "route Qwen3.5-MoE sparse experts";
const COMBINE_EXPERTS_OPERATION: &str = "combine Qwen3.5-MoE sparse and shared experts";
pub(super) const MINIMUM_SORTED_EXPERT_ASSIGNMENTS: usize = 64;
const SORTED_EXPERT_WEIGHTED_SUM_SOURCE: &str = r#"
    auto output_index = thread_position_in_grid.x;
    auto token_index = output_index / output_dimension;
    auto output_dimension_index = output_index % output_dimension;
    float weighted_sum = 0.0f;
    for (int selected_expert_index = 0;
         selected_expert_index < selected_expert_count;
         ++selected_expert_index) {
        auto assignment_index =
            token_index * selected_expert_count + selected_expert_index;
        auto sorted_assignment_index = inverse_order[assignment_index];
        weighted_sum +=
            (float)sorted_outputs[sorted_assignment_index * output_dimension +
                                  output_dimension_index] *
            (float)scores[assignment_index];
    }
    weighted_outputs[output_index] = (OutputT)weighted_sum;
"#;

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
    let selected_index_shape = selected_indices.shape();
    let expanded_state_shape = expanded_states.shape();
    let expert_count_per_token = selected_index_shape.last().copied().ok_or_else(|| {
        route_experts_error("selected expert indices must include an expert dimension")
    })?;
    let assignment_count = i32::try_from(selected_indices.element_count()).map_err(|_| {
        route_experts_error("expert assignment count exceeds the MLX integer range")
    })?;
    if selected_index_shape.len() < 2
        || expanded_state_shape.len() < 3
        || expert_count_per_token <= 0
        || assignment_count <= 0
        || expanded_state_shape[expanded_state_shape.len() - 2] != 1
    {
        return Err(route_experts_error(
            "expert assignments and expanded hidden states have invalid shapes",
        ));
    }
    let token_count = assignment_count / expert_count_per_token;
    let hidden_dimension = *expanded_state_shape
        .last()
        .ok_or_else(|| route_experts_error("expanded hidden states must not be scalar"))?;
    let expanded_token_count = expanded_state_shape[..expanded_state_shape.len() - 2]
        .iter()
        .try_fold(1_i32, |token_product, dimension| {
            token_product.checked_mul(*dimension)
        })
        .ok_or_else(|| route_experts_error("expanded hidden-state shape overflows"))?;
    if assignment_count % expert_count_per_token != 0
        || expanded_token_count != token_count
        || hidden_dimension <= 0
    {
        return Err(route_experts_error(
            "expert assignments do not match expanded hidden-state tokens",
        ));
    }

    let flattened_indices = runtime.reshape(selected_indices, &[assignment_count])?;
    let sorted_order = runtime.argsort_axis(&flattened_indices, 0)?;
    let inverse_order = runtime.argsort_axis(&sorted_order, 0)?;
    let assignments_per_token = runtime.array_from_u32(
        &[u32::try_from(expert_count_per_token).map_err(|_| {
            route_experts_error("expert count per token exceeds the unsigned integer range")
        })?],
        &[],
    )?;
    let sorted_token_indices = runtime.floor_divide(&sorted_order, &assignments_per_token)?;
    let flattened_states = runtime.reshape(expanded_states, &[token_count, 1, hidden_dimension])?;
    let sorted_states = runtime.take_axis(&flattened_states, &sorted_token_indices, 0)?;
    let sorted_indices = runtime.take_axis(&flattened_indices, &sorted_order, 0)?;
    Ok((sorted_states, sorted_indices, inverse_order))
}

/// Restores sorted gather-QMM outputs to their original token and top-k assignment order.
pub fn qwen3_5_moe_restore_expert_assignment_order(
    runtime: &MlxRuntime,
    sorted_expert_outputs: &MlxArray,
    inverse_order: &MlxArray,
    selected_index_shape: &[i32],
) -> Result<MlxArray, MlxRuntimeError> {
    let sorted_output_shape = sorted_expert_outputs.shape();
    let assignment_count = selected_index_shape
        .iter()
        .try_fold(1_i32, |assignment_product, dimension| {
            assignment_product.checked_mul(*dimension)
        })
        .ok_or_else(|| combine_experts_error("selected expert-index shape overflows"))?;
    if sorted_output_shape.len() != 3
        || sorted_output_shape[0] != assignment_count
        || sorted_output_shape[1] != 1
        || inverse_order.shape() != [assignment_count]
    {
        return Err(combine_experts_error(
            "sorted expert outputs and inverse order have incompatible shapes",
        ));
    }
    let output_dimension = sorted_output_shape[2];
    let original_order_outputs = runtime.take_axis(sorted_expert_outputs, inverse_order, 0)?;
    let mut restored_shape = selected_index_shape.to_vec();
    restored_shape.push(1);
    restored_shape.push(output_dimension);
    let restored_outputs = runtime.reshape(&original_order_outputs, &restored_shape)?;
    runtime.squeeze_axis(&restored_outputs, -2)
}

/// Combines sorted routed-expert outputs directly without materializing `[B, T, K, D]`.
pub fn qwen3_5_moe_sorted_expert_weighted_sum(
    runtime: &MlxRuntime,
    weighted_sum_kernel: &MlxMetalKernel,
    sorted_expert_outputs: &MlxArray,
    inverse_order: &MlxArray,
    selected_scores: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let sorted_output_shape = sorted_expert_outputs.shape();
    let score_shape = selected_scores.shape();
    if sorted_output_shape.len() != 3
        || sorted_output_shape[1] != 1
        || score_shape.len() < 2
        || inverse_order.dtype() != MlxDtype::UInt32
    {
        return Err(combine_experts_error(
            "sorted expert outputs, inverse order, and scores have invalid shapes or dtypes",
        ));
    }
    let selected_expert_count = *score_shape
        .last()
        .ok_or_else(|| combine_experts_error("selected scores must not be scalar"))?;
    let assignment_count = score_shape
        .iter()
        .try_fold(1_i32, |product, dimension| product.checked_mul(*dimension));
    let assignment_count =
        assignment_count.ok_or_else(|| combine_experts_error("selected-score shape overflows"))?;
    if selected_expert_count <= 0
        || sorted_output_shape[0] != assignment_count
        || inverse_order.shape() != [assignment_count]
    {
        return Err(combine_experts_error(
            "sorted expert outputs and inverse order must match selected scores",
        ));
    }
    let output_dimension = sorted_output_shape[2];
    let mut weighted_output_shape = score_shape;
    weighted_output_shape.pop();
    weighted_output_shape.push(output_dimension);
    let output_element_count = assignment_count
        .checked_div(selected_expert_count)
        .and_then(|token_count| token_count.checked_mul(output_dimension))
        .ok_or_else(|| combine_experts_error("weighted expert output shape overflows"))?;
    let mut kernel_outputs = runtime.apply_metal_kernel(
        weighted_sum_kernel,
        &[sorted_expert_outputs, inverse_order, selected_scores],
        &[MlxMetalKernelOutput::new(
            weighted_output_shape,
            sorted_expert_outputs.dtype(),
        )],
        [output_element_count, 1, 1],
        [256.min(output_element_count), 1, 1],
        &[
            MlxMetalKernelTemplateArgument::Dtype {
                name: "OutputT",
                dtype: sorted_expert_outputs.dtype(),
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "selected_expert_count",
                integer_template_argument: selected_expert_count,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "output_dimension",
                integer_template_argument: output_dimension,
            },
        ],
    )?;
    kernel_outputs.pop().ok_or_else(|| {
        combine_experts_error("sorted expert weighted-sum kernel returned no output")
    })
}

pub fn qwen3_5_moe_sorted_expert_weighted_sum_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    MlxMetalKernel::new(
        "qwen3_5_moe_sorted_expert_weighted_sum",
        &["sorted_outputs", "inverse_order", "scores"],
        &["weighted_outputs"],
        SORTED_EXPERT_WEIGHTED_SUM_SOURCE,
    )
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

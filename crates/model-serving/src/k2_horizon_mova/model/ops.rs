//! Family-owned grouped RMSNorm, K2 router, and attention-gate math.

use astronomical_runtime_integration::{
    MlxArray, MlxCompiledSwiGlu, MlxDtype, MlxMetalKernel, MlxRuntime, MlxRuntimeError,
};

use crate::PerformanceAttribution;
use crate::k2_horizon_mova::configuration::K2HorizonMoVAAttentionGateFunc;
use crate::sparse_experts::{
    ExpertAssignmentOrder, MINIMUM_SORTED_EXPERT_ASSIGNMENTS, StackedExpertProjection,
    gather_expert_projection, restore_expert_assignment_order, sort_expert_assignments,
    sorted_expert_weighted_sum, unsorted_expert_weighted_sum,
};

use super::affine::K2HorizonMoVAAffineLinear;
use super::error::K2HorizonMoVAExecutionError;

pub fn grouped_rms_norm(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weight: &MlxArray,
    group_count: usize,
    epsilon: f32,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let mut grouped_shape = hidden_states.shape();
    let last_axis =
        grouped_shape
            .last_mut()
            .ok_or(K2HorizonMoVAExecutionError::InvalidExecution {
                description: "grouped RMSNorm input is missing a last axis".to_owned(),
            })?;
    if *last_axis % group_count as i32 != 0 {
        return Err(K2HorizonMoVAExecutionError::InvalidExecution {
            description: "hidden size is not divisible by layernorm groups".to_owned(),
        });
    }
    let group_width = *last_axis / group_count as i32;
    *last_axis = group_count as i32;
    grouped_shape.push(group_width);
    let grouped = runtime.reshape(hidden_states, &grouped_shape)?;
    let normalized = runtime.rms_norm_without_weight(&grouped, epsilon)?;
    let restored = runtime.reshape(&normalized, &hidden_states.shape())?;
    Ok(runtime.multiply(&restored, weight)?)
}

pub fn attention_gate(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    projection: &K2HorizonMoVAAffineLinear,
    gate_func: K2HorizonMoVAAttentionGateFunc,
    num_heads: usize,
    head_dim: usize,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let projected = projection.project(runtime, hidden_states)?;
    let mut gated_shape = hidden_states.shape();
    gated_shape.pop();
    gated_shape.push(num_heads as i32);
    gated_shape.push(head_dim as i32);
    let reshaped = runtime.reshape(&projected, &gated_shape)?;
    let activated = match gate_func {
        K2HorizonMoVAAttentionGateFunc::Silu => runtime.silu(&reshaped)?,
        K2HorizonMoVAAttentionGateFunc::Softplus => {
            let ln2 = 2.0_f32.ln();
            let scaled = runtime.multiply_scalar(&reshaped, ln2)?;
            let exponent = runtime.exp(&scaled)?;
            let log = runtime.log1p(&exponent)?;
            runtime.multiply_scalar(&log, 1.0 / ln2)?
        }
    };
    Ok(runtime.transpose_axes(&activated, &[0, 2, 1, 3])?)
}

pub struct K2HorizonMoVARouterSelection {
    pub weights: MlxArray,
    pub indices: MlxArray,
}

pub fn route_k2_experts(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    router: &K2HorizonMoVAAffineLinear,
    top_k: usize,
    expert_count: usize,
    normalize: bool,
    scale: f32,
) -> Result<K2HorizonMoVARouterSelection, K2HorizonMoVAExecutionError> {
    let logits = router.project_without_dense_bias(runtime, hidden_states)?;
    let scores = runtime.sigmoid(&runtime.astype(&logits, MlxDtype::Float32)?)?;
    let choice_scores = router
        .dense_bias()
        .map(|router_bias| runtime.add(&scores, &runtime.astype(router_bias, scores.dtype())?))
        .transpose()?;
    let kth = i32::try_from(expert_count.saturating_sub(top_k)).unwrap_or(0);
    let partitioned =
        runtime.argpartition_axis(choice_scores.as_ref().unwrap_or(&scores), kth, -1)?;
    let token_count = partitioned.shape().first().copied().unwrap_or(0);
    let expert_axis = partitioned.shape().last().copied().unwrap_or(0);
    let indices = runtime.slice(
        &partitioned,
        &[0, expert_axis - top_k as i32],
        &[token_count, expert_axis],
        &[1, 1],
    )?;
    let mut selected = runtime.take_along_axis(&scores, &indices, -1)?;
    if normalize {
        let total = runtime.sum_axis(&selected, -1, true)?;
        selected = runtime.divide(&selected, &total)?;
    }
    let weights =
        runtime.multiply_scalar(&runtime.astype(&selected, hidden_states.dtype())?, scale)?;
    Ok(K2HorizonMoVARouterSelection { weights, indices })
}

pub fn gathered_fused_swiglu(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    gate_up: &K2HorizonMoVAAffineLinear,
    down: &K2HorizonMoVAAffineLinear,
    indices: &MlxArray,
    router_weights: &MlxArray,
    compiled_swiglu: &MlxCompiledSwiGlu,
    sorted_expert_reduction_kernel: Option<&MlxMetalKernel>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let gathered_assignments =
        prepare_gathered_assignments(runtime, hidden_states, indices, performance_attribution)?;
    let fused_output = gather_affine(
        runtime,
        gathered_assignments.activations(),
        gate_up,
        gathered_assignments.indices(),
        gathered_assignments.assignment_order(),
        performance_attribution,
    )?;
    let (gate_out, up_out) = K2HorizonMoVAAffineLinear::split_fused_output(runtime, &fused_output)?;
    let hidden = runtime.apply_compiled_swiglu(compiled_swiglu, &gate_out, &up_out)?;
    let projected = gather_affine(
        runtime,
        &hidden,
        down,
        gathered_assignments.indices(),
        gathered_assignments.assignment_order(),
        performance_attribution,
    )?;
    gathered_assignments.reduce_outputs(
        runtime,
        projected,
        router_weights,
        sorted_expert_reduction_kernel,
        performance_attribution,
    )
}

pub fn dense_fused_swiglu(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    gate_up: &K2HorizonMoVAAffineLinear,
    down: &K2HorizonMoVAAffineLinear,
    compiled_swiglu: &MlxCompiledSwiGlu,
) -> Result<MlxArray, MlxRuntimeError> {
    let fused_output = gate_up.project(runtime, hidden_states)?;
    let (gate_out, up_out) = K2HorizonMoVAAffineLinear::split_fused_output(runtime, &fused_output)?;
    down.project(
        runtime,
        &runtime.apply_compiled_swiglu(compiled_swiglu, &gate_out, &up_out)?,
    )
}

pub fn gathered_value_experts(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    value_experts: &K2HorizonMoVAAffineLinear,
    indices: &MlxArray,
    router_weights: &MlxArray,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let routed = gathered_affine_projection(
        runtime,
        hidden_states,
        value_experts,
        indices,
        performance_attribution,
    )?;
    let activated = runtime.silu(&routed)?;
    Ok(unsorted_expert_weighted_sum(
        runtime,
        &activated,
        router_weights,
        performance_attribution,
    )?)
}

fn gathered_affine_projection(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    linear: &K2HorizonMoVAAffineLinear,
    indices: &MlxArray,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let gathered_assignments =
        prepare_gathered_assignments(runtime, hidden_states, indices, performance_attribution)?;
    let projected = gather_affine(
        runtime,
        gathered_assignments.activations(),
        linear,
        gathered_assignments.indices(),
        gathered_assignments.assignment_order(),
        performance_attribution,
    )?;
    gathered_assignments.restore_outputs(runtime, projected)
}

enum GatheredAssignments<'a> {
    Original {
        activations: MlxArray,
        indices: &'a MlxArray,
    },
    Sorted {
        activations: MlxArray,
        indices: MlxArray,
        inverse_order: MlxArray,
        original_index_shape: Vec<i32>,
    },
}

impl GatheredAssignments<'_> {
    fn activations(&self) -> &MlxArray {
        match self {
            Self::Original { activations, .. } | Self::Sorted { activations, .. } => activations,
        }
    }

    fn indices(&self) -> &MlxArray {
        match self {
            Self::Original { indices, .. } => indices,
            Self::Sorted { indices, .. } => indices,
        }
    }

    fn assignment_order(&self) -> ExpertAssignmentOrder {
        match self {
            Self::Original { .. } => ExpertAssignmentOrder::Original,
            Self::Sorted { .. } => ExpertAssignmentOrder::SortedByExpert,
        }
    }

    fn restore_outputs(
        &self,
        runtime: &MlxRuntime,
        projected: MlxArray,
    ) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
        match self {
            Self::Original { .. } => Ok(runtime.squeeze_axis(&projected, -2)?),
            Self::Sorted {
                inverse_order,
                original_index_shape,
                ..
            } => Ok(restore_expert_assignment_order(
                runtime,
                &projected,
                inverse_order,
                original_index_shape,
            )?),
        }
    }

    fn reduce_outputs(
        &self,
        runtime: &MlxRuntime,
        projected: MlxArray,
        router_weights: &MlxArray,
        sorted_expert_reduction_kernel: Option<&MlxMetalKernel>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
        if let (Self::Sorted { inverse_order, .. }, Some(sorted_expert_reduction_kernel)) =
            (self, sorted_expert_reduction_kernel)
        {
            let already_three_d =
                projected.shape().len() == 3 && projected.shape().get(1) == Some(&1);
            let squeezed_outputs = (!already_three_d)
                .then(|| three_d_sorted_outputs(runtime, &projected))
                .flatten();
            if already_three_d || squeezed_outputs.is_some() {
                let sorted_outputs = squeezed_outputs.as_ref().unwrap_or(&projected);
                return Ok(sorted_expert_weighted_sum(
                    runtime,
                    sorted_expert_reduction_kernel,
                    sorted_outputs,
                    inverse_order,
                    router_weights,
                    performance_attribution,
                )?);
            }
        }
        let restored = self.restore_outputs(runtime, projected)?;
        Ok(unsorted_expert_weighted_sum(
            runtime,
            &restored,
            router_weights,
            performance_attribution,
        )?)
    }
}

fn three_d_sorted_outputs(runtime: &MlxRuntime, projected: &MlxArray) -> Option<MlxArray> {
    let mut sorted_outputs = runtime.squeeze_axis(projected, -2).ok()?;
    while sorted_outputs.shape().len() > 3 {
        sorted_outputs = runtime.squeeze_axis(&sorted_outputs, -2).ok()?;
    }
    if sorted_outputs.shape().len() == 2 {
        sorted_outputs = runtime.expand_dims(&sorted_outputs, 1).ok()?;
    }
    let shape = sorted_outputs.shape();
    (shape.len() == 3 && shape[1] == 1).then_some(sorted_outputs)
}

fn prepare_gathered_assignments<'a>(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    indices: &'a MlxArray,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<GatheredAssignments<'a>, K2HorizonMoVAExecutionError> {
    let expanded = expand_for_gather(runtime, hidden_states)?;
    if indices.element_count() < MINIMUM_SORTED_EXPERT_ASSIGNMENTS {
        return Ok(GatheredAssignments::Original {
            activations: expanded,
            indices,
        });
    }
    let sorted = sort_expert_assignments(runtime, &expanded, indices, performance_attribution)?;
    Ok(GatheredAssignments::Sorted {
        activations: sorted.sorted_states,
        indices: sorted.sorted_indices,
        inverse_order: sorted.inverse_order,
        original_index_shape: indices.shape(),
    })
}

fn expand_for_gather(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let expanded = runtime.expand_dims(hidden_states, -2)?;
    runtime.expand_dims(&expanded, -3)
}

fn gather_affine(
    runtime: &MlxRuntime,
    activations: &MlxArray,
    linear: &K2HorizonMoVAAffineLinear,
    indices: &MlxArray,
    assignment_order: ExpertAssignmentOrder,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    Ok(gather_expert_projection(
        runtime,
        activations,
        StackedExpertProjection::Affine {
            packed_weights: linear.packed_weight(),
            scales: linear.scales(),
            biases: linear.biases(),
            group_size: linear.group_size(),
            bits: linear.bits(),
        },
        indices,
        assignment_order,
        performance_attribution,
    )?)
}

pub fn dense_swiglu(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    mlp_gate: &K2HorizonMoVAAffineLinear,
    mlp_up: &K2HorizonMoVAAffineLinear,
    mlp_down: &K2HorizonMoVAAffineLinear,
    compiled_swiglu: &MlxCompiledSwiGlu,
) -> Result<MlxArray, MlxRuntimeError> {
    let gate = mlp_gate.project(runtime, hidden_states)?;
    let up = mlp_up.project(runtime, hidden_states)?;
    mlp_down.project(
        runtime,
        &runtime.apply_compiled_swiglu(compiled_swiglu, &gate, &up)?,
    )
}

pub fn reshape_heads(
    runtime: &MlxRuntime,
    projected: &MlxArray,
    num_heads: usize,
    head_dim: usize,
) -> Result<MlxArray, MlxRuntimeError> {
    let mut shape = projected.shape();
    shape.pop();
    shape.push(num_heads as i32);
    shape.push(head_dim as i32);
    let reshaped = runtime.reshape(projected, &shape)?;
    runtime.transpose_axes(&reshaped, &[0, 2, 1, 3])
}

pub fn merge_heads(
    runtime: &MlxRuntime,
    attention: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let transposed = runtime.transpose_axes(attention, &[0, 2, 1, 3])?;
    let shape = transposed.shape();
    let merged = [shape[0], shape[1], shape[2] * shape[3]];
    runtime.reshape(&transposed, &merged)
}

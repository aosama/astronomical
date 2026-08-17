//! Independent resident-MoE oracle composed without Laguna execution helpers.

use std::collections::HashMap;

use astronomical_model_serving::{
    LagunaExpertProjection, LagunaLayerTensorRole, LagunaMoeDescriptor, LagunaTensorId,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

use super::tensor_identity::layer_id;

pub(super) fn reference_moe(
    runtime: &MlxRuntime,
    tensors: &HashMap<LagunaTensorId, MlxArray>,
    layer_index: usize,
    descriptor: &LagunaMoeDescriptor,
    hidden_states: &MlxArray,
    router_logit_softcap: f64,
) -> Result<MlxArray, MlxRuntimeError> {
    let router_logits = dense_linear(
        runtime,
        tensor(
            tensors,
            layer_id(layer_index, LagunaLayerTensorRole::Router),
        ),
        hidden_states,
    )?;
    let correction_bias = tensors.get(&layer_id(
        layer_index,
        LagunaLayerTensorRole::RouterCorrectionBias,
    ));
    let (selected_expert_indices, selected_expert_scores) = select_experts(
        runtime,
        &router_logits,
        correction_bias,
        descriptor.experts_per_token() as i32,
        router_logit_softcap,
        descriptor.normalizes_top_k_probabilities(),
    )?;
    let expanded_hidden_states = runtime.expand_dims(hidden_states, -2)?;
    let expert_input_states = if descriptor.applies_router_weight_on_input() {
        let expanded_scores = runtime.expand_dims(&selected_expert_scores, -1)?;
        let weighted_inputs = runtime.multiply(&expanded_hidden_states, &expanded_scores)?;
        runtime.astype(&weighted_inputs, hidden_states.dtype())?
    } else {
        expanded_hidden_states
    };
    let gate = gathered_expert_linear(
        runtime,
        tensor(
            tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
            ),
        ),
        &expert_input_states,
        &selected_expert_indices,
    )?;
    let up = gathered_expert_linear(
        runtime,
        tensor(
            tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Up),
            ),
        ),
        &expert_input_states,
        &selected_expert_indices,
    )?;
    let activated = runtime.multiply(&runtime.silu(&gate)?, &up)?;
    let selected_outputs = gathered_expert_linear(
        runtime,
        tensor(
            tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Down),
            ),
        ),
        &activated,
        &selected_expert_indices,
    )?;
    let routed_output = if descriptor.applies_router_weight_on_input() {
        runtime.sum_axis(&selected_outputs, -2, false)?
    } else {
        let expanded_scores = runtime.expand_dims(&selected_expert_scores, -1)?;
        let weighted_outputs = runtime.multiply(&selected_outputs, &expanded_scores)?;
        let accumulated = runtime.sum_axis(&weighted_outputs, -2, false)?;
        runtime.astype(&accumulated, selected_outputs.dtype())?
    };
    let scaled_routed_output =
        runtime.multiply_scalar(&routed_output, descriptor.routed_scaling_factor() as f32)?;
    if descriptor.shared_expert_intermediate_size() == 0 {
        return Ok(scaled_routed_output);
    }
    let shared_output = shared_expert(runtime, tensors, layer_index, hidden_states)?;
    runtime.add(&scaled_routed_output, &shared_output)
}

fn select_experts(
    runtime: &MlxRuntime,
    router_logits: &MlxArray,
    correction_bias: Option<&MlxArray>,
    experts_per_token: i32,
    router_logit_softcap: f64,
    normalizes_top_k_probabilities: bool,
) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
    let float_logits = runtime.astype(router_logits, MlxDtype::Float32)?;
    let bounded_logits = if router_logit_softcap > 0.0 {
        let softcap = router_logit_softcap as f32;
        let scaled_logits = runtime.multiply_scalar(&float_logits, softcap.recip())?;
        runtime.multiply_scalar(&runtime.tanh(&scaled_logits)?, softcap)?
    } else {
        float_logits
    };
    let original_scores = runtime.sigmoid(&bounded_logits)?;
    let ranking_scores = correction_bias.map_or_else(
        || original_scores.retain(),
        |bias| runtime.add(&original_scores, bias),
    )?;
    // Stable argsort is intentionally different from production's argpartition.
    // It also gives deterministic lower-index ordering when equal scores occur.
    let ranked_indices = runtime.argsort_axis(&runtime.negative(&ranking_scores)?, -1)?;
    let rank_shape = ranked_indices.shape();
    let slice_starts = vec![0; rank_shape.len()];
    let mut slice_stops = rank_shape;
    let slice_strides = vec![1; slice_starts.len()];
    let expert_axis = slice_stops.len() - 1;
    slice_stops[expert_axis] = experts_per_token;
    let selected_indices =
        runtime.slice(&ranked_indices, &slice_starts, &slice_stops, &slice_strides)?;
    let mut selected_scores = runtime.take_along_axis(&original_scores, &selected_indices, -1)?;
    if normalizes_top_k_probabilities {
        let selected_score_sums = runtime.sum_axis(&selected_scores, -1, true)?;
        selected_scores = runtime.divide(&selected_scores, &selected_score_sums)?;
    }
    Ok((selected_indices, selected_scores))
}

fn gathered_expert_linear(
    runtime: &MlxRuntime,
    expert_weights: &MlxArray,
    assignment_inputs: &MlxArray,
    selected_expert_indices: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    // Explicitly selecting complete matrices and using batched matmul avoids the
    // production gather_mm/gather_qmm path while preserving expert alignment.
    let selected_weights = runtime.take_axis(expert_weights, selected_expert_indices, 0)?;
    let selected_weight_shape = selected_weights.shape();
    let weight_rank = selected_weight_shape.len();
    let mut transpose_axes = (0..weight_rank as i32).collect::<Vec<_>>();
    transpose_axes.swap(weight_rank - 2, weight_rank - 1);
    let transposed_weights = runtime.transpose_axes(&selected_weights, &transpose_axes)?;
    let matrix_inputs = runtime.expand_dims(assignment_inputs, -2)?;
    let projected = runtime.matmul(&matrix_inputs, &transposed_weights)?;
    runtime.squeeze_axis(&projected, -2)
}

fn shared_expert(
    runtime: &MlxRuntime,
    tensors: &HashMap<LagunaTensorId, MlxArray>,
    layer_index: usize,
    hidden_states: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let project = |projection| {
        dense_linear(
            runtime,
            tensor(
                tensors,
                layer_id(layer_index, LagunaLayerTensorRole::SharedExpert(projection)),
            ),
            hidden_states,
        )
    };
    let gate = project(LagunaExpertProjection::Gate)?;
    let up = project(LagunaExpertProjection::Up)?;
    let activated = runtime.multiply(&runtime.silu(&gate)?, &up)?;
    dense_linear(
        runtime,
        tensor(
            tensors,
            layer_id(
                layer_index,
                LagunaLayerTensorRole::SharedExpert(LagunaExpertProjection::Down),
            ),
        ),
        &activated,
    )
}

fn dense_linear(
    runtime: &MlxRuntime,
    weight: &MlxArray,
    input: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    runtime.matmul(input, &runtime.transpose_axes(weight, &[1, 0])?)
}

fn tensor(tensors: &HashMap<LagunaTensorId, MlxArray>, tensor_id: LagunaTensorId) -> &MlxArray {
    tensors
        .get(&tensor_id)
        .unwrap_or_else(|| panic!("reference tensor {tensor_id:?} should exist"))
}

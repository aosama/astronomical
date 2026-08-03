use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use super::routing::combine_experts_error;

/// Combines selected expert outputs and the separately gated shared expert.
pub fn qwen3_5_moe_combine_experts(
    runtime: &MlxRuntime,
    selected_expert_outputs: &MlxArray,
    selected_scores: &MlxArray,
    shared_expert_output: &MlxArray,
    shared_expert_gate_logits: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    validate_combination_shapes(
        selected_expert_outputs,
        selected_scores,
        shared_expert_output,
        shared_expert_gate_logits,
    )?;
    let expanded_scores = runtime.expand_dims(selected_scores, -1)?;
    let weighted_expert_outputs = runtime.multiply(selected_expert_outputs, &expanded_scores)?;
    let sparse_expert_output = runtime.sum_axis(&weighted_expert_outputs, -2, false)?;
    combine_sparse_and_shared_experts(
        runtime,
        &sparse_expert_output,
        shared_expert_output,
        shared_expert_gate_logits,
    )
}

pub(super) fn combine_sparse_and_shared_experts(
    runtime: &MlxRuntime,
    sparse_expert_output: &MlxArray,
    shared_expert_output: &MlxArray,
    shared_expert_gate_logits: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let shared_expert_gate = runtime.sigmoid(shared_expert_gate_logits)?;
    let gated_shared_expert_output = runtime.multiply(shared_expert_output, &shared_expert_gate)?;
    runtime.add(sparse_expert_output, &gated_shared_expert_output)
}

fn validate_combination_shapes(
    selected_expert_outputs: &MlxArray,
    selected_scores: &MlxArray,
    shared_expert_output: &MlxArray,
    shared_expert_gate_logits: &MlxArray,
) -> Result<(), MlxRuntimeError> {
    let selected_output_shape = selected_expert_outputs.shape();
    if selected_output_shape.len() < 3 {
        return Err(combine_experts_error(
            "selected expert outputs must include token, expert, and output dimensions",
        ));
    }
    let mut expected_score_shape = selected_output_shape.clone();
    expected_score_shape.pop();
    if selected_scores.shape() != expected_score_shape {
        return Err(combine_experts_error(
            "selected scores must match selected expert outputs without the output dimension",
        ));
    }
    let expert_axis = selected_output_shape.len() - 2;
    let mut expected_shared_output_shape = selected_output_shape;
    expected_shared_output_shape.remove(expert_axis);
    if shared_expert_output.shape() != expected_shared_output_shape {
        return Err(combine_experts_error(
            "shared expert output must match the combined sparse-expert shape",
        ));
    }
    let gate_axis = expected_shared_output_shape.len() - 1;
    expected_shared_output_shape[gate_axis] = 1;
    if shared_expert_gate_logits.shape() != expected_shared_output_shape {
        return Err(combine_experts_error(
            "shared expert gate must provide one logit per token",
        ));
    }
    Ok(())
}

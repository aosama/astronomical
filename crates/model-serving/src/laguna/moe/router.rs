//! GPU Laguna router: sigmoid scores, selection-only bias, original-score gather.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use crate::laguna::normalization::LagunaMoeDescriptor;
use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::super::model::LagunaExecutionError;

/// Routes one token batch and returns `(selected_indices, gathered_scores)`.
pub(super) fn route_laguna_experts(
    runtime: &MlxRuntime,
    router_logits: &MlxArray,
    correction_bias: Option<&MlxArray>,
    moe_descriptor: &LagunaMoeDescriptor,
    router_logit_softcap: f64,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<(MlxArray, MlxArray), LagunaExecutionError> {
    performance_attribution.measure_operation(PerformanceOperation::RouterScoreSelection, |_| {
        route_laguna_experts_inner(
            runtime,
            router_logits,
            correction_bias,
            moe_descriptor,
            router_logit_softcap,
        )
    })
}

fn route_laguna_experts_inner(
    runtime: &MlxRuntime,
    router_logits: &MlxArray,
    correction_bias: Option<&MlxArray>,
    moe_descriptor: &LagunaMoeDescriptor,
    router_logit_softcap: f64,
) -> Result<(MlxArray, MlxArray), LagunaExecutionError> {
    let router_shape = router_logits.shape();
    if router_shape.len() < 2 {
        return Err(LagunaExecutionError::invalid_geometry(
            "router logits must include token and expert dimensions",
        ));
    }
    let expert_count = *router_shape.last().ok_or_else(|| {
        LagunaExecutionError::invalid_geometry("router logits must not be scalar")
    })?;
    let experts_per_token = i32::try_from(moe_descriptor.experts_per_token()).map_err(|_| {
        LagunaExecutionError::invalid_geometry("experts per token exceed the MLX integer range")
    })?;
    if expert_count <= 0 || experts_per_token <= 0 || experts_per_token > expert_count {
        return Err(LagunaExecutionError::invalid_geometry(
            "selected expert count must be positive and no larger than the router dimension",
        ));
    }

    // Score in Float32 so sigmoid and softcap match the CPU reference.
    let float_logits = runtime.astype(router_logits, MlxDtype::Float32)?;
    let softcapped_logits = if router_logit_softcap > 0.0 {
        let softcap = router_logit_softcap as f32;
        let scaled = runtime.multiply_scalar(&float_logits, 1.0 / softcap)?;
        let bounded = runtime.tanh(&scaled)?;
        runtime.multiply_scalar(&bounded, softcap)?
    } else {
        float_logits
    };
    let original_scores = runtime.sigmoid(&softcapped_logits)?;
    let biased_selection_scores = if let Some(bias) = correction_bias {
        let float_bias = runtime.astype(bias, MlxDtype::Float32)?;
        Some(runtime.add(&original_scores, &float_bias)?)
    } else {
        None
    };
    let selection_scores = biased_selection_scores.as_ref().unwrap_or(&original_scores);

    // argpartition on negated scores puts the K largest into the leading slice.
    let negated_selection_scores = runtime.negative(selection_scores)?;
    let partitioned_indices =
        runtime.argpartition_axis(&negated_selection_scores, experts_per_token - 1, -1)?;
    let slice_starts = vec![0; router_shape.len()];
    let mut slice_stops = router_shape;
    let slice_strides = vec![1; slice_starts.len()];
    let expert_axis = slice_starts.len() - 1;
    slice_stops[expert_axis] = experts_per_token;
    let selected_indices = runtime.slice(
        &partitioned_indices,
        &slice_starts,
        &slice_stops,
        &slice_strides,
    )?;
    let mut selected_scores = runtime.take_along_axis(&original_scores, &selected_indices, -1)?;
    if moe_descriptor.normalizes_top_k_probabilities() {
        let selected_score_sums = runtime.sum_axis(&selected_scores, -1, true)?;
        selected_scores = runtime.divide(&selected_scores, &selected_score_sums)?;
    }
    Ok((selected_indices, selected_scores))
}

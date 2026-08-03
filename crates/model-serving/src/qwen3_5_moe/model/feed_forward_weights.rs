use std::collections::HashMap;

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5::model::weights::{take_quantized_affine_weights, take_tensor};
use crate::qwen3_5::{Qwen3_5Config, Qwen3_5ExecutionError};

/// Router gate weights that can be affine-quantized or plain bfloat16.
#[derive(Debug)]
pub(crate) enum Qwen3_5MoERouterGateWeights {
    Affine(Qwen3_5AffineWeights),
    Unquantized(MlxArray),
}

/// Resident router and shared-expert weights for one sparse decoder layer.
#[derive(Debug)]
pub(crate) struct Qwen3_5MoEFeedForwardWeights {
    pub(crate) router_projection: Qwen3_5MoERouterGateWeights,
    pub(crate) shared_expert_gate_projection: Qwen3_5AffineWeights,
    pub(crate) shared_expert_up_projection: Qwen3_5AffineWeights,
    pub(crate) shared_expert_down_projection: Qwen3_5AffineWeights,
    pub(crate) shared_expert_output_gate_projection: Qwen3_5AffineWeights,
}

impl Qwen3_5MoEFeedForwardWeights {
    pub(crate) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        match &self.router_projection {
            Qwen3_5MoERouterGateWeights::Affine(affine_weights) => {
                affine_weights.append_array_references(arrays);
            }
            Qwen3_5MoERouterGateWeights::Unquantized(unquantized_weight) => {
                arrays.push(unquantized_weight);
            }
        }
        self.shared_expert_gate_projection
            .append_array_references(arrays);
        self.shared_expert_up_projection
            .append_array_references(arrays);
        self.shared_expert_down_projection
            .append_array_references(arrays);
        self.shared_expert_output_gate_projection
            .append_array_references(arrays);
    }
}

pub(crate) fn bind_qwen3_5_moe_feed_forward_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
    decoder_layer_prefix: &str,
) -> Result<Qwen3_5MoEFeedForwardWeights, Qwen3_5ExecutionError> {
    let mixture_of_experts_prefix = format!("{decoder_layer_prefix}.mlp");
    let gate_module_name = format!("{mixture_of_experts_prefix}.gate");
    let gate_scales_name = format!("{gate_module_name}.scales");
    let router_projection = if bound_tensors.contains_key(&gate_scales_name) {
        Qwen3_5MoERouterGateWeights::Affine(take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &gate_module_name,
        )?)
    } else {
        Qwen3_5MoERouterGateWeights::Unquantized(take_tensor(
            bound_tensors,
            format!("{gate_module_name}.weight"),
        )?)
    };
    let bind_projection = |bound_tensors: &mut HashMap<String, MlxArray>, suffix: &str| {
        take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{mixture_of_experts_prefix}.{suffix}"),
        )
    };
    Ok(Qwen3_5MoEFeedForwardWeights {
        router_projection,
        shared_expert_gate_projection: bind_projection(bound_tensors, "shared_expert.gate_proj")?,
        shared_expert_up_projection: bind_projection(bound_tensors, "shared_expert.up_proj")?,
        shared_expert_down_projection: bind_projection(bound_tensors, "shared_expert.down_proj")?,
        shared_expert_output_gate_projection: bind_projection(bound_tensors, "shared_expert_gate")?,
    })
}

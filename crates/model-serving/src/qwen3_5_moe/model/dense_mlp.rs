use std::collections::HashMap;

use astronomical_runtime_integration::MlxArray;

use super::decoder_layer_weights::Qwen3_5DenseMlpWeights;
use super::weights::take_quantized_affine_weights;
use super::{Qwen3_5MoEConfig, Qwen3_5MoEExecutionError, Qwen3_5MoEModel};

pub(super) fn take_dense_mlp_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
    decoder_layer_prefix: &str,
) -> Result<Qwen3_5DenseMlpWeights, Qwen3_5MoEExecutionError> {
    let dense_mlp_prefix = format!("{decoder_layer_prefix}.mlp");
    Ok(Qwen3_5DenseMlpWeights {
        gate_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_moe_config,
            &format!("{dense_mlp_prefix}.gate_proj"),
        )?,
        up_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_moe_config,
            &format!("{dense_mlp_prefix}.up_proj"),
        )?,
        down_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_moe_config,
            &format!("{dense_mlp_prefix}.down_proj"),
        )?,
    })
}

impl Qwen3_5MoEModel {
    pub(super) fn forward_dense_mlp(
        &self,
        normalized_attention: &MlxArray,
        dense_mlp_weights: &Qwen3_5DenseMlpWeights,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let gate_activations =
            self.quantized_linear(normalized_attention, &dense_mlp_weights.gate_projection)?;
        let up_activations =
            self.quantized_linear(normalized_attention, &dense_mlp_weights.up_projection)?;
        let activated_intermediate_states = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &gate_activations,
            &up_activations,
        )?;
        self.quantized_linear(
            &activated_intermediate_states,
            &dense_mlp_weights.down_projection,
        )
    }
}

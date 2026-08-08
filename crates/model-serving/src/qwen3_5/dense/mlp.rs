use std::collections::HashMap;

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::Qwen3_5Config;
use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5::model::weights::take_quantized_affine_weights;
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;

/// Pre-bound projections for one dense Qwen3.5 SwiGLU MLP layer.
#[derive(Debug)]
pub(crate) struct Qwen3_5DenseMlpWeights {
    pub(crate) gate_projection: Qwen3_5AffineWeights,
    pub(crate) up_projection: Qwen3_5AffineWeights,
    pub(crate) down_projection: Qwen3_5AffineWeights,
}

impl Qwen3_5DenseMlpWeights {
    pub(crate) fn append_array_references<'weights>(
        &'weights self,
        arrays: &mut Vec<&'weights MlxArray>,
    ) {
        self.gate_projection.append_array_references(arrays);
        self.up_projection.append_array_references(arrays);
        self.down_projection.append_array_references(arrays);
    }
}

pub(crate) fn bind_qwen3_5_dense_mlp_weights(
    bound_tensors: &mut HashMap<String, MlxArray>,
    qwen3_5_config: &Qwen3_5Config,
    decoder_layer_prefix: &str,
) -> Result<Qwen3_5DenseMlpWeights, Qwen3_5ExecutionError> {
    let dense_mlp_prefix = format!("{decoder_layer_prefix}.mlp");
    Ok(Qwen3_5DenseMlpWeights {
        gate_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{dense_mlp_prefix}.gate_proj"),
        )?,
        up_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{dense_mlp_prefix}.up_proj"),
        )?,
        down_projection: take_quantized_affine_weights(
            bound_tensors,
            qwen3_5_config,
            &format!("{dense_mlp_prefix}.down_proj"),
        )?,
    })
}

impl Qwen3_5Model {
    pub(crate) fn forward_qwen3_5_dense_mlp(
        &self,
        normalized_attention: &MlxArray,
        dense_mlp_weights: &Qwen3_5DenseMlpWeights,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let gate_activations = self.quantized_linear_for_paged_prefill_execution_mode(
            normalized_attention,
            &dense_mlp_weights.gate_projection,
            paged_prefill_execution_mode,
        )?;
        let up_activations = self.quantized_linear_for_paged_prefill_execution_mode(
            normalized_attention,
            &dense_mlp_weights.up_projection,
            paged_prefill_execution_mode,
        )?;
        let activated_intermediate_states = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &gate_activations,
            &up_activations,
        )?;
        self.quantized_linear_for_paged_prefill_execution_mode(
            &activated_intermediate_states,
            &dense_mlp_weights.down_projection,
            paged_prefill_execution_mode,
        )
    }
}

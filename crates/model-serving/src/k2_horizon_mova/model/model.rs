//! Resident K2 Horizon MoVA forward pass.

use astronomical_runtime_integration::{MlxArray, MlxCompiledSwiGlu, MlxMetalKernel, MlxRuntime};

use crate::PerformanceAttribution;
use crate::gpu_token_sampling::build_sampled_token;
use crate::k2_horizon_mova::K2HorizonMoVAInferenceRequest;
use crate::k2_horizon_mova::configuration::K2HorizonMoVAConfig;

use super::decoder::{K2HorizonMoVAKvState, forward_layer};

use super::error::K2HorizonMoVAExecutionError;
use super::fused_expert_decode::FusedExpertDecodeKernels;
use super::ops::grouped_rms_norm;
use super::weights::K2HorizonMoVAWeights;

pub struct K2HorizonMoVAModel {
    pub runtime: MlxRuntime,
    pub config: K2HorizonMoVAConfig,
    pub weights: K2HorizonMoVAWeights,
    pub compiled_swiglu: MlxCompiledSwiGlu,
    pub sorted_expert_reduction_kernel: Option<MlxMetalKernel>,
    pub fused_expert_decode_kernels: Option<FusedExpertDecodeKernels>,
}

impl K2HorizonMoVAModel {
    pub fn embed(&self, token_ids: &MlxArray) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
        let packed =
            self.runtime
                .take_axis(self.weights.embed_tokens.packed_weight(), token_ids, 0)?;
        let scales = self
            .runtime
            .take_axis(self.weights.embed_tokens.scales(), token_ids, 0)?;
        let biases = self
            .runtime
            .take_axis(self.weights.embed_tokens.biases(), token_ids, 0)?;
        Ok(self.runtime.dequantize_affine(
            &packed,
            &scales,
            &biases,
            self.weights.embed_tokens.group_size(),
            self.weights.embed_tokens.bits(),
        )?)
    }

    pub fn forward(
        &self,
        token_ids: &[u32],
        caches: &mut [K2HorizonMoVAKvState],
        performance_attribution: &mut PerformanceAttribution,
        stage_attribution: bool,
    ) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
        let token_array = self
            .runtime
            .array_from_u32(token_ids, &[token_ids.len() as i32])?;
        let embeddings = self.embed(&token_array)?;
        let mut hidden_states = self.runtime.reshape(
            &embeddings,
            &[1, token_ids.len() as i32, self.config.hidden_size() as i32],
        )?;
        // Every layer receives the same token stream, so the first slab's
        // logical offset is the rope position for this forward.
        let rope_offset = caches
            .first()
            .map(K2HorizonMoVAKvState::offset_tokens)
            .unwrap_or(0);
        let is_prefill = rope_offset == 0;
        for (layer_index, layer_weights) in self.weights.layers.iter().enumerate() {
            let layer_cache = caches.get_mut(layer_index).ok_or_else(|| {
                K2HorizonMoVAExecutionError::InvalidExecution {
                    description: format!(
                        "K2 Horizon MoVA decoder state is missing layer {layer_index}"
                    ),
                }
            })?;
            let layer_output = forward_layer(
                &self.runtime,
                &self.config,
                &hidden_states,
                layer_weights,
                layer_cache,
                rope_offset,
                is_prefill,
                &self.compiled_swiglu,
                self.sorted_expert_reduction_kernel.as_ref(),
                self.fused_expert_decode_kernels.as_ref(),
                performance_attribution,
                stage_attribution,
            )?;
            hidden_states = layer_output;
        }
        let hidden_states = grouped_rms_norm(
            &self.runtime,
            &hidden_states,
            &self.weights.final_norm,
            self.config.layernorm_num_groups(),
            self.config.rms_norm_eps(),
        )?;
        let token_count = hidden_states.shape().get(1).copied().unwrap_or(1);
        let last_token_hidden_states = if token_count <= 1 {
            hidden_states
        } else {
            self.runtime.slice(
                &hidden_states,
                &[0, token_count - 1, 0],
                &[
                    1,
                    token_count,
                    hidden_states.shape().get(2).copied().unwrap_or(0),
                ],
                &[1, 1, 1],
            )?
        };
        // mlx-lm evaluates last-token hidden/logits, not the full sequence tensor.
        // Last-token attention still depends on every position's KV, so caches materialize.
        performance_attribution.measure_operation(
            crate::PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_| self.runtime.evaluate_arrays(&[&last_token_hidden_states]),
        )?;
        Ok(last_token_hidden_states)
    }

    pub fn logits_for_last_token(
        &self,
        hidden_states: &MlxArray,
    ) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
        let shape = hidden_states.shape();
        let token_count = shape.get(1).copied().unwrap_or(1);
        let last = self.runtime.slice(
            hidden_states,
            &[0, token_count - 1, 0],
            &[1, token_count, shape.get(2).copied().unwrap_or(0)],
            &[1, 1, 1],
        )?;
        let logits = self.weights.lm_head.project(&self.runtime, &last)?;
        Ok(self
            .runtime
            .reshape(&logits, &[1, 1, self.config.vocab_size() as i32])?)
    }

    pub fn sample_token(
        &self,
        logits: &MlxArray,
        request: &K2HorizonMoVAInferenceRequest,
        random_state: &mut MlxArray,
    ) -> Result<u32, K2HorizonMoVAExecutionError> {
        let sampled = build_sampled_token(
            &self.runtime,
            logits,
            request.temperature_thousandths(),
            request.top_p_thousandths(),
            None,
            random_state,
        )
        .map_err(|error| K2HorizonMoVAExecutionError::InvalidExecution {
            description: format!("{error}"),
        })?;
        sampled.evaluate()?;
        Ok(sampled.item_u32()?)
    }
}

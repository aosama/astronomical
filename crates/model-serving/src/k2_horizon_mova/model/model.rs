//! Resident K2 Horizon MoVA forward pass.

use astronomical_runtime_integration::MlxRuntime;

use crate::PerformanceAttribution;
use crate::gpu_token_sampling;
use crate::k2_horizon_mova::K2HorizonMoVAInferenceRequest;
use crate::k2_horizon_mova::configuration::K2HorizonMoVAConfig;

use super::decoder;
use super::decoder::K2HorizonMoVAKvState;

use super::error::K2HorizonMoVAExecutionError;
use super::fused_expert_decode::FusedExpertDecodeKernels;
use super::ops;
use super::weights::K2HorizonMoVAWeights;
use astronomical_mlx_c_rust::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel,
};

pub struct K2HorizonMoVAModel {
    pub runtime: MlxRuntime,
    pub config: K2HorizonMoVAConfig,
    pub weights: K2HorizonMoVAWeights,
    pub compiled_swiglu: MlxCompiledSwiGlu,
    pub compiled_elementwise_graphs: MlxCompiledElementwiseGraphs,
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
        stage_attribution: Option<super::stage_attribution_scope::StageAttributionScope>,
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
        // Causality is a property of the query length, not of the cache
        // offset: every multi-token chunk must mask its own future tokens
        // whether or not a restored prefix sits before it, while a
        // single-token decode step is causally equivalent unmasked.
        let is_causal = token_ids.len() > 1;
        for (layer_index, layer_weights) in self.weights.layers.iter().enumerate() {
            let layer_cache = caches.get_mut(layer_index).ok_or_else(|| {
                K2HorizonMoVAExecutionError::InvalidExecution {
                    description: format!(
                        "K2 Horizon MoVA decoder state is missing layer {layer_index}"
                    ),
                }
            })?;
            let layer_output = decoder::forward_layer(
                &self.runtime,
                &self.config,
                &hidden_states,
                layer_weights,
                layer_cache,
                rope_offset,
                is_causal,
                &self.compiled_swiglu,
                &self.compiled_elementwise_graphs,
                self.sorted_expert_reduction_kernel.as_ref(),
                self.fused_expert_decode_kernels.as_ref(),
                performance_attribution,
                stage_attribution,
            )?;
            hidden_states = layer_output;
        }
        let hidden_states = ops::grouped_rms_norm(
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
        penalized_token_ids: &[u32],
        random_state: &mut MlxArray,
    ) -> Result<u32, K2HorizonMoVAExecutionError> {
        // The penalty transforms raw logits before temperature, top-p, and the
        // categorical draw, matching the HF reference ordering; a neutral
        // penalty returns None and keeps the sampling graph unchanged.
        let penalized_logits = gpu_token_sampling::apply_repetition_penalty(
            &self.runtime,
            logits,
            penalized_token_ids,
            request.repetition_penalty_thousandths(),
        )
        .map_err(|error| K2HorizonMoVAExecutionError::InvalidExecution {
            description: format!("{error}"),
        })?;
        let logits = penalized_logits.as_ref().unwrap_or(logits);
        let sampled = gpu_token_sampling::build_sampled_token(
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

    /// Runs one prefill-shaped and one decode-shaped dummy forward so the
    /// first real request pays neither the Metal pipeline-state compilation
    /// nor the allocator's first-touch of the resident weights. The prefill
    /// warmup runs at the configured serving chunk width so its compiled
    /// pipeline states match the shapes a real first chunk dispatches; the
    /// decode warmup runs causally unmasked at the resulting offset, which is
    /// exactly the decode shape. The dummy identifiers are spread across the
    /// vocabulary so the routers fan every layer's assignments across all of
    /// its experts: identical dummy rows would all route to the same few
    /// experts and leave the rest of the resident weights' pages cold, which
    /// would hand the first real chunk a multi-second first-touch read of
    /// every expert it routes to. The caches use the serving KV mode — a
    /// warmup on the other storage mode compiles the wrong kernel family and
    /// leaves the first real chunk to pay the quantized or full-precision
    /// path's compilation itself. Every throwaway cache is dropped afterward,
    /// so no served state or request math is affected.
    pub fn run_load_warmup(
        &self,
        chunk_token_count: usize,
        quantized_kv_cache_enabled: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), K2HorizonMoVAExecutionError> {
        let vocabulary_size = self.config.vocab_size().max(1) as usize;
        let prefill_warmup_token_ids = (0..chunk_token_count.max(1))
            .map(|warmup_token_index| ((warmup_token_index * 7919) % vocabulary_size) as u32)
            .collect::<Vec<_>>();
        let mut prefill_caches = self.build_throwaway_caches(quantized_kv_cache_enabled)?;
        let prefill_hidden_states = self.forward(
            &prefill_warmup_token_ids,
            &mut prefill_caches,
            performance_attribution,
            None,
        )?;
        let prefill_logits = self.logits_for_last_token(&prefill_hidden_states)?;
        self.runtime.evaluate_arrays(&[&prefill_logits])?;

        let mut decode_caches = self.build_throwaway_caches(quantized_kv_cache_enabled)?;
        // A single-token forward is causally unmasked regardless of offset,
        // so a fresh cache yields the same kernel shapes as real decode.
        let decode_hidden_states =
            self.forward(&[0], &mut decode_caches, performance_attribution, None)?;
        let decode_logits = self.logits_for_last_token(&decode_hidden_states)?;
        self.runtime.evaluate_arrays(&[&decode_logits])?;
        Ok(())
    }

    fn build_throwaway_caches(
        &self,
        quantized_kv_cache_enabled: bool,
    ) -> Result<Vec<K2HorizonMoVAKvState>, K2HorizonMoVAExecutionError> {
        (0..self.config.num_hidden_layers())
            .map(|_| {
                K2HorizonMoVAKvState::build(quantized_kv_cache_enabled, 256)
                    .map_err(K2HorizonMoVAExecutionError::from)
            })
            .collect()
    }
}

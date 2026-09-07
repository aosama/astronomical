//! Dense GQA, MoVA attention, and dense/sparse FFN for one decoder layer.

use astronomical_runtime_integration::{MlxArray, MlxCompiledSwiGlu, MlxMetalKernel, MlxRuntime};

use crate::PerformanceAttribution;
use crate::decoder_cache::{FullAttentionKeyValueState, QuantizedFullAttentionKeyValueState};
use crate::k2_horizon_mova::configuration::K2HorizonMoVAConfig;

use super::affine::K2HorizonMoVAAffineLinear;
use super::error::K2HorizonMoVAExecutionError;
use super::fused_expert_decode::FusedExpertDecodeKernels;
use super::ops::{
    attention_gate, dense_fused_swiglu, dense_swiglu, gathered_fused_swiglu,
    gathered_value_experts, grouped_rms_norm, merge_heads, reshape_heads, route_k2_experts,
};
use super::quantized_attention::quantized_scaled_dot_product_attention;
use super::weights::{
    K2HorizonMoVADenseAttentionWeights, K2HorizonMoVADenseMlpWeights, K2HorizonMoVALayerWeights,
    K2HorizonMoVAMoVAAttentionWeights, K2HorizonMoVASparseMlpWeights,
};

/// One decoder layer's KV owner: full-precision or quantized slabs.
pub enum K2HorizonMoVAKvState {
    FullPrecision(FullAttentionKeyValueState),
    Quantized(QuantizedFullAttentionKeyValueState),
}

impl K2HorizonMoVAKvState {
    /// Logical token offset shared by both storage modes.
    #[must_use]
    pub fn offset_tokens(&self) -> i32 {
        match self {
            Self::FullPrecision(state) => state.offset_tokens(),
            Self::Quantized(state) => state.offset_tokens(),
        }
    }

    /// Builds one layer's KV owner in the configured storage mode.
    ///
    /// # Errors
    /// Returns an error when the slab geometry is invalid.
    pub fn build(
        quantized_kv_cache_enabled: bool,
        growth_tokens: u32,
    ) -> Result<Self, astronomical_runtime_integration::MlxRuntimeError> {
        if quantized_kv_cache_enabled {
            Ok(Self::Quantized(
                QuantizedFullAttentionKeyValueState::empty_with_growth_tokens(
                    growth_tokens.max(1) as i32,
                    QUANTIZED_KV_GROUP_SIZE,
                    QUANTIZED_KV_BITS,
                )?,
            ))
        } else {
            Ok(Self::FullPrecision(
                FullAttentionKeyValueState::empty_with_growth_tokens(growth_tokens.max(1) as i32)?,
            ))
        }
    }
}

const QUANTIZED_KV_GROUP_SIZE: i32 = 64;
const QUANTIZED_KV_BITS: i32 = 8;

pub fn forward_layer(
    runtime: &MlxRuntime,
    config: &K2HorizonMoVAConfig,
    hidden_states: &MlxArray,
    layer_weights: &K2HorizonMoVALayerWeights,
    cache: &mut K2HorizonMoVAKvState,
    rope_offset: i32,
    is_prefill: bool,
    compiled_swiglu: &MlxCompiledSwiGlu,
    sorted_expert_reduction_kernel: Option<&MlxMetalKernel>,
    fused_expert_decode: Option<&FusedExpertDecodeKernels>,
    performance_attribution: &mut PerformanceAttribution,
    stage_attribution: bool,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    match layer_weights {
        K2HorizonMoVALayerWeights::Dense {
            attention,
            mlp,
            input_norm,
            post_attention_norm,
        } => {
            let normalized = grouped_rms_norm(
                runtime,
                hidden_states,
                input_norm,
                config.layernorm_num_groups(),
                config.rms_norm_eps(),
            )?;
            let attention_output = dense_attention(
                runtime,
                config,
                &normalized,
                attention,
                cache,
                rope_offset,
                is_prefill,
                performance_attribution,
                stage_attribution,
            )?;
            let residual = runtime.add(hidden_states, &attention_output)?;
            let mlp_input = grouped_rms_norm(
                runtime,
                &residual,
                post_attention_norm,
                config.layernorm_num_groups(),
                config.rms_norm_eps(),
            )?;
            let mlp_output = dense_mlp(
                runtime,
                &mlp_input,
                mlp,
                compiled_swiglu,
                performance_attribution,
                stage_attribution,
            )?;
            Ok(runtime.add(&residual, &mlp_output)?)
        }
        K2HorizonMoVALayerWeights::SparseMixtureOfValues {
            attention,
            mlp,
            input_norm,
            post_attention_norm,
        } => {
            let normalized = grouped_rms_norm(
                runtime,
                hidden_states,
                input_norm,
                config.layernorm_num_groups(),
                config.rms_norm_eps(),
            )?;
            let attention_output = mova_attention(
                runtime,
                config,
                &normalized,
                attention,
                cache,
                rope_offset,
                is_prefill,
                fused_expert_decode,
                performance_attribution,
                stage_attribution,
            )?;
            let residual = runtime.add(hidden_states, &attention_output)?;
            let mlp_input = grouped_rms_norm(
                runtime,
                &residual,
                post_attention_norm,
                config.layernorm_num_groups(),
                config.rms_norm_eps(),
            )?;
            let mlp_output = sparse_mlp(
                runtime,
                config,
                &mlp_input,
                mlp,
                compiled_swiglu,
                sorted_expert_reduction_kernel,
                fused_expert_decode,
                performance_attribution,
                stage_attribution,
            )?;
            Ok(runtime.add(&residual, &mlp_output)?)
        }
    }
}

fn dense_attention(
    runtime: &MlxRuntime,
    config: &K2HorizonMoVAConfig,
    hidden_states: &MlxArray,
    weights: &K2HorizonMoVADenseAttentionWeights,
    cache: &mut K2HorizonMoVAKvState,
    rope_offset: i32,
    is_prefill: bool,
    performance_attribution: &mut PerformanceAttribution,
    stage_attribution: bool,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    // One fused quantized projection produces q, k, and v; rows are packed
    // independently so the fused route is bit-identical to three projections.
    let query_key_value = weights.query_key_value.project(runtime, hidden_states)?;
    let projections = K2HorizonMoVAAffineLinear::split_projection_output(
        runtime,
        &query_key_value,
        &[
            weights.query_row_count,
            weights.key_value_row_count,
            weights.key_value_row_count,
        ],
    )?;
    let queries = reshape_heads(
        runtime,
        &projections[0],
        config.num_attention_heads(),
        config.head_dim(),
    )?;
    let keys = reshape_heads(
        runtime,
        &projections[1],
        config.num_key_value_heads(),
        config.head_dim(),
    )?;
    let values = reshape_heads(
        runtime,
        &projections[2],
        config.num_key_value_heads(),
        config.head_dim(),
    )?;
    finish_attention(
        runtime,
        config,
        hidden_states,
        queries,
        keys,
        values,
        &weights.o_proj,
        weights.gate_proj.as_ref(),
        cache,
        rope_offset,
        is_prefill,
        performance_attribution,
        stage_attribution,
    )
}

fn mova_attention(
    runtime: &MlxRuntime,
    config: &K2HorizonMoVAConfig,
    hidden_states: &MlxArray,
    weights: &K2HorizonMoVAMoVAAttentionWeights,
    cache: &mut K2HorizonMoVAKvState,
    rope_offset: i32,
    is_prefill: bool,
    fused_expert_decode: Option<&FusedExpertDecodeKernels>,
    performance_attribution: &mut PerformanceAttribution,
    stage_attribution: bool,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let token_count = hidden_states.shape().get(1).copied().unwrap_or(1);
    let flat = runtime.reshape(
        hidden_states,
        &[
            hidden_states.shape().first().copied().unwrap_or(1) * token_count,
            config.hidden_size() as i32,
        ],
    )?;
    let routed = route_k2_experts(
        runtime,
        &flat,
        &weights.v_router,
        config.mova_num_experts_per_tok(),
        config.mova_num_experts(),
        config.norm_topk_prob(),
        config.router_scaling_factor(),
    )?;
    let values_flat = match fused_expert_decode.filter(|_| token_count == 1) {
        Some(fused_expert_decode) => fused_expert_decode.fused_value_expert_decode(
            runtime,
            &flat,
            &routed.indices,
            &routed.weights,
            &weights.v_experts,
            performance_attribution,
        )?,
        None => gathered_value_experts(
            runtime,
            &flat,
            &weights.v_experts,
            &routed.indices,
            &routed.weights,
            performance_attribution,
        )?,
    };
    let values = reshape_heads(
        runtime,
        &runtime.reshape(
            &values_flat,
            &[
                hidden_states.shape().first().copied().unwrap_or(1),
                token_count,
                (config.num_key_value_heads() * config.head_dim()) as i32,
            ],
        )?,
        config.num_key_value_heads(),
        config.head_dim(),
    )?;
    let query_key = weights.query_key.project(runtime, hidden_states)?;
    let projections = K2HorizonMoVAAffineLinear::split_projection_output(
        runtime,
        &query_key,
        &[weights.query_row_count, weights.key_value_row_count],
    )?;
    let queries = reshape_heads(
        runtime,
        &projections[0],
        config.num_attention_heads(),
        config.head_dim(),
    )?;
    let keys = reshape_heads(
        runtime,
        &projections[1],
        config.num_key_value_heads(),
        config.head_dim(),
    )?;
    if stage_attribution {
        performance_attribution.measure_operation(
            crate::PerformanceOperation::DecodeMixtureOfValuesGraphicsProcessorCompletionWait,
            |_| runtime.evaluate_arrays(&[&values]),
        )?;
    }
    finish_attention(
        runtime,
        config,
        hidden_states,
        queries,
        keys,
        values,
        &weights.o_proj,
        weights.gate_proj.as_ref(),
        cache,
        rope_offset,
        is_prefill,
        performance_attribution,
        stage_attribution,
    )
}

fn finish_attention(
    runtime: &MlxRuntime,
    config: &K2HorizonMoVAConfig,
    hidden_states: &MlxArray,
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
    o_proj: &super::affine::K2HorizonMoVAAffineLinear,
    gate_proj: Option<&super::affine::K2HorizonMoVAAffineLinear>,
    cache: &mut K2HorizonMoVAKvState,
    rope_offset: i32,
    is_prefill: bool,
    performance_attribution: &mut PerformanceAttribution,
    stage_attribution: bool,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let queries = runtime.rope(
        &queries,
        config.rope_head_dim() as i32,
        config.rope_theta(),
        rope_offset,
    )?;
    let keys = runtime.rope(
        &keys,
        config.rope_head_dim() as i32,
        config.rope_theta(),
        rope_offset,
    )?;
    // Append into the over-allocated slab (one copy every `growth_tokens`
    // tokens, in-place writes between) and attend over a view of the written
    // prefix. Re-concatenating the whole cache per step triples KV bandwidth
    // at long context, which is the dominant decode cost.
    let attention = match cache {
        K2HorizonMoVAKvState::FullPrecision(state) => {
            let (keys, values) = state
                .update_and_fetch(runtime, &keys, &values, rope_offset)
                .map_err(
                    |update_error| K2HorizonMoVAExecutionError::InvalidExecution {
                        description: format!("K2 Horizon MoVA KV update failed: {update_error}"),
                    },
                )?;
            let scale = (config.head_dim() as f32).sqrt().recip();
            if is_prefill {
                runtime.causal_scaled_dot_product_attention(&queries, &keys, &values, scale)?
            } else {
                runtime.scaled_dot_product_attention(&queries, &keys, &values, scale)?
            }
        }
        K2HorizonMoVAKvState::Quantized(state) => {
            let views = state
                .update_and_fetch(runtime, &keys, &values, rope_offset)
                .map_err(
                    |update_error| K2HorizonMoVAExecutionError::InvalidExecution {
                        description: format!("K2 Horizon MoVA KV update failed: {update_error}"),
                    },
                )?;
            quantized_scaled_dot_product_attention(
                runtime,
                config,
                &queries,
                &views,
                state.group_size(),
                state.bits(),
                is_prefill,
            )?
        }
    };
    apply_attention_tail(
        runtime,
        config,
        hidden_states,
        attention,
        o_proj,
        gate_proj,
        performance_attribution,
        stage_attribution,
    )
}

/// Quantized attention: the score and value passes read packed KV through the
/// quantized matmul kernel, halving KV bytes read per decode token.
fn apply_attention_tail(
    runtime: &MlxRuntime,
    config: &K2HorizonMoVAConfig,
    hidden_states: &MlxArray,
    attention: MlxArray,
    o_proj: &super::affine::K2HorizonMoVAAffineLinear,
    gate_proj: Option<&super::affine::K2HorizonMoVAAffineLinear>,
    performance_attribution: &mut PerformanceAttribution,
    stage_attribution: bool,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let mut attention = attention;
    if let (Some(gate_proj), Some(gate_func)) = (gate_proj, config.attention_gate_func()) {
        let gate = attention_gate(
            runtime,
            hidden_states,
            gate_proj,
            gate_func,
            config.num_attention_heads(),
            config.head_dim(),
        )?;
        attention = runtime.multiply(&attention, &gate)?;
    }
    let merged = merge_heads(runtime, &attention)?;
    let attention_output = o_proj.project(runtime, &merged)?;
    if stage_attribution {
        performance_attribution.measure_operation(
            crate::PerformanceOperation::DecodeAttentionGraphicsProcessorCompletionWait,
            |_| runtime.evaluate_arrays(&[&attention_output]),
        )?;
    }
    Ok(attention_output)
}

fn dense_mlp(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    weights: &K2HorizonMoVADenseMlpWeights,
    compiled_swiglu: &MlxCompiledSwiGlu,
    performance_attribution: &mut PerformanceAttribution,
    stage_attribution: bool,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let mlp_output = dense_swiglu(
        runtime,
        hidden_states,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
        compiled_swiglu,
    )?;
    if stage_attribution {
        performance_attribution.measure_operation(
            crate::PerformanceOperation::DecodeFeedForwardGraphicsProcessorCompletionWait,
            |_| runtime.evaluate_arrays(&[&mlp_output]),
        )?;
    }
    Ok(mlp_output)
}

fn sparse_mlp(
    runtime: &MlxRuntime,
    config: &K2HorizonMoVAConfig,
    hidden_states: &MlxArray,
    weights: &K2HorizonMoVASparseMlpWeights,
    compiled_swiglu: &MlxCompiledSwiGlu,
    sorted_expert_reduction_kernel: Option<&MlxMetalKernel>,
    fused_expert_decode: Option<&FusedExpertDecodeKernels>,
    performance_attribution: &mut PerformanceAttribution,
    stage_attribution: bool,
) -> Result<MlxArray, K2HorizonMoVAExecutionError> {
    let token_count = hidden_states.shape().get(1).copied().unwrap_or(1);
    let batch = hidden_states.shape().first().copied().unwrap_or(1);
    let flat = runtime.reshape(
        hidden_states,
        &[batch * token_count, config.hidden_size() as i32],
    )?;
    let routed = route_k2_experts(
        runtime,
        &flat,
        &weights.router,
        config.num_experts_per_tok(),
        config.num_experts(),
        config.norm_topk_prob(),
        config.router_scaling_factor(),
    )?;
    let combined = match fused_expert_decode.filter(|_| token_count == 1 && batch == 1) {
        Some(fused_expert_decode) => fused_expert_decode.fused_routed_ffn_decode(
            runtime,
            &flat,
            &routed.indices,
            &routed.weights,
            &weights.switch_gate_up,
            &weights.switch_down,
            performance_attribution,
        )?,
        None => gathered_fused_swiglu(
            runtime,
            &flat,
            &weights.switch_gate_up,
            &weights.switch_down,
            &routed.indices,
            &routed.weights,
            compiled_swiglu,
            sorted_expert_reduction_kernel,
            performance_attribution,
        )?,
    };
    let shared = dense_fused_swiglu(
        runtime,
        &flat,
        &weights.shared_gate_up,
        &weights.shared_down,
        compiled_swiglu,
    )?;
    if stage_attribution {
        performance_attribution.measure_operation(
            crate::PerformanceOperation::DecodeFeedForwardGraphicsProcessorCompletionWait,
            |_| runtime.evaluate_arrays(&[&combined]),
        )?;
    }
    let summed = runtime.add(&combined, &shared)?;
    if stage_attribution {
        performance_attribution.measure_operation(
            crate::PerformanceOperation::DecodeSharedExpertGraphicsProcessorCompletionWait,
            |_| runtime.evaluate_arrays(&[&summed]),
        )?;
    }
    Ok(runtime.reshape(&summed, &[batch, token_count, config.hidden_size() as i32])?)
}

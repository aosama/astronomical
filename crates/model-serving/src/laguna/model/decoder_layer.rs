//! One Laguna decoder layer: residual attention plus descriptor-selected FFN.

use astronomical_runtime_integration::{MlxArray, MlxMetalKernel, MlxRuntime};

use crate::expert_paging::ExpertWeightPage;
use crate::laguna::artifacts::LagunaLayerTensorRole;
use crate::laguna::moe::{
    execute_paged_mixture_on_page, forward_paged_mixture_of_experts,
    forward_resident_mixture_of_experts, forward_retained_complete_mixture_of_experts,
    route_laguna_layer_experts, unique_routed_expert_ids,
};
use crate::laguna::normalization::{LagunaFeedForwardDescriptor, LagunaLayerDescriptor};
use crate::performance_attribution::PerformanceAttribution;

use super::attention::{LagunaAttentionMaskCache, forward_attention};
use super::decoder_state::LagunaDecoderState;
use super::dense_feed_forward::dense_swiglu;
use super::error::LagunaExecutionError;
use super::model::LagunaModel;

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_decoder_layer(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    model: &LagunaModel,
    layer_descriptor: &LagunaLayerDescriptor,
    decoder_state: &mut LagunaDecoderState,
    attention_mask_cache: &mut LagunaAttentionMaskCache,
    rms_norm_epsilon: f32,
    router_logit_softcap: f64,
    sorted_expert_reduction_kernel: Option<&MlxMetalKernel>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    let layer_index = layer_descriptor.layer_index();
    let weights = model.weights();
    let normalized_input = runtime.rms_norm(
        hidden_states,
        weights.layer(layer_index, LagunaLayerTensorRole::InputNormalization)?,
        rms_norm_epsilon,
    )?;
    let attention_delta = forward_attention(
        runtime,
        &normalized_input,
        weights,
        layer_descriptor.attention(),
        layer_index,
        decoder_state,
        attention_mask_cache,
        rms_norm_epsilon,
        performance_attribution,
    )?;
    let after_attention = runtime.add(hidden_states, &attention_delta)?;
    let normalized_after_attention = runtime.rms_norm(
        &after_attention,
        weights.layer(
            layer_index,
            LagunaLayerTensorRole::PostAttentionNormalization,
        )?,
        rms_norm_epsilon,
    )?;
    let feed_forward_delta = match layer_descriptor.feed_forward() {
        LagunaFeedForwardDescriptor::Dense(_) => dense_swiglu(
            runtime,
            &normalized_after_attention,
            weights,
            layer_index,
            model.compiled_swiglu(),
            performance_attribution,
        )?,
        LagunaFeedForwardDescriptor::Moe(moe_descriptor) => {
            let reduction_kernel = sorted_expert_reduction_kernel.ok_or_else(|| {
                LagunaExecutionError::invalid_geometry(
                    "a sparse Laguna layer requires the retained sorted-expert reduction kernel",
                )
            })?;
            forward_sparse_feed_forward(
                runtime,
                &normalized_after_attention,
                model,
                moe_descriptor,
                layer_index,
                router_logit_softcap,
                reduction_kernel,
                performance_attribution,
            )?
        }
    };
    Ok(runtime.add(&after_attention, &feed_forward_delta)?)
}

#[allow(clippy::too_many_arguments)]
fn forward_sparse_feed_forward(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    model: &LagunaModel,
    moe_descriptor: &crate::laguna::normalization::LagunaMoeDescriptor,
    layer_index: usize,
    router_logit_softcap: f64,
    reduction_kernel: &MlxMetalKernel,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, LagunaExecutionError> {
    if model.weights().has_routed_experts(layer_index) {
        return forward_resident_mixture_of_experts(
            runtime,
            hidden_states,
            model.weights(),
            moe_descriptor,
            layer_index,
            router_logit_softcap,
            reduction_kernel,
            model.compiled_swiglu(),
            performance_attribution,
        );
    }
    let sparse_layer_plan = model
        .residency()
        .paging_plan()
        .and_then(|plan| plan.sparse_layer_for_decoder(layer_index))
        .cloned()
        .ok_or_else(|| {
            LagunaExecutionError::invalid_geometry(
                "a sparse Laguna layer without resident routed experts requires a paging plan",
            )
        })?;
    let paging_slot_index = sparse_layer_plan.paging_slot_index();
    if model
        .residency()
        .has_retained_complete_layer(paging_slot_index)
    {
        return model.residency().with_retained_complete_layer(
            paging_slot_index,
            |retained_page| {
                forward_retained_complete_mixture_of_experts(
                    runtime,
                    hidden_states,
                    model.weights(),
                    moe_descriptor,
                    layer_index,
                    retained_page,
                    router_logit_softcap,
                    reduction_kernel,
                    performance_attribution,
                )
            },
        );
    }
    let (selected_indices, selected_scores) = route_laguna_layer_experts(
        runtime,
        model.weights(),
        moe_descriptor,
        layer_index,
        hidden_states,
        router_logit_softcap,
        performance_attribution,
    )?;
    let selected_expert_ids = unique_routed_expert_ids(&selected_indices)?;
    if model
        .residency()
        .retained_page_covers_experts(paging_slot_index, &selected_expert_ids)
    {
        return model.residency().with_retained_complete_layer(
            paging_slot_index,
            |retained_page| {
                execute_paged_mixture_on_page(
                    runtime,
                    hidden_states,
                    model.weights(),
                    moe_descriptor,
                    layer_index,
                    retained_page,
                    &selected_indices,
                    &selected_scores,
                    reduction_kernel,
                    performance_attribution,
                )
            },
        );
    }
    let (output, last_forward, streamed_page) = forward_paged_mixture_of_experts(
        runtime,
        hidden_states,
        model.weights(),
        moe_descriptor,
        layer_index,
        &sparse_layer_plan,
        router_logit_softcap,
        reduction_kernel,
        performance_attribution,
    )?;
    model.residency().record_disk_page_load();
    model.residency().record_forward(last_forward);
    let hidden_token_count = hidden_states
        .shape()
        .get(hidden_states.shape().len().saturating_sub(2))
        .copied()
        .unwrap_or(1);
    if hidden_token_count > 1 {
        if loaded_page_can_remain_resident_for_next_page(
            runtime,
            streamed_page.resident_payload_byte_count(),
        )? {
            model.residency().try_commit_complete_layer(
                paging_slot_index,
                sparse_layer_plan.expert_capacity(),
                streamed_page,
                performance_attribution,
            )?;
        }
    } else {
        if loaded_page_can_remain_resident_for_next_page(
            runtime,
            streamed_page.resident_payload_byte_count(),
        )? {
            model.residency().try_commit_routed_page(
                paging_slot_index,
                sparse_layer_plan.expert_capacity(),
                selected_expert_ids,
                streamed_page,
                performance_attribution,
            )?;
        }
    }
    Ok(output)
}

/// Keeps one page only when the next mandatory page still fits below MLX's
/// strict configured ceiling. The current active count already includes the
/// just-executed page; two additional slots cover its evaluated boundary graph
/// and the next mandatory page allocation.
fn loaded_page_can_remain_resident_for_next_page(
    runtime: &MlxRuntime,
    next_page_payload_bytes: u64,
) -> Result<bool, LagunaExecutionError> {
    let memory_snapshot = runtime.memory_snapshot()?;
    let current_active_memory_bytes =
        u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(u64::MAX);
    let configured_memory_ceiling_bytes =
        u64::try_from(runtime.memory_limits().active_memory_limit_bytes()).unwrap_or(u64::MAX);
    let next_two_page_payload_bytes = next_page_payload_bytes.saturating_mul(2);
    Ok(
        current_active_memory_bytes.saturating_add(next_two_page_payload_bytes)
            <= configured_memory_ceiling_bytes,
    )
}

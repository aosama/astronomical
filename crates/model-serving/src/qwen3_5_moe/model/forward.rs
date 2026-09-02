//! Sparse routing selects resident experts, cached experts, or SSD streaming.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

use super::super::expert_paging::expert_pager::Qwen3_5ExpertPager;
use super::Qwen3_5MoEPagedPrefillExecutionMode;
use super::expert_reuse::ExpertPageDisposition;
use super::feed_forward_weights::{Qwen3_5MoEFeedForwardWeights, Qwen3_5MoERouterGateWeights};
use super::routing::qwen3_5_moe_route_experts;

impl Qwen3_5Model {
    /// Routes once, then selects contiguous resident arrays or native paging.
    // Paged execution inputs stay explicit rather than introducing a parameter facade.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_qwen3_5_moe(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        should_use_compiled_elementwise_graphs: bool,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let hidden_state_shape = hidden_states.shape();
        let token_count = hidden_state_shape
            .get(hidden_state_shape.len().saturating_sub(2))
            .copied()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "paged MoE hidden states must include token and hidden dimensions",
            })?;
        let (selected_indices, selected_scores) = performance_attribution.measure_operation(
            if self.resident_expert_weights.is_some()
                && matches!(
                    paged_prefill_execution_mode,
                    Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault
                        | Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
                )
            {
                PerformanceOperation::ResidentMoeGraphConstruction
            } else {
                PerformanceOperation::PagedRouterGraphConstruction
            },
            |_performance_attribution| {
                let router_logits = match &mixture_of_experts_weights.router_projection {
                    Qwen3_5MoERouterGateWeights::Affine(quantized_weights) => self
                        .quantized_linear_for_paged_prefill_execution_mode(
                            hidden_states,
                            quantized_weights,
                            paged_prefill_execution_mode,
                        )?,
                    Qwen3_5MoERouterGateWeights::Unquantized(unquantized_weight) => {
                        let transposed_gate_weight =
                            self.runtime.transpose_axes(unquantized_weight, &[1, 0])?;
                        if paged_prefill_execution_mode
                            == Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
                            && token_count > 1
                        {
                            let hidden_dimension = hidden_state_shape.last().copied().ok_or(
                                Qwen3_5ExecutionError::InvalidInput {
                                    description: "paged MoE hidden dimension is missing",
                                },
                            )?;
                            let mut token_router_logits = Vec::with_capacity(token_count as usize);
                            for token_position_index in 0..token_count {
                                let token_hidden_states = self.runtime.slice(
                                    hidden_states,
                                    &[0, token_position_index, 0],
                                    &[1, token_position_index + 1, hidden_dimension],
                                    &[1, 1, 1],
                                )?;
                                token_router_logits.push(
                                    self.runtime
                                        .matmul(&token_hidden_states, &transposed_gate_weight)?,
                                );
                            }
                            let token_router_logit_references =
                                token_router_logits.iter().collect::<Vec<_>>();
                            self.runtime
                                .concatenate_axis(&token_router_logit_references, 1)?
                        } else {
                            self.runtime
                                .matmul(hidden_states, &transposed_gate_weight)?
                        }
                    }
                };
                qwen3_5_moe_route_experts(
                    &self.runtime,
                    &router_logits,
                    self.config.experts_per_token() as i32,
                    self.config.normalizes_top_k_probabilities(),
                )
                .map_err(Qwen3_5ExecutionError::from)
            },
        )?;

        let should_use_loaded_model_mode = matches!(
            paged_prefill_execution_mode,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault
                | Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
        );
        if should_use_loaded_model_mode
            && let Some(resident_expert_layer_weights) = self
                .resident_expert_weights
                .as_ref()
                .and_then(|resident_expert_weights| resident_expert_weights.layer(layer_index))
        {
            return self.forward_moe_resident_with_performance_attribution(
                hidden_states,
                mixture_of_experts_weights,
                resident_expert_layer_weights,
                &selected_indices,
                &selected_scores,
                should_use_compiled_elementwise_graphs,
                paged_prefill_execution_mode,
                performance_attribution,
            );
        }

        // A seated complete layer remaps every route on-device. Do not copy
        // router indices to the host for that layer.
        if should_use_loaded_model_mode && self.retained_experts.is_some() {
            let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
            let complete_expert_ids: Vec<usize> = (0..expert_capacity).collect();
            if matches!(
                self.expert_page_disposition(layer_index, expert_capacity),
                ExpertPageDisposition::FullHit,
            ) {
                if let Some((packed_weights, packed_manifest)) =
                    self.cached_packed_page(layer_index, &complete_expert_ids, expert_capacity)
                {
                    performance_attribution.record_counter(
                        PerformanceCounter::AvoidedCompleteLayerExpertSourcePayloadBytes,
                        packed_manifest.payload_byte_count,
                    );
                    if paged_prefill_execution_mode
                        == Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
                    {
                        return self
                            .forward_moe_paged_target_verification_with_performance_attribution(
                                hidden_states,
                                mixture_of_experts_weights,
                                &packed_weights,
                                &packed_manifest,
                                &selected_indices,
                                &complete_expert_ids,
                                &selected_scores,
                                performance_attribution,
                            );
                    }
                    return self.forward_moe_paged_with_performance_attribution(
                        hidden_states,
                        mixture_of_experts_weights,
                        &packed_weights,
                        &packed_manifest,
                        &selected_indices,
                        &complete_expert_ids,
                        &selected_scores,
                        should_use_compiled_elementwise_graphs,
                        performance_attribution,
                    );
                }
            }
        }

        let selected_expert_ids = if should_use_loaded_model_mode {
            Some(self.copy_selected_expert_ids(&selected_indices)?)
        } else {
            None
        };
        let sorted_unique_expert_ids = selected_expert_ids.as_ref().map(|selected_expert_ids| {
            let mut routed_expert_ids = selected_expert_ids.clone();
            routed_expert_ids.sort_unstable();
            routed_expert_ids.dedup();
            routed_expert_ids
        });
        if should_use_loaded_model_mode
            && let Some(selected_expert_ids) = selected_expert_ids.as_ref()
            && let Some(routed_expert_ids) = sorted_unique_expert_ids.as_ref()
            && self.retained_experts.is_some()
        {
            let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
            if let Some(retained_experts) = self.retained_experts.as_ref() {
                retained_experts.borrow_mut().record_expert_demand(
                    layer_index,
                    expert_capacity,
                    selected_expert_ids,
                );
            }
            let disposition = self.expert_page_disposition(layer_index, expert_capacity);
            match disposition {
                ExpertPageDisposition::FullHit => {
                    let complete_expert_ids: Vec<usize> = (0..expert_capacity).collect();
                    let (packed_weights, packed_manifest) = self
                        .cached_packed_page(layer_index, &complete_expert_ids, expert_capacity)
                        .ok_or(Qwen3_5ExecutionError::InvalidInput {
                            description: "cache hit reported a page that vanished",
                        })?;
                    performance_attribution.record_counter(
                        PerformanceCounter::RetainedRouteAssignmentHitCount,
                        u64::try_from(selected_expert_ids.len()).unwrap_or(u64::MAX),
                    );
                    if paged_prefill_execution_mode
                        == Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
                    {
                        return self
                            .forward_moe_paged_target_verification_with_performance_attribution(
                                hidden_states,
                                mixture_of_experts_weights,
                                &packed_weights,
                                &packed_manifest,
                                &selected_indices,
                                &complete_expert_ids,
                                &selected_scores,
                                performance_attribution,
                            );
                    }
                    return self.forward_moe_paged_with_performance_attribution(
                        hidden_states,
                        mixture_of_experts_weights,
                        &packed_weights,
                        &packed_manifest,
                        &selected_indices,
                        &complete_expert_ids,
                        &selected_scores,
                        should_use_compiled_elementwise_graphs,
                        performance_attribution,
                    );
                }
                ExpertPageDisposition::Miss => {
                    performance_attribution.record_counter(
                        PerformanceCounter::RetainedRouteAssignmentMissCount,
                        u64::try_from(selected_expert_ids.len()).unwrap_or(u64::MAX),
                    );
                    let (packed_weights, packed_manifest) = if token_count > 1 {
                        self.stream_complete_expert_layer(
                            expert_pager,
                            layer_index,
                            token_count,
                            expert_capacity,
                            routed_expert_ids.len(),
                            paged_prefill_execution_mode
                                == Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                            performance_attribution,
                        )?
                    } else if let Some((cached_weights, cached_manifest)) = self
                        .hot_expert_partial_page(
                            layer_index,
                            expert_capacity,
                            routed_expert_ids,
                            performance_attribution,
                        )
                    {
                        // Hot-expert cache hit: the warm table covers every
                        // routed expert of this token, so the routed set is
                        // served from retained RAM with no storage read
                        // (issue #372).
                        (cached_weights, cached_manifest)
                    } else {
                        self.stream_operation_local_routed_experts(
                            expert_pager,
                            layer_index,
                            token_count,
                            routed_expert_ids,
                            paged_prefill_execution_mode
                                == Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                            performance_attribution,
                        )?
                    };
                    let complete_expert_ids: Vec<usize> = (0..expert_capacity).collect();
                    let gather_expert_ids = if token_count > 1 {
                        complete_expert_ids.as_slice()
                    } else {
                        routed_expert_ids
                    };
                    if paged_prefill_execution_mode
                        == Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
                    {
                        return self
                            .forward_moe_paged_target_verification_with_performance_attribution(
                                hidden_states,
                                mixture_of_experts_weights,
                                &packed_weights,
                                &packed_manifest,
                                &selected_indices,
                                gather_expert_ids,
                                &selected_scores,
                                performance_attribution,
                            );
                    }
                    return self.forward_moe_paged_with_performance_attribution(
                        hidden_states,
                        mixture_of_experts_weights,
                        &packed_weights,
                        &packed_manifest,
                        &selected_indices,
                        gather_expert_ids,
                        &selected_scores,
                        should_use_compiled_elementwise_graphs,
                        performance_attribution,
                    );
                }
            }
        }

        if token_count > 1 {
            match paged_prefill_execution_mode {
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault => {
                    return self.forward_moe_with_expert_store_paging(
                        hidden_states,
                        mixture_of_experts_weights,
                        expert_pager,
                        layer_index,
                        token_count,
                        &selected_indices,
                        &selected_scores,
                        should_use_compiled_elementwise_graphs,
                        false,
                        paged_prefill_execution_mode,
                        sorted_unique_expert_ids.as_deref(),
                        performance_attribution,
                    );
                }
                Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow => {
                    return self.forward_moe_with_expert_store_paging(
                        hidden_states,
                        mixture_of_experts_weights,
                        expert_pager,
                        layer_index,
                        token_count,
                        &selected_indices,
                        &selected_scores,
                        should_use_compiled_elementwise_graphs,
                        true,
                        paged_prefill_execution_mode,
                        sorted_unique_expert_ids.as_deref(),
                        performance_attribution,
                    );
                }
                Qwen3_5MoEPagedPrefillExecutionMode::CompactPromptDiagnostic => {
                    return self.forward_moe_with_direct_prefill_paging(
                        hidden_states,
                        mixture_of_experts_weights,
                        expert_pager,
                        layer_index,
                        token_count,
                        &selected_indices,
                        &selected_scores,
                        should_use_compiled_elementwise_graphs,
                        performance_attribution,
                    );
                }
                Qwen3_5MoEPagedPrefillExecutionMode::TokenLocalDiagnostic => {
                    return self.forward_moe_with_per_token_paging(
                        hidden_states,
                        mixture_of_experts_weights,
                        expert_pager,
                        layer_index,
                        &selected_indices,
                        &selected_scores,
                        should_use_compiled_elementwise_graphs,
                        performance_attribution,
                    );
                }
            }
        }

        self.forward_moe_with_expert_store_paging(
            hidden_states,
            mixture_of_experts_weights,
            expert_pager,
            layer_index,
            token_count,
            &selected_indices,
            &selected_scores,
            should_use_compiled_elementwise_graphs,
            false,
            paged_prefill_execution_mode,
            sorted_unique_expert_ids.as_deref(),
            performance_attribution,
        )
    }
}

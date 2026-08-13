//! Shared sparse routing followed by one binary expert-weight owner.
//!
//! Routing is identical in both modes and keeps global expert identifiers. In
//! production modes, a published complete resident owner selects ordinary MLX
//! gathered projections; otherwise the retained pager prepares native one-expert
//! pages on demand. Diagnostic paging modes intentionally bypass resident arrays
//! so they continue to qualify the native fallback rather than the selected mode.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{ExpertSourceRequestPhase, PerformanceAttribution, PerformanceOperation};

use super::super::expert_paging::expert_pager::Qwen3_5ExpertPager;
use super::Qwen3_5MoEPagedPrefillExecutionMode;
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
        // Route experts using resident router weights to determine which
        // experts to load from disk and to provide routing scores for the
        // weighted combination step.
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

        // Target verification is a production MTP path and therefore follows
        // the loaded owner. The two diagnostic modes force paging by design.
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
            let avoided_source_evidence = if token_count > 1 {
                expert_pager.complete_layer_source_evidence(layer_index)?
            } else {
                let sorted_unique_expert_ids =
                    self.copy_sorted_unique_expert_ids(&selected_indices, performance_attribution)?;
                expert_pager
                    .source_evidence_for_expert_ids(layer_index, &sorted_unique_expert_ids)?
            };
            performance_attribution.record_expert_source_resident_hit(
                layer_index,
                expert_source_request_phase(token_count),
                avoided_source_evidence.payload_bytes,
                avoided_source_evidence.source_interval_count,
            );
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

        // Reuse a warm complete layer for decode or multi-token prefill when the
        // budget owner still admits it. Do not drop warm layers just because a
        // multi-token prefill started; only live budget pressure may shrink them.
        if should_use_loaded_model_mode
            && let Some(retained_expert_layers) = self.retained_expert_layers.as_ref()
            && let Some(retained_expert_layer) =
                retained_expert_layers.borrow().retained_layer(layer_index)
        {
            let sorted_unique_expert_ids =
                self.copy_sorted_unique_expert_ids(&selected_indices, performance_attribution)?;
            let avoided_source_evidence = if token_count > 1 {
                expert_pager.complete_layer_source_evidence(layer_index)?
            } else {
                expert_pager
                    .source_evidence_for_expert_ids(layer_index, &sorted_unique_expert_ids)?
            };
            performance_attribution.record_expert_source_resident_hit(
                layer_index,
                expert_source_request_phase(token_count),
                avoided_source_evidence.payload_bytes,
                avoided_source_evidence.source_interval_count,
            );
            if paged_prefill_execution_mode
                == Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
            {
                return self.forward_moe_paged_target_verification_with_performance_attribution(
                    hidden_states,
                    mixture_of_experts_weights,
                    &retained_expert_layer.weights,
                    &retained_expert_layer.manifest,
                    &selected_indices,
                    &sorted_unique_expert_ids,
                    &selected_scores,
                    performance_attribution,
                );
            }
            return self.forward_moe_paged_with_performance_attribution(
                hidden_states,
                mixture_of_experts_weights,
                &retained_expert_layer.weights,
                &retained_expert_layer.manifest,
                &selected_indices,
                &sorted_unique_expert_ids,
                &selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            );
        }

        // Reaching this branch means there is no eligible complete resident
        // owner. Every path below prepares weights through the Rust pager.
        if token_count > 1 {
            match paged_prefill_execution_mode {
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault => {
                    return self.forward_moe_with_layer_store_paging(
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
                        performance_attribution,
                    );
                }
                Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow => {
                    return self.forward_moe_with_layer_store_paging(
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
                        performance_attribution,
                    );
                }
                Qwen3_5MoEPagedPrefillExecutionMode::CompactPromptDiagnostic => {
                    return self.forward_moe_with_direct_prefill_paging(
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

        self.forward_moe_with_layer_store_paging(
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
            performance_attribution,
        )
    }

    // Paging dependencies stay explicit instead of adding a request-context abstraction.
    #[allow(clippy::too_many_arguments)]
    fn forward_moe_with_layer_store_paging(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_token_count: i32,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        should_execute_token_projections_separately: bool,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let sorted_unique_expert_ids =
            self.copy_sorted_unique_expert_ids(selected_indices, performance_attribution)?;
        let streamed_expert_ids = if route_token_count > 1 {
            (0..expert_pager.layer_plan(layer_index)?.expert_capacity).collect::<Vec<_>>()
        } else {
            sorted_unique_expert_ids.clone()
        };
        let (streamed_expert_weights, page_manifest) = expert_pager
            .load_rust_streamed_expert_layer(
                &self.runtime,
                layer_index,
                &streamed_expert_ids,
                expert_source_request_phase(route_token_count),
                performance_attribution,
            )?;
        if let Some(retained_expert_layers) = self.retained_expert_layers.as_ref() {
            retained_expert_layers.borrow_mut().record_disk_load(
                streamed_expert_ids.len(),
                page_manifest.source_manifests.len(),
            );
        }
        performance_attribution.measure_operation(
            PerformanceOperation::ExpertPagingDiagnosticLogging,
            |_performance_attribution| {
                tracing::debug!(
                    layer_index,
                    route_expert_count = sorted_unique_expert_ids.len(),
                    streamed_expert_count = streamed_expert_ids.len(),
                    streamed_payload_bytes = page_manifest.payload_byte_count,
                    ?paged_prefill_execution_mode,
                    "Rust expert layer streaming completed"
                );
            },
        );
        if should_execute_token_projections_separately {
            return self.forward_moe_paged_target_verification_with_performance_attribution(
                hidden_states,
                mixture_of_experts_weights,
                &streamed_expert_weights,
                &page_manifest,
                selected_indices,
                &streamed_expert_ids,
                selected_scores,
                performance_attribution,
            );
        }
        // Multi-token pages stay operation-local: execute, then drop. Pinning every
        // complete layer during prefill forces near-ceiling MLX residency and
        // collapses the measured streaming path.
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &streamed_expert_weights,
            &page_manifest,
            selected_indices,
            &sorted_unique_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }

    // Paging dependencies stay explicit instead of adding a request-context abstraction.
    #[allow(clippy::too_many_arguments)]
    fn forward_moe_with_direct_prefill_paging(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let sorted_unique_expert_ids =
            self.copy_sorted_unique_expert_ids(selected_indices, performance_attribution)?;
        let (streamed_expert_weights, page_manifest) = expert_pager
            .load_rust_streamed_expert_layer(
                &self.runtime,
                layer_index,
                &sorted_unique_expert_ids,
                ExpertSourceRequestPhase::Prefill,
                performance_attribution,
            )?;
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &streamed_expert_weights,
            &page_manifest,
            selected_indices,
            &sorted_unique_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }

    // This diagnostic mirrors the explicit production inputs for meaningful comparisons.
    #[allow(clippy::too_many_arguments)]
    fn forward_moe_with_per_token_paging(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let hidden_state_shape = hidden_states.shape();
        if hidden_state_shape.len() != 3 || hidden_state_shape[0] != 1 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "paged prefill MoE currently expects [1, tokens, hidden] states",
            });
        }
        let token_count = hidden_state_shape[1];
        let hidden_dimension = hidden_state_shape[2];
        let expert_count_per_token = selected_indices.shape().last().copied().ok_or(
            Qwen3_5ExecutionError::InvalidInput {
                description: "paged MoE selected indices must include an expert dimension",
            },
        )?;
        let mut token_moe_outputs = Vec::with_capacity(token_count as usize);

        for token_index in 0..token_count {
            let token_hidden_states = self.runtime.slice(
                hidden_states,
                &[0, token_index, 0],
                &[1, token_index + 1, hidden_dimension],
                &[1, 1, 1],
            )?;
            let token_selected_indices = self.runtime.slice(
                selected_indices,
                &[0, token_index, 0],
                &[1, token_index + 1, expert_count_per_token],
                &[1, 1, 1],
            )?;
            let token_selected_scores = self.runtime.slice(
                selected_scores,
                &[0, token_index, 0],
                &[1, token_index + 1, expert_count_per_token],
                &[1, 1, 1],
            )?;
            let token_moe_output = self.forward_moe_with_layer_store_paging(
                &token_hidden_states,
                mixture_of_experts_weights,
                expert_pager,
                layer_index,
                token_count,
                &token_selected_indices,
                &token_selected_scores,
                should_use_compiled_elementwise_graphs,
                false,
                Qwen3_5MoEPagedPrefillExecutionMode::TokenLocalDiagnostic,
                performance_attribution,
            )?;
            token_moe_outputs.push(token_moe_output);
        }

        let token_moe_output_references = token_moe_outputs.iter().collect::<Vec<_>>();
        Ok(self
            .runtime
            .concatenate_axis(&token_moe_output_references, 1)?)
    }

    fn copy_sorted_unique_expert_ids(
        &self,
        selected_indices: &MlxArray,
        _performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Vec<usize>, Qwen3_5ExecutionError> {
        let contiguous_ids = self
            .runtime
            .build_contiguous_row_major_copy(selected_indices)?;
        contiguous_ids.evaluate()?;
        let selected_ids = contiguous_ids.copy_evaluated_u32_values()?;
        let mut sorted_unique_ids = selected_ids
            .into_iter()
            .map(|expert_id| expert_id as usize)
            .collect::<Vec<_>>();
        sorted_unique_ids.sort_unstable();
        sorted_unique_ids.dedup();
        Ok(sorted_unique_ids)
    }
}

fn expert_source_request_phase(token_count: i32) -> ExpertSourceRequestPhase {
    if token_count > 1 {
        ExpertSourceRequestPhase::Prefill
    } else {
        ExpertSourceRequestPhase::Decode
    }
}

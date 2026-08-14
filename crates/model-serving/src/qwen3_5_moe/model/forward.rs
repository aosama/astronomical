//! Sparse routing selects complete residency, retained pages, or routed streaming.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{PagedDecodeLayerDisposition, PerformanceAttribution, PerformanceOperation};

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

        // A complete retained page can remap every possible route on-device.
        // Avoid synchronously copying router output to the host for this layer.
        if should_use_loaded_model_mode
            && let Some(retained_expert_layers) = self.retained_expert_layers.as_ref()
            && let Some(retained_expert_layer) =
                retained_expert_layers.borrow().retained_layer(layer_index)
            && retained_expert_layer.manifest.contains_all_experts()
        {
            if paged_prefill_execution_mode
                == Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
            {
                return self.forward_moe_paged_target_verification_with_performance_attribution(
                    hidden_states,
                    mixture_of_experts_weights,
                    &retained_expert_layer.weights,
                    &retained_expert_layer.manifest,
                    &selected_indices,
                    &retained_expert_layer.manifest.expert_ids,
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
                &retained_expert_layer.manifest.expert_ids,
                &selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            );
        }

        let selected_expert_ids = if should_use_loaded_model_mode {
            Some(self.copy_selected_expert_ids(&selected_indices)?)
        } else {
            None
        };
        let sorted_unique_expert_ids =
            if let Some(selected_expert_ids) = selected_expert_ids.as_ref() {
                if let Some(retained_expert_layers) = self.retained_expert_layers.as_ref() {
                    retained_expert_layers.borrow_mut().record_expert_demand(
                        layer_index,
                        expert_pager.layer_plan(layer_index)?.expert_capacity,
                        selected_expert_ids,
                    );
                }
                let mut sorted_unique_expert_ids = selected_expert_ids.clone();
                sorted_unique_expert_ids.sort_unstable();
                sorted_unique_expert_ids.dedup();
                Some(sorted_unique_expert_ids)
            } else {
                None
            };

        // Reuse a demand-selected page directly when it covers the complete route.
        if should_use_loaded_model_mode
            && let Some(retained_expert_layers) = self.retained_expert_layers.as_ref()
            && let Some(sorted_unique_expert_ids) = sorted_unique_expert_ids.as_ref()
            && let Some(retained_expert_layer) =
                retained_expert_layers.borrow().retained_layer(layer_index)
            && retained_expert_layer.contains_every_expert(sorted_unique_expert_ids)
        {
            if paged_prefill_execution_mode
                == Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
            {
                return self.forward_moe_paged_target_verification_with_performance_attribution(
                    hidden_states,
                    mixture_of_experts_weights,
                    &retained_expert_layer.weights,
                    &retained_expert_layer.manifest,
                    &selected_indices,
                    sorted_unique_expert_ids,
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
                sorted_unique_expert_ids,
                &selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            );
        }

        // A partial retained page still owns useful routed experts. Stream only
        // route misses, then partition assignments so each executes exactly once.
        if should_use_loaded_model_mode
            && token_count == 1
            && let Some(retained_expert_layers) = self.retained_expert_layers.as_ref()
            && let Some(sorted_unique_expert_ids) = sorted_unique_expert_ids.as_ref()
            && let Some(selected_expert_ids) = selected_expert_ids.as_ref()
        {
            // The cache borrow must die before stream or disk-load recording.
            // `from_retained_page` copies the decision into an owned value.
            // Returning into `forward_moe_with_layer_store_paging` while
            // `retained_expert_cache` is still borrowed panics the inference
            // owner with `RefCell already borrowed` because streaming records
            // a disk load through `borrow_mut()`.
            let decode_layer_disposition = {
                let retained_expert_cache = retained_expert_layers.borrow();
                PagedDecodeLayerDisposition::from_retained_page(
                    retained_expert_cache
                        .retained_layer(layer_index)
                        .map(|retained_expert_layer| &retained_expert_layer.manifest),
                    selected_expert_ids,
                )
            };
            match decode_layer_disposition {
                PagedDecodeLayerDisposition::StreamEntireLayer => {
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
                        Some(sorted_unique_expert_ids),
                        performance_attribution,
                    );
                }
                PagedDecodeLayerDisposition::SplitRetainedAndMissing(route_partition) => {
                    let (missing_expert_weights, missing_page_manifest) = expert_pager
                        .load_rust_streamed_expert_layer(
                            &self.runtime,
                            layer_index,
                            &route_partition.missing_expert_ids,
                            performance_attribution,
                        )?;
                    retained_expert_layers.borrow_mut().record_disk_load(
                        route_partition.missing_expert_ids.len(),
                        missing_page_manifest.source_manifests.len(),
                    );
                    let retained_expert_cache = retained_expert_layers.borrow();
                    let retained_expert_layer = retained_expert_cache
                        .retained_layer(layer_index)
                        .ok_or(Qwen3_5ExecutionError::InvalidInput {
                        description: "retained expert page disappeared during decode",
                    })?;
                    return self.forward_moe_split_paged_with_performance_attribution(
                        hidden_states,
                        mixture_of_experts_weights,
                        &retained_expert_layer.weights,
                        &retained_expert_layer.manifest,
                        &missing_expert_weights,
                        &missing_page_manifest,
                        &selected_indices,
                        &selected_scores,
                        &route_partition,
                        should_use_compiled_elementwise_graphs,
                        performance_attribution,
                    );
                }
            }
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
                        sorted_unique_expert_ids.as_deref(),
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
            sorted_unique_expert_ids.as_deref(),
            performance_attribution,
        )
    }

    // Paging dependencies stay explicit instead of adding a request-context abstraction.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_with_layer_store_paging(
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
        known_sorted_unique_expert_ids: Option<&[usize]>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let copied_sorted_unique_expert_ids;
        let sorted_unique_expert_ids = match known_sorted_unique_expert_ids {
            Some(sorted_unique_expert_ids) => sorted_unique_expert_ids,
            None => {
                copied_sorted_unique_expert_ids =
                    self.copy_sorted_unique_expert_ids(selected_indices)?;
                &copied_sorted_unique_expert_ids
            }
        };
        let streamed_expert_ids = if route_token_count > 1 {
            (0..expert_pager.layer_plan(layer_index)?.expert_capacity).collect::<Vec<_>>()
        } else {
            sorted_unique_expert_ids.to_vec()
        };
        let (streamed_expert_weights, page_manifest) = expert_pager
            .load_rust_streamed_expert_layer(
                &self.runtime,
                layer_index,
                &streamed_expert_ids,
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
            sorted_unique_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }
}

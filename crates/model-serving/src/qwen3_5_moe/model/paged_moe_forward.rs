//! Paged MoE forward pass for prefill and decode with expert paging.
//!
//! When expert paging is enabled, prefill and decode steps route experts using resident
//! router weights, retrieve the top-K selected expert weights, and compute the
//! MoE output using those paged weights. A retained complete layer keeps routing
//! indices on the graphics processor; otherwise prefill uses a direct page and
//! decode uses independently retained one-expert pages.

use astronomical_runtime_integration::MlxArray;

use crate::{PerformanceAttribution, PerformanceOperation};

use super::Qwen3_5MoEExecutionError;
use super::decoder_layer_weights::Qwen3_5MoEMixtureOfExpertsWeights;
use super::expert_paging::expert_pager::{ExpertPager, ExpertPagingError};
use super::moe::qwen3_5_moe_route_experts;
use super::{Qwen3_5MoEModel, Qwen3_5MoEPagedPrefillExecutionMode};

const FULL_LAYER_PAGE_MINIMUM_SELECTED_EXPERT_NUMERATOR: usize = 3;
const FULL_LAYER_PAGE_MINIMUM_SELECTED_EXPERT_DENOMINATOR: usize = 4;

impl Qwen3_5MoEModel {
    /// Paged MoE forward pass for prefill and decode steps.
    ///
    /// Routes experts using resident router weights, extracts the selected expert
    /// IDs, retrieves those experts through direct prefill pages or the decode
    /// memory cache, then computes the MoE output using pre-computed routing.
    // Paged execution inputs stay explicit rather than introducing a parameter facade.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_with_paging(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEMixtureOfExpertsWeights,
        expert_pager: &ExpertPager,
        layer_index: usize,
        should_use_compiled_elementwise_graphs: bool,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let hidden_state_shape = hidden_states.shape();
        let token_count = hidden_state_shape
            .get(hidden_state_shape.len().saturating_sub(2))
            .copied()
            .ok_or(Qwen3_5MoEExecutionError::InvalidInput {
                description: "paged MoE hidden states must include token and hidden dimensions",
            })?;
        // Route experts using resident router weights to determine which
        // experts to load from disk and to provide routing scores for the
        // weighted combination step.
        let (selected_indices, selected_scores) = performance_attribution.measure_operation(
            PerformanceOperation::PagedRouterGraphConstruction,
            |_performance_attribution| {
                let router_logits = match &mixture_of_experts_weights.router_projection {
                    super::decoder_layer_weights::RouterGateWeights::Affine(quantized_weights) => {
                        self.quantized_linear(hidden_states, quantized_weights)?
                    }
                    super::decoder_layer_weights::RouterGateWeights::Unquantized(
                        unquantized_weight,
                    ) => {
                        let transposed_gate_weight =
                            self.runtime.transpose_axes(unquantized_weight, &[1, 0])?;
                        self.runtime
                            .matmul(hidden_states, &transposed_gate_weight)?
                    }
                };
                qwen3_5_moe_route_experts(
                    &self.runtime,
                    &router_logits,
                    self.config.experts_per_token() as i32,
                    self.config.normalizes_top_k_probabilities(),
                )
                .map_err(Qwen3_5MoEExecutionError::from)
            },
        )?;

        if matches!(
            paged_prefill_execution_mode,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault
                | Qwen3_5MoEPagedPrefillExecutionMode::ProductionDecodeVerification
        ) {
            let mut expert_weight_memory_cache = self.expert_weight_memory_cache.borrow_mut();
            if let Some(complete_layer_expert_weights) =
                expert_weight_memory_cache.record_complete_layer_hit(layer_index)
            {
                return self.forward_moe_with_complete_layer_paging_and_performance_attribution(
                    hidden_states,
                    mixture_of_experts_weights,
                    complete_layer_expert_weights,
                    &selected_indices,
                    &selected_scores,
                    should_use_compiled_elementwise_graphs,
                    performance_attribution,
                );
            }
        }

        if token_count > 1 {
            let should_allow_full_layer_page = match paged_prefill_execution_mode {
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault => true,
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDecodeVerification => {
                    return self.forward_moe_with_cached_paging(
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
                Qwen3_5MoEPagedPrefillExecutionMode::CompactMultiTokenDiagnostic => false,
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
            };
            return self.forward_moe_with_direct_prefill_paging(
                hidden_states,
                mixture_of_experts_weights,
                expert_pager,
                layer_index,
                &selected_indices,
                &selected_scores,
                should_use_compiled_elementwise_graphs,
                should_allow_full_layer_page,
                performance_attribution,
            );
        }

        self.forward_moe_with_cached_paging(
            hidden_states,
            mixture_of_experts_weights,
            expert_pager,
            layer_index,
            token_count,
            &selected_indices,
            &selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }

    // Paging dependencies stay explicit instead of adding a request-context abstraction.
    #[allow(clippy::too_many_arguments)]
    fn forward_moe_with_cached_paging(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEMixtureOfExpertsWeights,
        expert_pager: &ExpertPager,
        layer_index: usize,
        route_token_count: i32,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let sorted_unique_expert_ids =
            self.copy_sorted_unique_expert_ids(selected_indices, performance_attribution)?;
        performance_attribution.record_previous_token_expert_route_reuse(
            layer_index,
            route_token_count,
            &sorted_unique_expert_ids,
        );

        // Load paged expert weights through the model-owned memory cache.
        let (paged_weights, page_manifest, expert_weight_memory_cache_request_report) = {
            let mut expert_weight_memory_cache = self.expert_weight_memory_cache.borrow_mut();
            expert_pager.load_selected_experts_through_memory_cache_with_performance_attribution(
                &self.runtime,
                layer_index,
                &sorted_unique_expert_ids,
                &mut expert_weight_memory_cache,
                performance_attribution,
            )?
        };
        performance_attribution.measure_operation(
            PerformanceOperation::ExpertPagingDiagnosticLogging,
            |_performance_attribution| {
                tracing::debug!(
                    layer_index,
                    expert_weight_memory_cache_hits =
                        expert_weight_memory_cache_request_report.cache_hit_count,
                    expert_weight_memory_cache_misses =
                        expert_weight_memory_cache_request_report.cache_miss_count,
                    expert_weight_disk_page_loads =
                        expert_weight_memory_cache_request_report.disk_page_load_count,
                    expert_weight_disk_batch_loads =
                        expert_weight_memory_cache_request_report.disk_batch_load_count,
                    "expert memory cache request completed"
                );
            },
        );

        // Forward through paged MoE with pre-computed routing results.
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &paged_weights,
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
        mixture_of_experts_weights: &Qwen3_5MoEMixtureOfExpertsWeights,
        expert_pager: &ExpertPager,
        layer_index: usize,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        should_allow_full_layer_page: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let sorted_unique_expert_ids =
            self.copy_sorted_unique_expert_ids(selected_indices, performance_attribution)?;
        let sparse_expert_count = usize::try_from(self.config.expert_count()).map_err(|_| {
            Qwen3_5MoEExecutionError::InvalidInput {
                description: "paged prefill expert count exceeds the host integer range",
            }
        })?;
        let has_high_selected_expert_density = sorted_unique_expert_ids
            .len()
            .saturating_mul(FULL_LAYER_PAGE_MINIMUM_SELECTED_EXPERT_DENOMINATOR)
            >= sparse_expert_count
                .saturating_mul(FULL_LAYER_PAGE_MINIMUM_SELECTED_EXPERT_NUMERATOR);
        let should_load_full_layer_page_for_retention = if should_allow_full_layer_page {
            let (complete_layer_expert_payload_byte_count, complete_layer_memory_budget_snapshot) =
                expert_pager.complete_layer_retention_memory_budget_snapshot(
                    &self.runtime,
                    layer_index,
                    performance_attribution,
                )?;
            performance_attribution.measure_operation(
                PerformanceOperation::ExpertWeightMemoryCacheEviction,
                |_performance_attribution| {
                    let mut expert_weight_memory_cache =
                        self.expert_weight_memory_cache.borrow_mut();
                    expert_weight_memory_cache
                        .update_from_memory_budget_snapshot_while_protecting_selected_experts(
                            &complete_layer_memory_budget_snapshot,
                            layer_index,
                            &[],
                            complete_layer_expert_payload_byte_count,
                        );
                    complete_layer_memory_budget_snapshot.within_cap()
                        && expert_weight_memory_cache
                            .can_physically_retain_complete_layer_expert_payload(
                                layer_index,
                                complete_layer_expert_payload_byte_count,
                            )
                },
            )
        } else {
            false
        };
        let should_load_full_layer_page = should_allow_full_layer_page
            && (has_high_selected_expert_density || should_load_full_layer_page_for_retention);
        let all_sparse_expert_ids_for_layer =
            should_load_full_layer_page.then(|| (0..sparse_expert_count).collect::<Vec<_>>());
        let (paged_expert_weights, page_manifest, memory_budget_snapshot, did_load_full_layer_page) =
            match all_sparse_expert_ids_for_layer.as_ref() {
                Some(all_sparse_expert_ids_for_layer) => {
                    match expert_pager.load_selected_experts_with_performance_attribution(
                        &self.runtime,
                        layer_index,
                        all_sparse_expert_ids_for_layer,
                        None,
                        performance_attribution,
                    ) {
                        Ok((paged_expert_weights, page_manifest, memory_budget_snapshot)) => (
                            paged_expert_weights,
                            page_manifest,
                            memory_budget_snapshot,
                            true,
                        ),
                        Err(ExpertPagingError::MemoryBudget(memory_budget_error)) => {
                            performance_attribution.measure_operation(
                                PerformanceOperation::ExpertPagingDiagnosticLogging,
                                |_performance_attribution| {
                                    tracing::debug!(
                                        layer_index,
                                        selected_expert_count = sorted_unique_expert_ids.len(),
                                        error = %memory_budget_error,
                                        "full-layer prefill page exceeded the live memory budget; loading a compact page"
                                    );
                                },
                            );
                            let (paged_expert_weights, page_manifest, memory_budget_snapshot) = {
                                let mut expert_weight_memory_cache =
                                    self.expert_weight_memory_cache.borrow_mut();
                                expert_pager.load_selected_experts_with_performance_attribution(
                                    &self.runtime,
                                    layer_index,
                                    &sorted_unique_expert_ids,
                                    Some(&mut expert_weight_memory_cache),
                                    performance_attribution,
                                )?
                            };
                            (
                                paged_expert_weights,
                                page_manifest,
                                memory_budget_snapshot,
                                false,
                            )
                        }
                        Err(expert_paging_error) => return Err(expert_paging_error.into()),
                    }
                }
                None => {
                    let (paged_expert_weights, page_manifest, memory_budget_snapshot) = {
                        let mut expert_weight_memory_cache =
                            self.expert_weight_memory_cache.borrow_mut();
                        expert_pager.load_selected_experts_with_performance_attribution(
                            &self.runtime,
                            layer_index,
                            &sorted_unique_expert_ids,
                            Some(&mut expert_weight_memory_cache),
                            performance_attribution,
                        )?
                    };
                    (
                        paged_expert_weights,
                        page_manifest,
                        memory_budget_snapshot,
                        false,
                    )
                }
            };
        performance_attribution.measure_operation(
            PerformanceOperation::ExpertPagingDiagnosticLogging,
            |_performance_attribution| {
                tracing::debug!(
                    layer_index,
                    selected_expert_count = sorted_unique_expert_ids.len(),
                    sparse_expert_count,
                    did_load_full_layer_page,
                    "loaded direct paged prefill experts"
                );
            },
        );
        if did_load_full_layer_page {
            let mixture_of_experts_output = self
                .forward_moe_with_complete_layer_paging_and_performance_attribution(
                    hidden_states,
                    mixture_of_experts_weights,
                    &paged_expert_weights,
                    selected_indices,
                    selected_scores,
                    should_use_compiled_elementwise_graphs,
                    performance_attribution,
                )?;
            let should_retain_complete_layer = performance_attribution.measure_operation(
                PerformanceOperation::ExpertWeightMemoryCacheEviction,
                |_performance_attribution| {
                    let mut expert_weight_memory_cache =
                        self.expert_weight_memory_cache.borrow_mut();
                    expert_weight_memory_cache
                        .update_from_memory_budget_snapshot_while_protecting_selected_experts(
                            &memory_budget_snapshot,
                            layer_index,
                            &[],
                            page_manifest.payload_byte_count,
                        );
                    expert_weight_memory_cache.can_physically_retain_complete_layer_expert_payload(
                        layer_index,
                        page_manifest.payload_byte_count,
                    )
                },
            );
            if should_retain_complete_layer {
                self.expert_weight_memory_cache
                    .borrow_mut()
                    .remember_complete_layer_expert_weights(layer_index, paged_expert_weights);
            }
            return Ok(mixture_of_experts_output);
        }

        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &paged_expert_weights,
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
        mixture_of_experts_weights: &Qwen3_5MoEMixtureOfExpertsWeights,
        expert_pager: &ExpertPager,
        layer_index: usize,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let hidden_state_shape = hidden_states.shape();
        if hidden_state_shape.len() != 3 || hidden_state_shape[0] != 1 {
            return Err(Qwen3_5MoEExecutionError::InvalidInput {
                description: "paged prefill MoE currently expects [1, tokens, hidden] states",
            });
        }
        let token_count = hidden_state_shape[1];
        let hidden_dimension = hidden_state_shape[2];
        let expert_count_per_token = selected_indices.shape().last().copied().ok_or(
            Qwen3_5MoEExecutionError::InvalidInput {
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
            let token_moe_output = self.forward_moe_with_cached_paging(
                &token_hidden_states,
                mixture_of_experts_weights,
                expert_pager,
                layer_index,
                token_count,
                &token_selected_indices,
                &token_selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            )?;
            token_moe_outputs.push(token_moe_output);
        }

        let token_moe_output_references = token_moe_outputs.iter().collect::<Vec<_>>();
        Ok(self
            .runtime
            .concatenate_axis(&token_moe_output_references, 1)?)
    }
}

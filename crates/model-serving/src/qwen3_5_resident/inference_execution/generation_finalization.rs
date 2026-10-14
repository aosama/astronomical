//! One terminal cleanup path for success, rejection, cancellation, and failure.
//!
//! Request-owned lazy arrays are dropped before expert-retention policy resumes
//! and before allocator cleanup. After that workspace is gone, complete residency
//! is restored when the leftover ceiling admits it so a fitting model is not left
//! streaming. Final telemetry then reports that restored ownership.

use crate::decoder_cache::DecoderCacheState;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::{
    GenerationFinalization, GenerationPerformanceAttributionMetadata, InferenceEngineError,
    MlxMemoryTelemetry, PerformanceAttribution, PerformanceAttributionOutcome,
    PerformanceOperation, PersistentPromptCacheBlockCausalInput, PersistentPromptCacheBlockKey,
    RetainedPrefixRetentionDecision, RetainedPrefixRetentionRequirements,
};

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::retained_live_prefix::RetainedLiveDecoderPrefix;

impl Qwen3_5EngineState {
    pub(super) fn collect_current_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        let Some(model) = self.model.as_ref() else {
            return Ok(None);
        };
        let mlx_memory_snapshot = model
            .runtime()
            .memory_snapshot()
            .map_err(super::qwen3_5_runtime_error)?;
        let active_memory_bytes = u64::try_from(mlx_memory_snapshot.active_memory_bytes())
            .map_err(|_| {
                super::fatal_engine_error("MLX active memory bytes exceed the u64 range")
            })?;
        let allocator_cache_memory_bytes =
            u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes()).map_err(|_| {
                super::fatal_engine_error("MLX allocator-cache memory bytes exceed the u64 range")
            })?;
        let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes())
            .map_err(|_| super::fatal_engine_error("MLX peak memory bytes exceed the u64 range"))?;
        let active_memory_breakdown = model.finalized_active_memory_breakdown(
            active_memory_bytes,
            self.live_retained_context_state_payload_bytes(),
            0,
        );
        Ok(Some(
            MlxMemoryTelemetry::new(
                active_memory_bytes,
                allocator_cache_memory_bytes,
                peak_memory_bytes,
                active_memory_breakdown,
            )
            // The claim is derived from the breakdown reconciling this exact
            // snapshot, so the published pair cannot straddle a promotion or
            // demote (issue #337).
            .with_expert_residency_telemetry(
                model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown),
            )
            // Issue #510: the idle sample answers "is the remaining headroom
            // free or promised" by planning the next request's decode budget
            // against a fresh zero-token context.
            .with_memory_ceiling_utilization(
                model.memory_ceiling_utilization_for_breakdown(
                    crate::MemoryPhase::Decode,
                    0,
                    1,
                    active_memory_bytes,
                    active_memory_breakdown,
                ),
            ),
        ))
    }

    pub(super) fn finalize_generation_request_after_error(
        &mut self,
        active_request: Qwen3_5EngineRequest,
        generation_error: &InferenceEngineError,
        rejection_description: &'static str,
        failure_description: &'static str,
    ) {
        let (performance_attribution_outcome, performance_attribution_failure_description) =
            match generation_error {
                InferenceEngineError::InvalidRequest { .. }
                | InferenceEngineError::ResidentForkRequired { .. }
                | InferenceEngineError::ResidentStreamingReplacementRequired { .. }
                | InferenceEngineError::MlxMemoryLimitRejected { .. }
                | InferenceEngineError::EngineBusy => (
                    PerformanceAttributionOutcome::Rejected,
                    rejection_description,
                ),
                InferenceEngineError::Fatal { .. } => {
                    (PerformanceAttributionOutcome::Failed, failure_description)
                }
            };
        let _generation_finalization = self.finalize_generation_request(
            active_request,
            performance_attribution_outcome,
            Some(performance_attribution_failure_description),
        );
    }

    pub(super) fn finalize_generation_request(
        &mut self,
        mut active_request: Qwen3_5EngineRequest,
        outcome: PerformanceAttributionOutcome,
        failure_description: Option<&'static str>,
    ) -> GenerationFinalization {
        let request_id = active_request.request_id;
        let configured_maximum_output_tokens = active_request.maximum_output_tokens;
        // Retain the completed generation's live state for in-memory prefix
        // adoption (issue #1167) before the request drops. The memory package
        // owns the decision; the engine measures facts and enacts it. The
        // retained KV stays live, so the finalized memory snapshot honestly
        // reports it as active.
        self.retained_live_prefix = self.retain_completed_live_prefix(&mut active_request);
        // Attribution must outlive the request because cleanup and the final
        // memory snapshot are part of the user-visible request cost.
        let mut performance_attribution = std::mem::replace(
            &mut active_request.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        // Drop pending tokens and decoder state before sampling post-request memory.
        drop(active_request);
        // `release_request_memory` synchronizes the model stream before clearing
        // allocator storage, so one-token-ahead decode cannot race reclamation.
        let mlx_memory_snapshot = performance_attribution.measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| {
                self.release_request_memory(request_id, self.adaptive_ram_growth_guard_enabled)
            },
        );
        let generation_finalization =
            self.collect_generation_finalization(&mut performance_attribution);
        self.record_generation_performance_attribution(
            performance_attribution,
            outcome,
            request_id,
            configured_maximum_output_tokens,
            mlx_memory_snapshot,
            failure_description,
        );
        generation_finalization
    }

    /// Extracts the completed generation's live decoder state for adoption by
    /// the next request, or `None` when the retention policy drops it.
    ///
    /// The adoptable boundary is the smaller of the prompt's complete
    /// published-block chain and the live state's verified KV offset — a
    /// failed or cancelled generation must not retain uninitialized slab
    /// regions behind its processed offset.
    fn retain_completed_live_prefix(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
    ) -> Option<RetainedLiveDecoderPrefix> {
        let model_contract = self.persistent_prompt_cache_model_contract.as_ref()?;
        if !active_request.can_use_persistent_prompt_cache {
            return None;
        }
        let block_token_count = model_contract.block_token_count();
        let verified_state_offset_tokens = (0..active_request.request_decoder_state.layer_count())
            .filter_map(|layer_index| {
                match active_request.request_decoder_state.layer(layer_index) {
                    Some(DecoderCacheState::AppendOnlyAttention { attention }) => {
                        Some(attention.offset_tokens())
                    }
                    _ => None,
                }
            })
            .min()?;
        let verified_state_offset_count =
            usize::try_from(verified_state_offset_tokens.max(0)).unwrap_or(0);
        let prompt_complete_block_count = active_request.input_token_ids.len() / block_token_count;
        let adoptable_block_count = verified_state_offset_count / block_token_count;
        let adoptable_block_count = adoptable_block_count.min(prompt_complete_block_count);
        let retention_decision = RetainedPrefixRetentionRequirements {
            adoption_enabled: self.chunking.experimental_in_memory_prefix_adoption_enabled,
            adoptable_token_count: adoptable_block_count * block_token_count,
            retained_state_payload_bytes: active_request.request_decoder_state.payload_byte_count(),
            effective_mlx_memory_ceiling_bytes: u64::try_from(
                self.memory_limits.active_memory_limit_bytes(),
            )
            .unwrap_or(u64::MAX),
        }
        .decide();
        if retention_decision != RetainedPrefixRetentionDecision::Retain {
            return None;
        }
        let mut complete_block_keys = Vec::with_capacity(adoptable_block_count);
        let mut parent_block_key: Option<PersistentPromptCacheBlockKey> = None;
        let empty_block_causal_input = PersistentPromptCacheBlockCausalInput::empty();
        for block_index in 0..adoptable_block_count {
            let block_start = block_index * block_token_count;
            let block_end = block_start + block_token_count;
            let block_causal_input = active_request
                .persistent_prompt_cache_block_causal_inputs
                .get(block_index)
                .unwrap_or(&empty_block_causal_input);
            let block_key = match parent_block_key.as_ref() {
                Some(parent_block_key) => parent_block_key.for_child_block_with_causal_input(
                    &active_request.input_token_ids[block_start..block_end],
                    block_causal_input,
                ),
                None => PersistentPromptCacheBlockKey::for_root_block_with_causal_input(
                    model_contract,
                    &active_request.input_token_ids[block_start..block_end],
                    block_causal_input,
                ),
            }
            .ok()?;
            parent_block_key = Some(block_key.clone());
            complete_block_keys.push(block_key);
        }
        let placeholder_decoder_state =
            RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                self.model.as_ref()?.decoder_cache_layout(),
                self.full_attention_kv_state_growth_tokens,
            )
            .ok()?;
        let decoder_state = std::mem::replace(
            &mut active_request.request_decoder_state,
            placeholder_decoder_state,
        );
        Some(RetainedLiveDecoderPrefix {
            decoder_state,
            complete_block_keys,
            adoptable_token_count: adoptable_block_count * block_token_count,
        })
    }

    fn collect_generation_finalization(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> GenerationFinalization {
        let Some(model) = self.model.as_ref() else {
            return GenerationFinalization::default();
        };
        let expert_memory_mode = Some(model.expert_memory_mode());
        let mlx_memory_telemetry = match performance_attribution.measure_operation(
            PerformanceOperation::FinalizedMlxMemorySnapshot,
            |_performance_attribution| model.runtime().memory_snapshot(),
        ) {
            Ok(mlx_memory_snapshot) => match (
                u64::try_from(mlx_memory_snapshot.active_memory_bytes()),
                u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes()),
                u64::try_from(mlx_memory_snapshot.peak_memory_bytes()),
            ) {
                (
                    Ok(active_memory_bytes),
                    Ok(allocator_cache_memory_bytes),
                    Ok(peak_memory_bytes),
                ) => {
                    // The finalized claim derives from the breakdown reconciling
                    // this exact snapshot, so the pair cannot straddle a
                    // promotion or demote (issue #337).
                    let active_memory_breakdown = model.finalized_active_memory_breakdown(
                        active_memory_bytes,
                        self.live_retained_context_state_payload_bytes(),
                        0,
                    );
                    let expert_residency_telemetry = Some(
                        model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown),
                    );
                    (
                        Some(
                            MlxMemoryTelemetry::new(
                                active_memory_bytes,
                                allocator_cache_memory_bytes,
                                peak_memory_bytes,
                                active_memory_breakdown,
                            )
                            .with_memory_ceiling_utilization(
                                model.memory_ceiling_utilization_for_breakdown(
                                    crate::MemoryPhase::Decode,
                                    0,
                                    1,
                                    active_memory_bytes,
                                    active_memory_breakdown,
                                ),
                            ),
                        ),
                        expert_residency_telemetry,
                    )
                }
                memory_counter_results => {
                    tracing::warn!(
                        ?memory_counter_results,
                        "could not publish finalized MLX memory because a runtime counter exceeds the u64 range"
                    );
                    (None, None)
                }
            },
            Err(memory_snapshot_error) => {
                tracing::warn!(
                    error = %memory_snapshot_error,
                    "could not capture finalized MLX memory after expert residency recovery"
                );
                (None, None)
            }
        };
        let (mlx_memory_telemetry, expert_residency_telemetry) = mlx_memory_telemetry;
        GenerationFinalization::new(
            expert_memory_mode,
            mlx_memory_telemetry,
            expert_residency_telemetry,
        )
    }

    pub(super) fn record_generation_performance_attribution(
        &mut self,
        performance_attribution: PerformanceAttribution,
        outcome: PerformanceAttributionOutcome,
        request_id: astronomical_ipc_protocol::RequestId,
        configured_maximum_output_tokens: u16,
        mlx_memory_snapshot: Option<astronomical_runtime_integration::MlxMemorySnapshot>,
        failure_description: Option<&'static str>,
    ) {
        if !performance_attribution.is_enabled() {
            return;
        }
        let (Some(model_id), Some(model_revision)) =
            (self.model_id.clone(), self.model_revision.clone())
        else {
            tracing::error!(
                request_id = request_id.value(),
                "completed generation lacked loaded-model attribution metadata"
            );
            return;
        };
        let Some(performance_attribution_report) =
            performance_attribution.finish_generation(GenerationPerformanceAttributionMetadata {
                outcome,
                model_id,
                model_revision,
                prefill_transient_observation_completed: self
                    .adaptive_ram_growth_guard
                    .has_completed_growth_observation(crate::MemoryPhase::Prefill),
                prefill_observed_transient_high_water_bytes: u64::try_from(
                    self.adaptive_ram_growth_guard
                        .observed_transient_high_water_bytes(crate::MemoryPhase::Prefill),
                )
                .unwrap_or(u64::MAX),
                request_id: request_id.value(),
                configured_maximum_output_tokens,
                mlx_active_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_allocator_cache_memory_bytes: mlx_memory_snapshot.as_ref().and_then(
                    |snapshot| u64::try_from(snapshot.allocator_cache_memory_bytes()).ok(),
                ),
                mlx_peak_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
                failure_description: failure_description.map(str::to_owned),
            })
        else {
            return;
        };
        if let Err(performance_attribution_write_error) = self
            .performance_attribution_log
            .record(&performance_attribution_report)
        {
            tracing::warn!(
                error = %performance_attribution_write_error,
                "failed to append generation performance attribution"
            );
        }
    }
}

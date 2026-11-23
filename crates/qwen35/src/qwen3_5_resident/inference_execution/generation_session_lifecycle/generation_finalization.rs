//! One terminal cleanup path for success, rejection, cancellation, and failure.
//!
//! The live prefix is kept for the next reply before the request's arrays are
//! dropped; allocator cleanup follows, then the memory governor ends the
//! request, and the final telemetry reports what remains. The telemetry, the
//! attribution record and the after-error path are shared with the streaming
//! engine in `qwen3_5_core::shared_execution::generation_finalization_steps`.

use crate::astro_decoder_state_cache::AstroDecoderStateCacheState;
use crate::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;
use crate::qwen3_5_core::shared_execution::generation_finalization_steps;
use crate::{
    AstroSsdPromptCacheBlockKey, GenerationFinalization, PerformanceAttribution,
    PerformanceAttributionOutcome, PerformanceOperation, RetainedPrefixRetentionDecision,
    RetainedPrefixRetentionRequirements,
};

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::retained_live_prefix::RetainedLiveDecoderPrefix;
use crate::qwen3_5_core::shared_execution::request_memory_release::release_request_memory;

impl generation_finalization_steps::Qwen3_5FinalizationHooks for Qwen3_5EngineState {
    fn retained_live_context_state_payload_bytes(&self) -> u64 {
        Qwen3_5EngineState::live_retained_context_state_payload_bytes(self)
    }

    fn measured_prompt_mlx_working_buffer_bytes(&self) -> Option<u64> {
        self.measured_prompt_chunk_mlx_working_buffer_bytes()
    }

    fn finalize_generation_request(
        &mut self,
        active_request: Qwen3_5EngineRequest,
        performance_attribution_outcome: PerformanceAttributionOutcome,
        failure_description: Option<&'static str>,
    ) -> GenerationFinalization {
        Qwen3_5EngineState::finalize_generation_request(
            self,
            active_request,
            performance_attribution_outcome,
            failure_description,
        )
    }
}

impl Qwen3_5EngineState {
    pub(in crate::qwen3_5_resident::inference_execution) fn finalize_generation_request(
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
        let mut drafting_statistics = active_request.engine_extension.drafting_statistics;
        // A reply that ends partway through a kept chain leaves its drafts in
        // the emission queue; every entry but the tail (the round's
        // correction token, not a draft) was counted as kept but never sent.
        let unsent_accepted_token_count = active_request
            .engine_extension
            .speculative_emission_queue
            .len()
            .saturating_sub(1);
        drafting_statistics.withdraw_unsent_accepted_tokens(
            u32::try_from(unsent_accepted_token_count).unwrap_or(u32::MAX),
        );
        // Drop pending tokens and decoder state before sampling post-request memory.
        drop(active_request);
        // `release_request_memory` synchronizes the model stream before clearing
        // allocator storage, so one-token-ahead decode cannot race reclamation.
        let mlx_memory_snapshot = performance_attribution.measure_operation(
            PerformanceOperation::MlxBufferCacheCleanup,
            |_performance_attribution| {
                release_request_memory(self.core.model.as_deref(), request_id)
            },
        );
        self.govern_request_end();
        let generation_finalization = self
            .collect_generation_finalization(
                &mut performance_attribution,
                mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
            )
            .with_drafting_statistics(drafting_statistics);
        generation_finalization_steps::record_generation_performance_attribution(
            self,
            generation_finalization_steps::GenerationAttributionRecord {
                performance_attribution,
                outcome,
                request_id,
                configured_maximum_output_tokens,
                mlx_memory_snapshot,
                failure_description,
            },
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
        let model_contract = self.core.astro_ssd_prompt_cache_model_contract.as_ref()?;
        if !active_request.can_use_astro_ssd_prompt_cache {
            return None;
        }
        let block_token_count = model_contract.block_token_count();
        let verified_state_offset_tokens = (0..active_request.request_decoder_state.layer_count())
            .filter_map(|layer_index| {
                match active_request.request_decoder_state.layer(layer_index) {
                    Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }) => {
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
            adoption_enabled: self
                .core
                .chunking
                .experimental_in_memory_prefix_adoption_enabled,
            adoptable_token_count: adoptable_block_count * block_token_count,
            retained_state_payload_bytes: active_request.request_decoder_state.payload_byte_count(),
            effective_mlx_memory_ceiling_bytes: u64::try_from(
                self.core.memory_limits.active_memory_limit_bytes(),
            )
            .unwrap_or(u64::MAX),
        }
        .decide();
        if retention_decision != RetainedPrefixRetentionDecision::Retain {
            return None;
        }
        let mut complete_block_keys = Vec::with_capacity(adoptable_block_count);
        let mut parent_block_key: Option<AstroSsdPromptCacheBlockKey> = None;
        let empty_block_causal_input = AstroSsdPromptCacheBlockCausalInput::empty();
        for block_index in 0..adoptable_block_count {
            let block_start = block_index * block_token_count;
            let block_end = block_start + block_token_count;
            let block_causal_input = active_request
                .astro_ssd_prompt_cache_block_causal_inputs
                .get(block_index)
                .unwrap_or(&empty_block_causal_input);
            let block_key = match parent_block_key.as_ref() {
                Some(parent_block_key) => parent_block_key.for_child_block_with_causal_input(
                    &active_request.input_token_ids[block_start..block_end],
                    block_causal_input,
                ),
                None => AstroSsdPromptCacheBlockKey::for_root_block_with_causal_input(
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
            RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
                self.core.model.as_ref()?.astro_decoder_state_cache_layout(),
                self.core.full_attention_kv_state_growth_tokens,
            )
            .ok()?;
        let decoder_state = std::mem::replace(
            &mut active_request.request_decoder_state,
            placeholder_decoder_state,
        );
        Some(RetainedLiveDecoderPrefix {
            decoder_state,
            complete_block_keys,
        })
    }

    fn collect_generation_finalization(
        &self,
        performance_attribution: &mut PerformanceAttribution,
        release_peak_memory_bytes: Option<u64>,
    ) -> GenerationFinalization {
        let Some(model) = self.core.model.as_ref() else {
            return GenerationFinalization::default();
        };
        let expert_memory_mode = Some(model.expert_memory_mode());
        let mlx_memory_telemetry = match performance_attribution.measure_operation(
            PerformanceOperation::FinalizedMlxMemorySnapshot,
            |_performance_attribution| model.runtime().memory_snapshot(),
        ) {
            Ok(mlx_memory_snapshot) => match (
                u64::try_from(mlx_memory_snapshot.active_memory_bytes()),
                u64::try_from(mlx_memory_snapshot.mlx_buffer_cache_bytes()),
                u64::try_from(mlx_memory_snapshot.peak_memory_bytes()),
            ) {
                (
                    Ok(active_memory_bytes),
                    Ok(mlx_buffer_cache_bytes),
                    Ok(sampled_peak_memory_bytes),
                ) => {
                    // The MLX peak counter resets at MLX buffer-cache cleanups,
                    // so this post-cleanup sample can read a lower window peak
                    // than the synchronized release sample taken moments
                    // earlier. The finalized claim is a request maximum, so it
                    // must retain the higher of the two windows.
                    let peak_memory_bytes = match release_peak_memory_bytes {
                        Some(release_peak_memory_bytes) => {
                            release_peak_memory_bytes.max(sampled_peak_memory_bytes)
                        }
                        None => sampled_peak_memory_bytes,
                    };
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
                                mlx_buffer_cache_bytes,
                                peak_memory_bytes,
                                active_memory_breakdown,
                            )
                            .with_memory_ceiling_utilization(
                                model.between_request_memory_ceiling_utilization_for_breakdown(
                                    active_memory_bytes,
                                    active_memory_breakdown,
                                ),
                            )
                            .with_memory_ledger_reading(model.latest_memory_ledger_reading()),
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
}

use crate::MlxMemoryTelemetry;

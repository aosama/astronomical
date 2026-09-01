//! Owner-thread Laguna execution used by `MlxInferenceEngine`.

use std::sync::Arc;

use astronomical_ipc_protocol::{RequestId, WorkerEvent, WorkerPromptWorkReuse};
use astronomical_runtime_integration::MlxRuntime;

use crate::artifact_validation::ValidatedWeightsFile;
use crate::laguna::{
    LagunaDecoderState, LagunaExpertPagingPlan, LagunaInferenceRequest, LagunaModel,
    LagunaPromptProcessingChunkSizer, LagunaTargetContract, LagunaTensorContract,
};
use crate::{
    AdaptiveRamGrowthGuard, EngineGenerationStart, EngineLoadResult, GeneratedToken,
    GenerationFinalization, InferenceEngineError, MemoryPhase, MlxInferenceExecution,
    MlxMemoryLimitAdjustment, MlxRamBudget, PerformanceAttribution, PerformanceAttributionLog,
    PerformanceCounter, PersistentPromptCacheCounters, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, build_persistent_prompt_cache_stats_event,
};

use super::active_generation::LagunaActiveGeneration;
use super::memory::complete_laguna_forward_memory_observation;
use super::prefill_capacity_recovery::LagunaPrefillFailureInjection;

/// Deferred Laguna construction that must run on the MLX owner thread.
pub(in crate::laguna) struct LagunaPendingStartup {
    pub(in crate::laguna) target_contract: LagunaTargetContract,
    pub(in crate::laguna) tensor_contract: LagunaTensorContract,
    pub(in crate::laguna) shard_files: std::collections::BTreeMap<String, ValidatedWeightsFile>,
    pub(in crate::laguna) paging_plan: LagunaExpertPagingPlan,
    pub(in crate::laguna) load_routed_experts: bool,
    pub(in crate::laguna) mlx_ram_budget: MlxRamBudget,
    pub(in crate::laguna) effective_mlx_memory_ceiling_bytes: usize,
    pub(in crate::laguna) allocator_cache_memory_limit_bytes: usize,
    pub(in crate::laguna) prompt_processing_chunk_sizer: LagunaPromptProcessingChunkSizer,
    pub(in crate::laguna) minimum_mlx_memory_ceiling_bytes: u64,
    pub(in crate::laguna) prompt_cache_disk_store_config:
        Option<PersistentPromptCacheDiskStoreConfig>,
    pub(in crate::laguna) prompt_cache_model_id: String,
    pub(in crate::laguna) prompt_cache_model_revision: String,
    pub(in crate::laguna) configured_prompt_cache_block_token_count: Option<usize>,
    pub(in crate::laguna) prompt_cache_common_prefix_stride_blocks: u32,
    pub(in crate::laguna) prefill_graph_submission_layer_interval: u32,
    pub(in crate::laguna) experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
    pub(in crate::laguna) experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
    pub(in crate::laguna) model_loading_performance_attribution: PerformanceAttribution,
    pub(in crate::laguna) performance_attribution_log: PerformanceAttributionLog,
    pub(in crate::laguna) attribution_model_id: String,
    pub(in crate::laguna) attribution_model_revision: String,
    pub(in crate::laguna) total_artifact_payload_bytes: u64,
    pub(in crate::laguna) model_shard_count: usize,
}

/// One loaded Laguna model plus at most one active generation.
pub struct LagunaInferenceExecution {
    /// Deferred startup payload consumed once by the MLX owner thread.
    pub(super) pending_startup: Option<LagunaPendingStartup>,
    /// MLX owner-thread runtime used by both prompt and decode advances.
    pub(super) runtime: Option<MlxRuntime>,
    /// Loaded Laguna weights, decoder layers, and expert-residency owner.
    pub(super) model: Option<LagunaModel>,
    /// Request state returned to the protocol loop after every bounded advance.
    pub(super) active_request: Option<LagunaActiveGeneration>,
    /// Deterministic fixed-size planner that bounds one prompt advance.
    pub(super) prompt_processing_chunk_sizer: Option<LagunaPromptProcessingChunkSizer>,
    /// Optional model-and-revision SSD cache shared by sequential requests.
    pub(super) persistent_prompt_cache: Option<Arc<PersistentPromptCacheDiskStore>>,
    /// Resolved global quota retained for process-scoped cache statistics.
    pub(super) persistent_prompt_cache_disk_store_config:
        Option<PersistentPromptCacheDiskStoreConfig>,
    /// Process-lifetime hit, miss, and restored-token totals.
    pub(super) persistent_prompt_cache_counters: PersistentPromptCacheCounters,
    /// Machine-relative memory policy recomposed for prefill and decode.
    pub(super) mlx_ram_budget: Option<MlxRamBudget>,
    /// Exact-context forward admission and learned transient high-water evidence.
    pub(super) adaptive_ram_growth_guard: Option<AdaptiveRamGrowthGuard>,
    /// Smallest ceiling that preserves model core plus mandatory page/transient work.
    pub(super) minimum_mlx_memory_ceiling_bytes: u64,
    /// Startup-resolved allocator-cache cap restored when a lowered active ceiling rises again.
    pub(super) maximum_allocator_cache_memory_limit_bytes: usize,
    /// Family-owned writer for model-load and generation timing reports.
    pub(super) performance_attribution_log: PerformanceAttributionLog,
    /// Loaded identity copied into every completed generation report.
    pub(super) attribution_model_id: Option<String>,
    pub(super) attribution_model_revision: Option<String>,
    /// Direct-MLX acceptance seam that is a zero-sized no-op in production builds.
    pub(super) prefill_failure_injection: LagunaPrefillFailureInjection,
}

impl LagunaInferenceExecution {
    pub(in crate::laguna) fn pending(pending_startup: LagunaPendingStartup) -> Self {
        let minimum_mlx_memory_ceiling_bytes = pending_startup.minimum_mlx_memory_ceiling_bytes;
        let maximum_allocator_cache_memory_limit_bytes =
            pending_startup.allocator_cache_memory_limit_bytes;
        Self {
            pending_startup: Some(pending_startup),
            runtime: None,
            model: None,
            active_request: None,
            prompt_processing_chunk_sizer: None,
            persistent_prompt_cache: None,
            persistent_prompt_cache_disk_store_config: None,
            persistent_prompt_cache_counters: PersistentPromptCacheCounters::default(),
            mlx_ram_budget: None,
            adaptive_ram_growth_guard: None,
            minimum_mlx_memory_ceiling_bytes,
            maximum_allocator_cache_memory_limit_bytes,
            performance_attribution_log: PerformanceAttributionLog::disabled(),
            attribution_model_id: None,
            attribution_model_revision: None,
            prefill_failure_injection: LagunaPrefillFailureInjection::default(),
        }
    }

    /// Forces two typed failures to prove one unchanged retry followed by bounded fallback.
    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub fn inject_two_prefill_capacity_failures_for_test(&mut self) {
        self.prefill_failure_injection.arm_two_failures();
    }
}

impl MlxInferenceExecution for LagunaInferenceExecution {
    type Request = LagunaInferenceRequest;

    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        self.load_model_on_owner_thread()
    }

    fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        if self.active_request.is_some() {
            return Err(InferenceEngineError::EngineBusy);
        }
        let request_id = inference_request.request_id();
        let sampling_strategy = inference_request.sampling_strategy().clone();
        let prompt_token_ids = inference_request.prompt_token_ids().to_vec();
        let remaining_output_tokens = inference_request.maximum_output_tokens();
        if prompt_token_ids.is_empty() {
            return Err(InferenceEngineError::InvalidRequest {
                reason: "a Laguna generation requires prompt tokens".to_owned(),
            });
        }
        let mut performance_attribution = inference_request.into_performance_attribution();
        // Prefill never owns a full-prompt rotating workspace. Admission must
        // reserve only the largest configured chunk or a later long prompt will
        // demote a fitting resident model.
        let maximum_forward_token_count = self
            .prompt_processing_chunk_sizer
            .as_ref()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna prompt-processing chunk sizer is not loaded".to_owned(),
            })?
            .maximum_prompt_processing_chunk_size_tokens();
        let persistent_prompt_cache = self.persistent_prompt_cache.clone();
        let prompt_cache_lookup = persistent_prompt_cache.as_deref().map(|store| {
            super::prompt_cache::lookup_prompt_prefix(
                store,
                &prompt_token_ids,
                &mut performance_attribution,
            )
        });
        let restored_prompt_prefix_token_count = prompt_cache_lookup
            .as_ref()
            .map_or(0, |lookup_result| lookup_result.restored_token_count());
        self.admit_generation_context(
            prompt_token_ids.len(),
            maximum_forward_token_count,
            // Boundary snapshot bytes exist only when this request may publish cache.
            persistent_prompt_cache.is_some(),
            restored_prompt_prefix_token_count,
            &mut performance_attribution,
        )?;
        let Some(runtime) = self.runtime.as_ref() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna runtime is not loaded".to_owned(),
            });
        };
        let Some(model) = self.model.as_mut() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            });
        };
        let random_state =
            super::token_sampling::random_state_for_strategy(runtime, &sampling_strategy)?;
        super::token_sampling::log_executed_sampling(request_id.value(), &sampling_strategy);
        let prompt_context_token_count = u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX);
        let mut decoder_state = LagunaDecoderState::empty(model.contract()).map_err(|_| {
            InferenceEngineError::Fatal {
                reason: "Laguna decoder state could not be allocated".to_owned(),
            }
        })?;
        let (last_published_block_key, restored_prompt_prefix_token_count) =
            if let (Some(persistent_prompt_cache), Some(prompt_cache_lookup)) = (
                persistent_prompt_cache.as_deref(),
                prompt_cache_lookup.as_ref(),
            ) {
                super::prompt_cache::restore_prompt_prefix(
                    runtime,
                    persistent_prompt_cache,
                    &prompt_token_ids,
                    prompt_cache_lookup,
                    &mut decoder_state,
                    &mut performance_attribution,
                )?
            } else {
                (None, 0)
            };
        if restored_prompt_prefix_token_count > 0 {
            super::memory_admission::admit_generation_context(
                runtime,
                model,
                Some(&decoder_state),
                prompt_token_ids
                    .len()
                    .saturating_sub(restored_prompt_prefix_token_count as usize),
                maximum_forward_token_count,
                false,
                0,
                &mut performance_attribution,
            )?;
        }
        if persistent_prompt_cache.is_some() {
            if restored_prompt_prefix_token_count == 0 {
                self.persistent_prompt_cache_counters.record_cache_miss();
            } else {
                self.persistent_prompt_cache_counters
                    .record_cache_hit(restored_prompt_prefix_token_count as usize);
            }
        }
        let prompt_token_count = u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX);
        let expert_memory_mode = model.expert_memory_mode();
        self.active_request = Some(LagunaActiveGeneration {
            request_id,
            decoder_state,
            next_input_token_ids: Vec::new(),
            remaining_output_tokens,
            configured_maximum_output_tokens: remaining_output_tokens,
            performance_attribution,
            context_token_count: prompt_context_token_count,
            prompt_token_ids,
            next_prompt_token_position: restored_prompt_prefix_token_count as usize,
            last_published_block_key,
            terminal_prompt_logits: None,
            prompt_work_reuse: WorkerPromptWorkReuse {
                target_eligible_token_count: prompt_token_count,
                target_restored_token_count: u64::from(restored_prompt_prefix_token_count),
                drafter_eligible_token_count: 0,
                drafter_restored_token_count: 0,
            },
            sampling_strategy,
            random_state,
        });
        Ok(EngineGenerationStart::with_expert_memory_mode(
            restored_prompt_prefix_token_count,
            expert_memory_mode,
        )
        .with_restored_prompt_prefix_token_count(restored_prompt_prefix_token_count))
    }

    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        if self
            .active_request
            .as_ref()
            .is_some_and(|active_request| active_request.remaining_output_tokens == 0)
        {
            let _generation_finalization = self.cancel_generation(request_id)?;
            return Ok(GeneratedToken::EndOfSequence);
        }
        if let Some(prefill_progress) = self.advance_pending_prompt_prefill(request_id)? {
            return Ok(prefill_progress);
        }
        let Some(runtime) = self.runtime.as_ref() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna runtime is not loaded".to_owned(),
            });
        };
        let Some(model) = self.model.as_mut() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            });
        };
        let mlx_ram_budget = self
            .mlx_ram_budget
            .as_mut()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna RAM budget is not loaded".to_owned(),
            })?;
        let adaptive_ram_growth_guard =
            self.adaptive_ram_growth_guard
                .as_mut()
                .ok_or(InferenceEngineError::Fatal {
                    reason: "the Laguna adaptive RAM growth guard is not loaded".to_owned(),
                })?;
        let active_request =
            self.active_request
                .as_mut()
                .ok_or(InferenceEngineError::InvalidRequest {
                    reason: "no Laguna generation is active".to_owned(),
                })?;
        if active_request.request_id != request_id {
            return Err(InferenceEngineError::InvalidRequest {
                reason: "Laguna generation request identifiers do not match".to_owned(),
            });
        }
        if active_request.next_input_token_ids.is_empty() {
            let prompt_logits = active_request.terminal_prompt_logits.take().ok_or(
                InferenceEngineError::Fatal {
                    reason: "Laguna prompt processing produced no logits".to_owned(),
                },
            )?;
            let first_generated_token_id = super::token_sampling::select_next_token_id(
                runtime,
                &prompt_logits,
                &active_request.sampling_strategy,
                &mut active_request.random_state,
                &mut active_request.performance_attribution,
            )?;
            active_request.next_input_token_ids = vec![first_generated_token_id];
        }
        let token_id = active_request.next_input_token_ids.last().copied().ok_or(
            InferenceEngineError::Fatal {
                reason: "Laguna decode is missing the previously sampled token".to_owned(),
            },
        )?;
        let decode_token_array =
            runtime
                .array_from_u32(&[token_id], &[1])
                .map_err(|_| InferenceEngineError::Fatal {
                    reason: "Laguna decode tokens could not be placed on the runtime".to_owned(),
                })?;
        let next_token_id = active_request.performance_attribution.measure_operation(
            crate::PerformanceOperation::DecodeAdvanceSpan,
            |performance_attribution| -> Result<u32, InferenceEngineError> {
                let (adaptive_ram_growth_context, memory_baseline) =
                    super::memory::admit_laguna_forward_memory(
                        runtime,
                        model,
                        adaptive_ram_growth_guard,
                        &active_request.decoder_state,
                        1,
                        0,
                        MemoryPhase::Decode,
                        active_request.context_token_count.saturating_add(1),
                        performance_attribution,
                    )?;
                let decode_logits = model
                    .forward_decode(
                        runtime,
                        &decode_token_array,
                        &mut active_request.decoder_state,
                        performance_attribution,
                    )
                    .map_err(|decode_error| {
                        tracing::error!(?decode_error, "Laguna decode forward failed");
                        InferenceEngineError::Fatal {
                            reason: "Laguna token generation failed".to_owned(),
                        }
                    })?;
                let next_token_id = super::token_sampling::select_next_token_id(
                    runtime,
                    &decode_logits,
                    &active_request.sampling_strategy,
                    &mut active_request.random_state,
                    performance_attribution,
                )?;
                complete_laguna_forward_memory_observation(
                    runtime,
                    model,
                    adaptive_ram_growth_guard,
                    adaptive_ram_growth_context,
                    mlx_ram_budget,
                    memory_baseline,
                    active_request.context_token_count.saturating_add(1),
                    performance_attribution,
                )?;
                Ok(next_token_id)
            },
        )?;
        active_request.next_input_token_ids = vec![next_token_id];
        active_request.remaining_output_tokens =
            active_request.remaining_output_tokens.saturating_sub(1);
        active_request.context_token_count = active_request.context_token_count.saturating_add(1);
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::GeneratedTokenCount, 1);
        let expert_memory_mode = model.expert_memory_mode();
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
        Ok(GeneratedToken::TokenId {
            token_id,
            is_reasoning_token: false,
            expert_memory_mode: Some(expert_memory_mode),
            mlx_memory_telemetry,
            first_decode_forward_elapsed_millis: None,
            generation_finalization: None,
        })
    }

    fn inject_input_tokens(
        &mut self,
        request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        super::injected_tokens::inject_input_tokens(self, request_id, input_token_ids)
    }

    fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        let Some(mut active_request) = self.active_request.take() else {
            return Err(InferenceEngineError::InvalidRequest {
                reason: "no Laguna generation is active".to_owned(),
            });
        };
        if active_request.request_id != request_id {
            self.active_request = Some(active_request);
            return Err(InferenceEngineError::InvalidRequest {
                reason: "Laguna generation request identifiers do not match".to_owned(),
            });
        }
        let configured_maximum_output_tokens = active_request.configured_maximum_output_tokens;
        let mut performance_attribution = std::mem::replace(
            &mut active_request.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        performance_attribution.measure_operation(
            crate::PerformanceOperation::GenerationFinalization,
            |_performance_attribution| {
                // Request arrays must be released before the stable model-memory snapshot.
                drop(active_request);
            },
        );
        if let Some(model) = self.model.as_ref() {
            model.resume_expert_retention_after_request_pressure();
        }
        let runtime = self.runtime.as_ref().ok_or(InferenceEngineError::Fatal {
            reason: "the Laguna runtime is not loaded".to_owned(),
        })?;
        performance_attribution
            .measure_operation(
                crate::PerformanceOperation::MlxAllocatorCacheCleanup,
                |_performance_attribution| {
                    runtime.synchronize_gpu_stream_and_clear_allocator_cache()
                },
            )
            .map_err(|cleanup_error| InferenceEngineError::Fatal {
                reason: format!("Laguna request-finalization cleanup failed: {cleanup_error}"),
            })?;
        self.record_generation_performance_attribution(
            performance_attribution,
            request_id,
            configured_maximum_output_tokens,
        );
        Ok(GenerationFinalization::new(
            self.model.as_ref().map(LagunaModel::expert_memory_mode),
            self.collect_current_mlx_memory_telemetry(),
            self.model
                .as_ref()
                .map(LagunaModel::expert_residency_telemetry),
        ))
    }

    fn collect_persistent_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let persistent_prompt_cache = match self.persistent_prompt_cache.as_ref() {
            Some(persistent_prompt_cache) => persistent_prompt_cache,
            None => return Ok(None),
        };
        let global_prompt_cache_maximum_size_bytes = self
            .persistent_prompt_cache_disk_store_config
            .as_ref()
            .map(PersistentPromptCacheDiskStoreConfig::global_prompt_cache_maximum_size_bytes)
            .unwrap_or(0);
        Ok(Some(build_persistent_prompt_cache_stats_event(
            &self.persistent_prompt_cache_counters,
            u64::try_from(
                persistent_prompt_cache
                    .model_contract_ref()
                    .block_token_count(),
            )
            .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.sequence_state_block_count()).unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.boundary_state_snapshot_count())
                .unwrap_or(u64::MAX),
            u64::try_from(persistent_prompt_cache.visual_embedding_count()).unwrap_or(u64::MAX),
            persistent_prompt_cache.total_size_bytes(),
            persistent_prompt_cache.visual_embedding_total_size_bytes(),
            global_prompt_cache_maximum_size_bytes,
        )))
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        self.apply_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
    }
}

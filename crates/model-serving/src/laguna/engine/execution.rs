//! Owner-thread Laguna execution used by `MlxInferenceEngine`.

use std::sync::Arc;

use astronomical_ipc_protocol::{RequestId, WorkerEvent, WorkerPromptWorkReuse};
use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::artifact_validation::ValidatedWeightsFile;
use crate::laguna::{
    LagunaDecoderState, LagunaExpertPagingPlan, LagunaInferenceRequest, LagunaModel,
    LagunaPromptProcessingChunkSizer, LagunaPromptProcessingExecutionProfile, LagunaTargetContract,
    LagunaTensorContract,
};
use crate::{
    EngineGenerationStart, EngineLoadResult, GeneratedToken, GenerationFinalization,
    InferenceEngineError, MlxInferenceExecution, MlxMemoryLimitAdjustment, MlxRamBudget,
    MlxRamBudgetPhase, PerformanceAttribution, PerformanceAttributionLog, PerformanceCounter,
    PersistentPromptCacheCounters, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, build_persistent_prompt_cache_stats_event,
};

use super::active_generation::LagunaActiveGeneration;
use super::memory::laguna_ram_budget_snapshot;

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
    pub(in crate::laguna) prompt_processing_execution_profile:
        LagunaPromptProcessingExecutionProfile,
    pub(in crate::laguna) minimum_mlx_memory_ceiling_bytes: u64,
    pub(in crate::laguna) prompt_cache_disk_store_config:
        Option<PersistentPromptCacheDiskStoreConfig>,
    pub(in crate::laguna) prompt_cache_model_id: String,
    pub(in crate::laguna) prompt_cache_model_revision: String,
    pub(in crate::laguna) configured_prompt_cache_block_token_count: Option<usize>,
    pub(in crate::laguna) prompt_cache_common_prefix_stride_blocks: u32,
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
    /// Adaptive or fixed selector that bounds one prompt advance.
    pub(super) prompt_processing_chunk_sizer: Option<LagunaPromptProcessingChunkSizer>,
    /// Descriptor-derived identity used to isolate optimizer measurements.
    pub(super) prompt_processing_execution_profile: Option<LagunaPromptProcessingExecutionProfile>,
    /// Optional model-and-revision SSD cache shared by sequential requests.
    pub(super) persistent_prompt_cache: Option<Arc<PersistentPromptCacheDiskStore>>,
    /// Resolved global quota retained for process-scoped cache statistics.
    pub(super) persistent_prompt_cache_disk_store_config:
        Option<PersistentPromptCacheDiskStoreConfig>,
    /// Process-lifetime hit, miss, and restored-token totals.
    pub(super) persistent_prompt_cache_counters: PersistentPromptCacheCounters,
    /// Machine-relative memory policy recomposed for prefill and decode.
    pub(super) mlx_ram_budget: Option<MlxRamBudget>,
    /// Smallest ceiling that preserves model core plus mandatory page/transient work.
    pub(super) minimum_mlx_memory_ceiling_bytes: u64,
    /// Family-owned writer for model-load and generation timing reports.
    pub(super) performance_attribution_log: PerformanceAttributionLog,
    /// Loaded identity copied into every completed generation report.
    pub(super) attribution_model_id: Option<String>,
    pub(super) attribution_model_revision: Option<String>,
}

impl LagunaInferenceExecution {
    pub(in crate::laguna) fn pending(pending_startup: LagunaPendingStartup) -> Self {
        let minimum_mlx_memory_ceiling_bytes = pending_startup.minimum_mlx_memory_ceiling_bytes;
        Self {
            pending_startup: Some(pending_startup),
            runtime: None,
            model: None,
            active_request: None,
            prompt_processing_chunk_sizer: None,
            prompt_processing_execution_profile: None,
            persistent_prompt_cache: None,
            persistent_prompt_cache_disk_store_config: None,
            persistent_prompt_cache_counters: PersistentPromptCacheCounters::default(),
            mlx_ram_budget: None,
            minimum_mlx_memory_ceiling_bytes,
            performance_attribution_log: PerformanceAttributionLog::disabled(),
            attribution_model_id: None,
            attribution_model_revision: None,
        }
    }

    fn sample_next_token_id(
        runtime: &MlxRuntime,
        logits: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<u32, InferenceEngineError> {
        LagunaModel::greedy_token_id(runtime, logits, performance_attribution).map_err(
            |sampling_error| InferenceEngineError::Fatal {
                reason: format!("Laguna greedy sampling failed: {sampling_error:?}"),
            },
        )
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
        let prompt_token_ids = inference_request.prompt_token_ids().to_vec();
        let remaining_output_tokens = inference_request.maximum_output_tokens();
        if prompt_token_ids.is_empty() {
            return Err(InferenceEngineError::InvalidRequest {
                reason: "a Laguna generation requires prompt tokens".to_owned(),
            });
        }
        let Some(runtime) = self.runtime.as_ref() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna runtime is not loaded".to_owned(),
            });
        };
        let Some(model) = self.model.as_ref() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            });
        };
        let prompt_context_token_count = u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX);
        if let Some(mlx_ram_budget) = self.mlx_ram_budget.as_ref() {
            let retained_expert_budget_bytes = laguna_ram_budget_snapshot(
                mlx_ram_budget,
                MlxRamBudgetPhase::Prefill,
                prompt_context_token_count,
            )
            .retained_expert_budget_bytes;
            model
                .set_retained_expert_ceiling(retained_expert_budget_bytes)
                .map_err(|_| InferenceEngineError::Fatal {
                    reason: "Laguna prefill expert budget could not be applied".to_owned(),
                })?;
        }
        let mut decoder_state = LagunaDecoderState::empty(model.contract()).map_err(|_| {
            InferenceEngineError::Fatal {
                reason: "Laguna decoder state could not be allocated".to_owned(),
            }
        })?;
        let mut performance_attribution = inference_request.into_performance_attribution();
        let prompt_processing_chunk_sizer =
            self.prompt_processing_chunk_sizer
                .as_mut()
                .ok_or(InferenceEngineError::Fatal {
                    reason: "Laguna prompt-processing chunk sizer is missing".to_owned(),
                })?;
        let persistent_prompt_cache = self.persistent_prompt_cache.clone();
        let (last_published_block_key, restored_prompt_prefix_token_count) =
            if let Some(persistent_prompt_cache) = persistent_prompt_cache.as_deref() {
                super::prompt_cache::restore_prompt_prefix(
                    runtime,
                    persistent_prompt_cache,
                    &prompt_token_ids,
                    &mut decoder_state,
                    &mut performance_attribution,
                )?
            } else {
                (None, 0)
            };
        if persistent_prompt_cache.is_some() {
            if restored_prompt_prefix_token_count == 0 {
                self.persistent_prompt_cache_counters.record_cache_miss();
            } else {
                self.persistent_prompt_cache_counters
                    .record_cache_hit(restored_prompt_prefix_token_count as usize);
            }
        }
        prompt_processing_chunk_sizer
            .start_prompt_processing_request(restored_prompt_prefix_token_count as usize);
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
            self.active_request = None;
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
        let Some(model) = self.model.as_ref() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            });
        };
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
            let first_generated_token_id = Self::sample_next_token_id(
                runtime,
                &prompt_logits,
                &mut active_request.performance_attribution,
            )?;
            active_request.next_input_token_ids = vec![first_generated_token_id];
        }
        if let Some(mlx_ram_budget) = self.mlx_ram_budget.as_ref() {
            let decode_retained_expert_budget_bytes = laguna_ram_budget_snapshot(
                mlx_ram_budget,
                MlxRamBudgetPhase::Decode,
                active_request.context_token_count,
            )
            .retained_expert_budget_bytes;
            model
                .set_retained_expert_ceiling(decode_retained_expert_budget_bytes)
                .map_err(|_| InferenceEngineError::Fatal {
                    reason: "Laguna decode expert budget could not be applied".to_owned(),
                })?;
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
        let decode_logits = model
            .forward(
                runtime,
                &decode_token_array,
                &mut active_request.decoder_state,
                &mut active_request.performance_attribution,
            )
            .map_err(|decode_error| {
                tracing::error!(?decode_error, "Laguna decode forward failed");
                InferenceEngineError::Fatal {
                    reason: "Laguna token generation failed".to_owned(),
                }
            })?;
        let next_token_id = Self::sample_next_token_id(
            runtime,
            &decode_logits,
            &mut active_request.performance_attribution,
        )?;
        active_request.next_input_token_ids = vec![next_token_id];
        active_request.remaining_output_tokens =
            active_request.remaining_output_tokens.saturating_sub(1);
        active_request.context_token_count = active_request.context_token_count.saturating_add(1);
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::GeneratedTokenCount, 1);
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
        Ok(GeneratedToken::TokenId {
            token_id,
            is_reasoning_token: false,
            expert_memory_mode: Some(model.expert_memory_mode()),
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
        if input_token_ids.is_empty() {
            return Ok(());
        }
        let Some(runtime) = self.runtime.as_ref() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna runtime is not loaded".to_owned(),
            });
        };
        let Some(model) = self.model.as_ref() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            });
        };
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
        let injected_token_array = runtime
            .array_from_u32(
                &input_token_ids,
                &[i32::try_from(input_token_ids.len()).unwrap_or(i32::MAX)],
            )
            .map_err(|_| InferenceEngineError::Fatal {
                reason: "Laguna injected tokens could not be placed on the runtime".to_owned(),
            })?;
        model
            .forward(
                runtime,
                &injected_token_array,
                &mut active_request.decoder_state,
                &mut active_request.performance_attribution,
            )
            .map_err(|_| InferenceEngineError::Fatal {
                reason: "Laguna injected-token processing failed".to_owned(),
            })?;
        if let Some(last_injected_token_id) = input_token_ids.last().copied() {
            active_request.next_input_token_ids = vec![last_injected_token_id];
        }
        Ok(())
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
        let performance_attribution = std::mem::replace(
            &mut active_request.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        // Release request-owned context before measuring and publishing stable model state.
        drop(active_request);
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

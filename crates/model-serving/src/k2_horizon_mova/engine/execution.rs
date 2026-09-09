//! Owner-thread K2 Horizon MoVA execution.

use std::time::Instant;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId, WorkerEvent};
use astronomical_runtime_integration::{MlxArray, MlxCompiledSwiGlu, MlxMemoryLimits, MlxRuntime};

use crate::k2_horizon_mova::model::K2HorizonMoVAKvState;
use crate::k2_horizon_mova::model::{
    FusedExpertDecodeKernels, K2HorizonMoVAModel, K2HorizonMoVAWeights,
};
use crate::k2_horizon_mova::{
    K2HorizonMoVAInferenceRequest, K2HorizonMoVAThinkingBudgetState, ValidatedK2HorizonMoVAArtifact,
};
use crate::sampling_seed::{current_time_millis_since_unix_epoch, resolve_sampling_seed};
use crate::{
    CustomMetalKernelFamily, EngineGenerationStart, EngineLoadResult, GeneratedToken,
    GenerationFinalization, InferenceEngineError, MlxInferenceExecution, MlxMemoryLimitAdjustment,
    MlxMemoryTelemetry, PerformanceAttribution, PerformanceAttributionLog, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheCounters, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, sorted_expert_weighted_sum_kernel,
    worker_process_kernel_capabilities,
};

pub struct K2HorizonMoVAPendingStartup {
    pub validated_artifact: ValidatedK2HorizonMoVAArtifact,
    pub effective_mlx_memory_ceiling_bytes: usize,
    pub allocator_cache_memory_limit_bytes: usize,
    pub prompt_processing_chunk_tokens: u32,
    pub performance_attribution: PerformanceAttribution,
    pub performance_attribution_enabled: bool,
    pub performance_attribution_log: PerformanceAttributionLog,
    pub attribution_model_id: String,
    pub attribution_model_revision: String,
    pub prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
    pub configured_prompt_cache_block_token_count: Option<usize>,
    pub prompt_cache_common_prefix_stride_blocks: u32,
    pub full_attention_kv_state_growth_tokens: u32,
    pub decode_stage_attribution_enabled: bool,
    pub quantized_kv_cache_enabled: bool,
    pub fused_expert_decode_enabled: bool,
}

pub struct K2HorizonMoVAInferenceExecution {
    pending_startup: Option<K2HorizonMoVAPendingStartup>,
    pub(super) model: Option<K2HorizonMoVAModel>,
    pub(super) active: Option<K2HorizonMoVAActiveGeneration>,
    pub(super) ceiling_bytes: u64,
    pub(super) allocator_cache_bytes: u64,
    prompt_processing_chunk_tokens: u32,
    pub(super) performance_attribution_log: PerformanceAttributionLog,
    performance_attribution_enabled: bool,
    pub(super) attribution_model_id: Option<String>,
    pub(super) attribution_model_revision: Option<String>,
    pub(super) total_artifact_payload_bytes: Option<u64>,
    pub(super) model_shard_count: Option<usize>,
    pub(super) persistent_prompt_cache: Option<std::sync::Arc<PersistentPromptCacheDiskStore>>,
    pub(super) persistent_prompt_cache_disk_store_config:
        Option<PersistentPromptCacheDiskStoreConfig>,
    pub(super) persistent_prompt_cache_counters: PersistentPromptCacheCounters,
    configured_prompt_cache_block_token_count: Option<usize>,
    prompt_cache_common_prefix_stride_blocks: u32,
    full_attention_kv_state_growth_tokens: u32,
    decode_stage_attribution_enabled: bool,
    quantized_kv_cache_enabled: bool,
    fused_expert_decode_enabled: bool,
}

pub(super) struct K2HorizonMoVAActiveGeneration {
    pub(super) request_id: RequestId,
    pub(super) remaining_output_tokens: u32,
    pub(super) remaining_prompt_token_ids: Vec<u32>,
    pub(super) prompt_token_ids: Vec<u32>,
    pub(super) processed_prompt_token_count: u32,
    pub(super) caches: Vec<K2HorizonMoVAKvState>,
    pub(super) random_state: MlxArray,
    pub(super) next_input_token_ids: Vec<u32>,
    pub(super) generated_count: u32,
    pub(super) request: K2HorizonMoVAInferenceRequest,
    pub(super) prefill_started_at: Instant,
    pub(super) performance_attribution: PerformanceAttribution,
    pub(super) last_published_block_key: Option<PersistentPromptCacheBlockKey>,
    pub(super) cached_token_count: u32,
    pub(super) thinking_budget: K2HorizonMoVAThinkingBudgetState,
}

impl K2HorizonMoVAInferenceExecution {
    pub(crate) fn pending(pending_startup: K2HorizonMoVAPendingStartup) -> Self {
        let ceiling_bytes = pending_startup.effective_mlx_memory_ceiling_bytes as u64;
        let allocator_cache_bytes = pending_startup.allocator_cache_memory_limit_bytes as u64;
        let prompt_processing_chunk_tokens = pending_startup.prompt_processing_chunk_tokens.max(1);
        Self {
            pending_startup: Some(pending_startup),
            model: None,
            active: None,
            ceiling_bytes,
            allocator_cache_bytes,
            prompt_processing_chunk_tokens,
            performance_attribution_log: PerformanceAttributionLog::disabled(),
            performance_attribution_enabled: false,
            attribution_model_id: None,
            attribution_model_revision: None,
            total_artifact_payload_bytes: None,
            model_shard_count: None,
            persistent_prompt_cache: None,
            persistent_prompt_cache_disk_store_config: None,
            persistent_prompt_cache_counters: PersistentPromptCacheCounters::default(),
            configured_prompt_cache_block_token_count: None,
            prompt_cache_common_prefix_stride_blocks: 1,
            full_attention_kv_state_growth_tokens: 256,
            decode_stage_attribution_enabled: false,
            quantized_kv_cache_enabled: false,
            fused_expert_decode_enabled: false,
        }
    }
}

impl MlxInferenceExecution for K2HorizonMoVAInferenceExecution {
    type Request = K2HorizonMoVAInferenceRequest;

    fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        let pending = self
            .pending_startup
            .take()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "K2 Horizon MoVA startup was already consumed".to_owned(),
            })?;
        let mut performance_attribution = pending.performance_attribution;
        let runtime = MlxRuntime::initialize(
            MlxMemoryLimits::new(
                pending.effective_mlx_memory_ceiling_bytes,
                pending.allocator_cache_memory_limit_bytes,
            )
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("K2 Horizon MoVA MLX limits failed: {error}"),
            })?,
        )
        .map_err(|error| InferenceEngineError::Fatal {
            reason: format!("K2 Horizon MoVA MLX runtime failed: {error}"),
        })?;
        let weights = K2HorizonMoVAWeights::load(
            &runtime,
            &pending.validated_artifact,
            &mut performance_attribution,
        )
        .map_err(super::prefill::execution_error)?;
        let config = pending.validated_artifact.config().clone();
        let compiled_swiglu =
            MlxCompiledSwiGlu::new().map_err(|error| InferenceEngineError::Fatal {
                reason: format!("K2 Horizon MoVA SwiGLU compilation failed: {error}"),
            })?;
        let kernel_capabilities =
            worker_process_kernel_capabilities(&runtime, &mut performance_attribution);
        let sorted_expert_reduction_kernel = if kernel_capabilities
            .is_custom_kernel_supported(CustomMetalKernelFamily::SortedExpertWeightedSum)
        {
            Some(sorted_expert_weighted_sum_kernel().map_err(|error| {
                InferenceEngineError::Fatal {
                    reason: format!(
                        "K2 Horizon MoVA sorted expert reduction kernel failed: {error}"
                    ),
                }
            })?)
        } else {
            None
        };
        let fused_expert_decode_kernels = if pending.fused_expert_decode_enabled
            && kernel_capabilities
                .is_custom_kernel_supported(CustomMetalKernelFamily::FusedQuantizedExpertDecode)
        {
            Some(
                FusedExpertDecodeKernels::new().map_err(|error| InferenceEngineError::Fatal {
                    reason: format!("K2 Horizon MoVA fused expert decode kernel failed: {error}"),
                })?,
            )
        } else {
            None
        };
        self.model = Some(K2HorizonMoVAModel {
            runtime,
            config,
            weights,
            compiled_swiglu,
            sorted_expert_reduction_kernel,
            fused_expert_decode_kernels,
        });
        self.performance_attribution_enabled = pending.performance_attribution_enabled;
        self.performance_attribution_log = pending.performance_attribution_log;
        self.attribution_model_id = Some(pending.attribution_model_id.clone());
        self.attribution_model_revision = Some(pending.attribution_model_revision.clone());
        self.total_artifact_payload_bytes = Some(pending.validated_artifact.total_payload_bytes());
        self.model_shard_count = Some(pending.validated_artifact.shard_count());
        self.configured_prompt_cache_block_token_count =
            pending.configured_prompt_cache_block_token_count;
        self.prompt_cache_common_prefix_stride_blocks =
            pending.prompt_cache_common_prefix_stride_blocks;
        self.full_attention_kv_state_growth_tokens = pending.full_attention_kv_state_growth_tokens;
        self.decode_stage_attribution_enabled = pending.decode_stage_attribution_enabled;
        self.quantized_kv_cache_enabled = pending.quantized_kv_cache_enabled;
        self.fused_expert_decode_enabled = pending.fused_expert_decode_enabled;
        self.persistent_prompt_cache_disk_store_config =
            pending.prompt_cache_disk_store_config.clone();
        if let Some(prompt_cache_disk_store_config) = pending.prompt_cache_disk_store_config {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "K2 Horizon MoVA model is not loaded".to_owned(),
                })?;
            self.persistent_prompt_cache = Some(super::prompt_cache::open_prompt_cache_store(
                pending.effective_mlx_memory_ceiling_bytes,
                &model.config,
                &pending.attribution_model_id,
                &pending.attribution_model_revision,
                pending.configured_prompt_cache_block_token_count,
                pending.prompt_cache_common_prefix_stride_blocks,
                prompt_cache_disk_store_config,
                &mut performance_attribution,
            )?);
        }
        super::memory::reject_unenacted_expert_streaming(
            self.model
                .as_ref()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "K2 Horizon MoVA model is not loaded".to_owned(),
                })?,
            self.ceiling_bytes,
        )?;
        self.record_model_loading_performance_attribution(performance_attribution);
        Ok(EngineLoadResult::new().with_expert_memory_mode(Some(ExpertMemoryMode::Resident)))
    }

    fn start_generation(
        &mut self,
        inference_request: Self::Request,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "K2 Horizon MoVA model is not loaded".to_owned(),
            })?;
        if inference_request.prompt_token_ids().is_empty() {
            return Err(InferenceEngineError::InvalidRequest {
                reason: "K2 Horizon MoVA prompt must contain tokens".to_owned(),
            });
        }
        let mut caches = (0..model.config.num_hidden_layers())
            .map(|_| {
                K2HorizonMoVAKvState::build(
                    self.quantized_kv_cache_enabled,
                    self.full_attention_kv_state_growth_tokens,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("K2 Horizon MoVA KV state is invalid: {error}"),
            })?;
        let sampling_seed = resolve_sampling_seed(
            inference_request.seed(),
            current_time_millis_since_unix_epoch,
        );
        let random_state = model.runtime.random_key(sampling_seed).map_err(|error| {
            InferenceEngineError::Fatal {
                reason: format!("K2 Horizon MoVA random key failed: {error}"),
            }
        })?;
        let mut performance_attribution = if self.performance_attribution_enabled {
            PerformanceAttribution::enabled()
        } else {
            PerformanceAttribution::disabled()
        };
        let prompt_token_ids = inference_request.prompt_token_ids().to_vec();
        let mut last_published_block_key = None;
        let mut cached_token_count = 0;
        if let Some(persistent_prompt_cache) = self.persistent_prompt_cache.clone() {
            let lookup_result = super::prompt_cache::lookup_prompt_prefix(
                &persistent_prompt_cache,
                &prompt_token_ids,
                &mut performance_attribution,
            );
            let mut restored_token_count = lookup_result.restored_token_count();
            let block_token_count = persistent_prompt_cache.model_contract.block_token_count();
            if restored_token_count >= prompt_token_ids.len() && block_token_count > 0 {
                restored_token_count = restored_token_count.saturating_sub(block_token_count);
            }
            if restored_token_count > 0 {
                let (restored_block_key, restored_tokens) =
                    super::prompt_cache::restore_prompt_prefix(
                        &model.runtime,
                        &persistent_prompt_cache,
                        &prompt_token_ids,
                        restored_token_count,
                        &mut caches,
                        &mut performance_attribution,
                    )?;
                last_published_block_key = restored_block_key;
                cached_token_count = restored_tokens;
                self.persistent_prompt_cache_counters
                    .record_cache_hit(restored_tokens as usize);
            } else {
                self.persistent_prompt_cache_counters.record_cache_miss();
            }
        }
        let mut remaining_prompt_token_ids = prompt_token_ids.clone();
        remaining_prompt_token_ids.drain(..cached_token_count as usize);
        let thinking_budget = K2HorizonMoVAThinkingBudgetState::new(
            inference_request.thinking_budget(),
            inference_request
                .forced_thinking_transition_token_ids()
                .to_vec(),
            inference_request.natural_reasoning_end_token_ids().to_vec(),
        )
        .map_err(|budget_error| InferenceEngineError::Fatal {
            reason: format!("K2 Horizon MoVA thinking budget is invalid: {budget_error}"),
        })?;
        self.active = Some(K2HorizonMoVAActiveGeneration {
            request_id: RequestId::new(0),
            remaining_output_tokens: inference_request.max_output_tokens(),
            remaining_prompt_token_ids,
            prompt_token_ids,
            processed_prompt_token_count: cached_token_count,
            caches,
            random_state,
            next_input_token_ids: Vec::new(),
            generated_count: 0,
            request: inference_request,
            prefill_started_at: Instant::now(),
            performance_attribution,
            last_published_block_key,
            cached_token_count,
            thinking_budget,
        });
        Ok(EngineGenerationStart::with_expert_memory_mode(
            cached_token_count,
            ExpertMemoryMode::Resident,
        ))
    }

    fn decode_next_token(
        &mut self,
        request_id: RequestId,
    ) -> Result<GeneratedToken, InferenceEngineError> {
        let prompt_processing_chunk_tokens = self.prompt_processing_chunk_tokens;
        let has_remaining_prompt = self.active.as_ref().is_some_and(|active_generation| {
            !active_generation.remaining_prompt_token_ids.is_empty()
        });
        if has_remaining_prompt {
            let mut prefill_progress = {
                let model = self
                    .model
                    .as_ref()
                    .ok_or_else(|| InferenceEngineError::Fatal {
                        reason: "K2 Horizon MoVA model is not loaded".to_owned(),
                    })?;
                let active = self
                    .active
                    .as_mut()
                    .ok_or_else(|| InferenceEngineError::Fatal {
                        reason: "K2 Horizon MoVA has no active generation".to_owned(),
                    })?;
                active.request_id = request_id;
                super::prefill::prefill_next_chunk(
                    model,
                    active,
                    prompt_processing_chunk_tokens,
                    self.persistent_prompt_cache.as_deref(),
                )?
            };
            if let GeneratedToken::PrefillProgress {
                mlx_memory_telemetry,
                expert_residency_telemetry,
                ..
            } = &mut prefill_progress
            {
                let current_mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
                *expert_residency_telemetry = current_mlx_memory_telemetry
                    .and_then(MlxMemoryTelemetry::expert_residency_telemetry);
                *mlx_memory_telemetry = current_mlx_memory_telemetry;
            }
            return Ok(prefill_progress);
        }
        let token_id = {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "K2 Horizon MoVA model is not loaded".to_owned(),
                })?;
            let active = self
                .active
                .as_mut()
                .ok_or_else(|| InferenceEngineError::Fatal {
                    reason: "K2 Horizon MoVA has no active generation".to_owned(),
                })?;
            active.request_id = request_id;
            let forced_token_id = active
                .thinking_budget
                .next_forced_transition_token_id()
                .map_err(|budget_error| InferenceEngineError::Fatal {
                    reason: format!("K2 Horizon MoVA thinking budget failed: {budget_error}"),
                })?;
            let selected_token_id =
                forced_token_id.or_else(|| active.next_input_token_ids.first().copied());
            match selected_token_id {
                None => None,
                Some(token_id)
                    if forced_token_id.is_none()
                        && model.config.eos_token_ids().contains(&token_id) =>
                {
                    None
                }
                Some(token_id) => {
                    let mut performance_attribution = std::mem::replace(
                        &mut active.performance_attribution,
                        PerformanceAttribution::disabled(),
                    );
                    let hidden_states = performance_attribution
                        .measure_operation(
                            PerformanceOperation::DecodeAdvanceSpan,
                            |performance_attribution| {
                                let stage_attribution = performance_attribution.is_enabled()
                                    && self.decode_stage_attribution_enabled;
                                model.forward(
                                    &[token_id],
                                    &mut active.caches,
                                    performance_attribution,
                                    stage_attribution,
                                )
                            },
                        )
                        .map_err(super::prefill::execution_error)?;
                    let logits = model
                        .logits_for_last_token(&hidden_states)
                        .map_err(super::prefill::execution_error)?;
                    let next_token = performance_attribution
                        .measure_operation(
                            PerformanceOperation::DecodeSamplingSpan,
                            |_performance_attribution| {
                                model.sample_token(
                                    &logits,
                                    &active.request,
                                    &mut active.random_state,
                                )
                            },
                        )
                        .map_err(super::prefill::execution_error)?;
                    let is_reasoning_token = active
                        .thinking_budget
                        .observe_committed_token(token_id)
                        .map_err(|budget_error| InferenceEngineError::Fatal {
                            reason: format!(
                                "K2 Horizon MoVA thinking budget failed: {budget_error}"
                            ),
                        })?;
                    active.remaining_output_tokens =
                        active.remaining_output_tokens.saturating_sub(1);
                    active.generated_count = active.generated_count.saturating_add(1);
                    active.next_input_token_ids = if active.remaining_output_tokens == 0
                        || model.config.eos_token_ids().contains(&next_token)
                    {
                        Vec::new()
                    } else {
                        vec![next_token]
                    };
                    active.performance_attribution = performance_attribution;
                    Some((token_id, is_reasoning_token))
                }
            }
        };
        let Some((token_id, is_reasoning_token)) = token_id else {
            self.take_active_and_record_generation_attribution();
            return Ok(GeneratedToken::EndOfSequence);
        };
        Ok(GeneratedToken::TokenId {
            token_id,
            is_reasoning_token,
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            mlx_memory_telemetry: self.collect_current_mlx_memory_telemetry(),
            first_decode_forward_elapsed_millis: None,
            generation_finalization: None,
        })
    }

    fn collect_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        Ok(self.collect_current_mlx_memory_telemetry())
    }

    fn inject_input_tokens(
        &mut self,
        _request_id: RequestId,
        input_token_ids: Vec<u32>,
    ) -> Result<(), InferenceEngineError> {
        if input_token_ids.is_empty() {
            return Ok(());
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "K2 Horizon MoVA model is not loaded".to_owned(),
            })?;
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| InferenceEngineError::Fatal {
                reason: "K2 Horizon MoVA has no active generation".to_owned(),
            })?;
        let mut performance_attribution = std::mem::replace(
            &mut active.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        let forward_result = model.forward(
            &input_token_ids,
            &mut active.caches,
            &mut performance_attribution,
            false,
        );
        active.performance_attribution = performance_attribution;
        forward_result.map_err(super::prefill::execution_error)?;
        Ok(())
    }

    fn cancel_generation(
        &mut self,
        _request_id: RequestId,
    ) -> Result<GenerationFinalization, InferenceEngineError> {
        self.take_active_and_record_generation_attribution();
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
        Ok(GenerationFinalization::new(
            Some(ExpertMemoryMode::Resident),
            mlx_memory_telemetry,
            mlx_memory_telemetry.and_then(MlxMemoryTelemetry::expert_residency_telemetry),
        ))
    }

    fn collect_persistent_prompt_cache_stats(
        &self,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        Ok(self.persistent_prompt_cache_stats_event())
    }

    fn clear_persistent_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<Option<WorkerEvent>, InferenceEngineError> {
        let Some(persistent_prompt_cache) = self.persistent_prompt_cache.as_ref() else {
            return Ok(None);
        };
        let clear_outcome = persistent_prompt_cache
            .clear_prompt_cache(model_id.as_deref())
            .map_err(|clear_error| InferenceEngineError::Fatal {
                reason: format!("K2 Horizon MoVA prompt cache could not be cleared: {clear_error}"),
            })?;
        Ok(Some(WorkerEvent::PromptCacheCleared {
            model_id: clear_outcome.model_id,
            blocks_removed: clear_outcome.blocks_removed,
            bytes_freed: clear_outcome.bytes_freed,
        }))
    }

    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, InferenceEngineError> {
        self.ceiling_bytes = requested_mlx_memory_ceiling_bytes.max(1);
        self.allocator_cache_bytes = self.allocator_cache_bytes.min(self.ceiling_bytes);
        if let Some(model) = self.model.as_mut() {
            let memory_limits = MlxMemoryLimits::new(
                usize::try_from(self.ceiling_bytes).unwrap_or(usize::MAX),
                usize::try_from(self.allocator_cache_bytes).unwrap_or(usize::MAX),
            )
            .map_err(|error| InferenceEngineError::Fatal {
                reason: format!("K2 Horizon MoVA MLX limits failed: {error}"),
            })?;
            model
                .runtime
                .update_memory_limits(memory_limits)
                .map_err(|error| InferenceEngineError::Fatal {
                    reason: format!(
                        "K2 Horizon MoVA could not apply the MLX memory ceiling: {error}"
                    ),
                })?;
        }
        let mlx_memory_telemetry = self.collect_current_mlx_memory_telemetry();
        Ok(MlxMemoryLimitAdjustment::new(
            self.ceiling_bytes,
            self.allocator_cache_bytes,
            self.minimum_mlx_memory_ceiling_bytes(),
            ExpertMemoryMode::Resident,
            mlx_memory_telemetry,
        ))
    }
}

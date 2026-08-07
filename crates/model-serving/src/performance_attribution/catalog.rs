use std::time::Duration;

/// One stable, domain-specific operation measured on a model-serving critical path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum PerformanceOperation {
    ArtifactValidation,
    TokenizerInitialization,
    MlxRuntimeInitialization,
    ModelSafetensorsMapping,
    ModelTensorBinding,
    ExpertPagerPlanConstruction,
    ResidentWeightMaterializationSynchronizationWait,
    PersistentPromptCacheOpenAndScan,
    ChatCommandValidation,
    ImagePreprocessing,
    VisionEmbeddingGraphConstruction,
    VisionEmbeddingEvaluationSynchronizationWait,
    PromptRendering,
    PromptTokenization,
    GenerationOutputDecoderInitialization,
    MemoryAdmissionSnapshot,
    AdaptiveRamGrowthMemoryAdmission,
    LinearAttentionGraphConstruction,
    FullAttentionGraphConstruction,
    PersistentPromptCachePrefixLookup,
    PersistentPromptCacheKvBlockRead,
    PersistentPromptCacheRecurrentSnapshotRead,
    PersistentPromptCacheStateReconstruction,
    PersistentPromptCacheStateMaterializationSynchronizationWait,
    PersistentPromptCacheStateExtraction,
    PersistentPromptCacheKvBlockSerialization,
    PersistentPromptCacheRecurrentSnapshotSerialization,
    PersistentPromptCacheRetentionCleanup,
    ExpertPageManifestConstruction,
    ExpertPageMemoryBudgetSnapshot,
    PagedRouterGraphConstruction,
    SelectedExpertIdContiguousGraphConstruction,
    SelectedExpertIdEvaluationSynchronizationWait,
    SelectedExpertIdHostMemoryCopy,
    ExpertWeightMemoryCacheLookup,
    ExpertWeightMemoryCacheEviction,
    PagedMoeGraphConstruction,
    ResidentMoeGraphConstruction,
    FinalLogitsGraphConstruction,
    TokenSamplingGraphConstruction,
    MtpHeadForwardGraphConstruction,
    MtpHeadStateEvaluationSynchronizationWait,
    MtpPromptHistoryInitializationSpan,
    MtpTargetVerificationSynchronizationWait,
    MtpRejectedDraftStateRestoration,
    PrefillStateEvaluationSynchronizationWait,
    ExpertBoundedSafetensorsLazyPageConstruction,
    ExpertWeightMemoryCachePageAssemblyGraphConstruction,
    ExpertPagingDiagnosticLogging,
    DecodeAsyncEvaluationSubmission,
    GeneratedTokenItemSynchronizationWait,
    CompletedForwardMemorySnapshot,
    PromptPrefillAdvanceSpan,
    DecodeAdvanceSpan,
    AttentionForwardSpan,
    MlpForwardSpan,
    MlxAllocatorCacheCleanup,
    FinalizedMlxMemorySnapshot,
}

impl PerformanceOperation {
    pub(super) const COUNT: usize = Self::FinalizedMlxMemorySnapshot as usize + 1;
    pub(super) const ALL: [Self; Self::COUNT] = [
        Self::ArtifactValidation,
        Self::TokenizerInitialization,
        Self::MlxRuntimeInitialization,
        Self::ModelSafetensorsMapping,
        Self::ModelTensorBinding,
        Self::ExpertPagerPlanConstruction,
        Self::ResidentWeightMaterializationSynchronizationWait,
        Self::PersistentPromptCacheOpenAndScan,
        Self::ChatCommandValidation,
        Self::ImagePreprocessing,
        Self::VisionEmbeddingGraphConstruction,
        Self::VisionEmbeddingEvaluationSynchronizationWait,
        Self::PromptRendering,
        Self::PromptTokenization,
        Self::GenerationOutputDecoderInitialization,
        Self::MemoryAdmissionSnapshot,
        Self::AdaptiveRamGrowthMemoryAdmission,
        Self::LinearAttentionGraphConstruction,
        Self::FullAttentionGraphConstruction,
        Self::PersistentPromptCachePrefixLookup,
        Self::PersistentPromptCacheKvBlockRead,
        Self::PersistentPromptCacheRecurrentSnapshotRead,
        Self::PersistentPromptCacheStateReconstruction,
        Self::PersistentPromptCacheStateMaterializationSynchronizationWait,
        Self::PersistentPromptCacheStateExtraction,
        Self::PersistentPromptCacheKvBlockSerialization,
        Self::PersistentPromptCacheRecurrentSnapshotSerialization,
        Self::PersistentPromptCacheRetentionCleanup,
        Self::ExpertPageManifestConstruction,
        Self::ExpertPageMemoryBudgetSnapshot,
        Self::PagedRouterGraphConstruction,
        Self::SelectedExpertIdContiguousGraphConstruction,
        Self::SelectedExpertIdEvaluationSynchronizationWait,
        Self::SelectedExpertIdHostMemoryCopy,
        Self::ExpertWeightMemoryCacheLookup,
        Self::ExpertWeightMemoryCacheEviction,
        Self::PagedMoeGraphConstruction,
        Self::ResidentMoeGraphConstruction,
        Self::FinalLogitsGraphConstruction,
        Self::TokenSamplingGraphConstruction,
        Self::MtpHeadForwardGraphConstruction,
        Self::MtpHeadStateEvaluationSynchronizationWait,
        Self::MtpPromptHistoryInitializationSpan,
        Self::MtpTargetVerificationSynchronizationWait,
        Self::MtpRejectedDraftStateRestoration,
        Self::PrefillStateEvaluationSynchronizationWait,
        Self::ExpertBoundedSafetensorsLazyPageConstruction,
        Self::ExpertWeightMemoryCachePageAssemblyGraphConstruction,
        Self::ExpertPagingDiagnosticLogging,
        Self::DecodeAsyncEvaluationSubmission,
        Self::GeneratedTokenItemSynchronizationWait,
        Self::CompletedForwardMemorySnapshot,
        Self::PromptPrefillAdvanceSpan,
        Self::DecodeAdvanceSpan,
        Self::AttentionForwardSpan,
        Self::MlpForwardSpan,
        Self::MlxAllocatorCacheCleanup,
        Self::FinalizedMlxMemorySnapshot,
    ];

    pub(super) const fn identifier(self) -> &'static str {
        match self {
            Self::ArtifactValidation => "artifact_validation",
            Self::TokenizerInitialization => "tokenizer_initialization",
            Self::MlxRuntimeInitialization => "mlx_runtime_initialization",
            Self::ModelSafetensorsMapping => "model_safetensors_mapping",
            Self::ModelTensorBinding => "model_tensor_binding",
            Self::ExpertPagerPlanConstruction => "expert_pager_plan_construction",
            Self::ResidentWeightMaterializationSynchronizationWait => {
                "resident_weight_materialization_synchronization_wait"
            }
            Self::PersistentPromptCacheOpenAndScan => "persistent_prompt_cache_open_and_scan",
            Self::ChatCommandValidation => "chat_command_validation",
            Self::ImagePreprocessing => "image_preprocessing",
            Self::VisionEmbeddingGraphConstruction => "vision_embedding_graph_construction",
            Self::VisionEmbeddingEvaluationSynchronizationWait => {
                "vision_embedding_evaluation_synchronization_wait"
            }
            Self::PromptRendering => "prompt_rendering",
            Self::PromptTokenization => "prompt_tokenization",
            Self::GenerationOutputDecoderInitialization => {
                "generation_output_decoder_initialization"
            }
            Self::MemoryAdmissionSnapshot => "memory_admission_snapshot",
            Self::AdaptiveRamGrowthMemoryAdmission => "adaptive_ram_growth_memory_admission",
            Self::LinearAttentionGraphConstruction => "linear_attention_graph_construction",
            Self::FullAttentionGraphConstruction => "full_attention_graph_construction",
            Self::PersistentPromptCachePrefixLookup => "persistent_prompt_cache_prefix_lookup",
            Self::PersistentPromptCacheKvBlockRead => "persistent_prompt_cache_kv_block_read",
            Self::PersistentPromptCacheRecurrentSnapshotRead => {
                "persistent_prompt_cache_recurrent_snapshot_read"
            }
            Self::PersistentPromptCacheStateReconstruction => {
                "persistent_prompt_cache_state_reconstruction"
            }
            Self::PersistentPromptCacheStateMaterializationSynchronizationWait => {
                "persistent_prompt_cache_state_materialization_synchronization_wait"
            }
            Self::PersistentPromptCacheStateExtraction => {
                "persistent_prompt_cache_state_extraction"
            }
            Self::PersistentPromptCacheKvBlockSerialization => {
                "persistent_prompt_cache_kv_block_serialization"
            }
            Self::PersistentPromptCacheRecurrentSnapshotSerialization => {
                "persistent_prompt_cache_recurrent_snapshot_serialization"
            }
            Self::PersistentPromptCacheRetentionCleanup => {
                "persistent_prompt_cache_retention_cleanup"
            }
            Self::ExpertPageManifestConstruction => "expert_page_manifest_construction",
            Self::ExpertPageMemoryBudgetSnapshot => "expert_page_memory_budget_snapshot",
            Self::PagedRouterGraphConstruction => "paged_router_graph_construction",
            Self::SelectedExpertIdContiguousGraphConstruction => {
                "selected_expert_id_contiguous_graph_construction"
            }
            Self::SelectedExpertIdEvaluationSynchronizationWait => {
                "selected_expert_id_evaluation_synchronization_wait"
            }
            Self::SelectedExpertIdHostMemoryCopy => "selected_expert_id_host_memory_copy",
            Self::ExpertWeightMemoryCacheLookup => "expert_weight_memory_cache_lookup",
            Self::ExpertWeightMemoryCacheEviction => "expert_weight_memory_cache_eviction",
            Self::PagedMoeGraphConstruction => "paged_moe_graph_construction",
            Self::ResidentMoeGraphConstruction => "resident_moe_graph_construction",
            Self::FinalLogitsGraphConstruction => "final_logits_graph_construction",
            Self::TokenSamplingGraphConstruction => "token_sampling_graph_construction",
            Self::MtpHeadForwardGraphConstruction => "mtp_head_forward_graph_construction",
            Self::MtpHeadStateEvaluationSynchronizationWait => {
                "mtp_head_state_evaluation_synchronization_wait"
            }
            Self::MtpPromptHistoryInitializationSpan => "mtp_prompt_history_initialization_span",
            Self::MtpTargetVerificationSynchronizationWait => {
                "mtp_target_verification_synchronization_wait"
            }
            Self::MtpRejectedDraftStateRestoration => "mtp_rejected_draft_state_restoration",
            Self::PrefillStateEvaluationSynchronizationWait => {
                "prefill_state_evaluation_synchronization_wait"
            }
            Self::ExpertBoundedSafetensorsLazyPageConstruction => {
                "expert_bounded_safetensors_lazy_page_construction"
            }
            Self::ExpertWeightMemoryCachePageAssemblyGraphConstruction => {
                "expert_weight_memory_cache_page_assembly_graph_construction"
            }
            Self::ExpertPagingDiagnosticLogging => "expert_paging_diagnostic_logging",
            Self::DecodeAsyncEvaluationSubmission => "decode_async_evaluation_submission",
            Self::GeneratedTokenItemSynchronizationWait => {
                "generated_token_item_synchronization_wait"
            }
            Self::CompletedForwardMemorySnapshot => "completed_forward_memory_snapshot",
            Self::PromptPrefillAdvanceSpan => "prompt_prefill_advance_span",
            Self::DecodeAdvanceSpan => "decode_advance_span",
            Self::AttentionForwardSpan => "attention_forward_span",
            Self::MlpForwardSpan => "mlp_forward_span",
            Self::MlxAllocatorCacheCleanup => "mlx_allocator_cache_cleanup",
            Self::FinalizedMlxMemorySnapshot => "finalized_mlx_memory_snapshot",
        }
    }

    pub(super) const fn contributes_to_attributed_elapsed(self) -> bool {
        !matches!(
            self,
            Self::PromptPrefillAdvanceSpan
                | Self::DecodeAdvanceSpan
                | Self::MtpPromptHistoryInitializationSpan
                | Self::AttentionForwardSpan
                | Self::MlpForwardSpan
        )
    }
}

/// One bounded numerical counter attached to a performance-attribution report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum PerformanceCounter {
    PromptTokenCount,
    RestoredPersistentPromptCacheTokenCount,
    GeneratedTokenCount,
    PrefillChunckCount,
    PrefillCapacityRejectionCount,
    PrefillCapacityRetryCount,
    ExpertWeightMemoryCacheHitCount,
    ExpertWeightMemoryCacheCompleteLayerHitCount,
    ExpertWeightMemoryCacheMissCount,
    ExpertWeightMemoryCacheEvictionCount,
    ExpertWeightDiskPageLoadCount,
    ExpertWeightDiskBatchLoadCount,
    ExpertPageLogicalPayloadBytes,
    PositionalFileReadCallCount,
    PositionalFileReadByteCount,
    PositionalFileReadElapsedNanoseconds,
    PositionalFileReadMaximumElapsedNanoseconds,
    PositionalFileReadMaximumConcurrentCount,
    PositionalFileReadFailureCount,
    ExpertRoutePredictedExpertCount,
    ExpertRouteMatchedExpertCount,
    ExpertRouteCompletelyMatchedLayerCount,
    ExpertRouteExaminedLayerCount,
    MtpMemoryAdmissionFallbackCount,
    MtpAdmittedAttemptCount,
    SpeculativePrefillTargetOnlyPrefixChunckCount,
    SpeculativePrefillTargetOnlyPrefixTokenCount,
    SpeculativePrefillTerminalCaptureChunckCount,
    SpeculativePrefillTerminalMtpHistoryTokenCount,
    MtpPromptHistoryInitializationFallbackCount,
    MtpFeedbackHistoryReseedCount,
    MtpAcceptedDraftCount,
    MtpRejectedDraftCount,
    MtpOperationalFallbackCount,
}

impl PerformanceCounter {
    pub(super) const COUNT: usize = Self::MtpOperationalFallbackCount as usize + 1;
    pub(super) const ALL: [Self; Self::COUNT] = [
        Self::PromptTokenCount,
        Self::RestoredPersistentPromptCacheTokenCount,
        Self::GeneratedTokenCount,
        Self::PrefillChunckCount,
        Self::PrefillCapacityRejectionCount,
        Self::PrefillCapacityRetryCount,
        Self::ExpertWeightMemoryCacheHitCount,
        Self::ExpertWeightMemoryCacheCompleteLayerHitCount,
        Self::ExpertWeightMemoryCacheMissCount,
        Self::ExpertWeightMemoryCacheEvictionCount,
        Self::ExpertWeightDiskPageLoadCount,
        Self::ExpertWeightDiskBatchLoadCount,
        Self::ExpertPageLogicalPayloadBytes,
        Self::PositionalFileReadCallCount,
        Self::PositionalFileReadByteCount,
        Self::PositionalFileReadElapsedNanoseconds,
        Self::PositionalFileReadMaximumElapsedNanoseconds,
        Self::PositionalFileReadMaximumConcurrentCount,
        Self::PositionalFileReadFailureCount,
        Self::ExpertRoutePredictedExpertCount,
        Self::ExpertRouteMatchedExpertCount,
        Self::ExpertRouteCompletelyMatchedLayerCount,
        Self::ExpertRouteExaminedLayerCount,
        Self::MtpMemoryAdmissionFallbackCount,
        Self::MtpAdmittedAttemptCount,
        Self::SpeculativePrefillTargetOnlyPrefixChunckCount,
        Self::SpeculativePrefillTargetOnlyPrefixTokenCount,
        Self::SpeculativePrefillTerminalCaptureChunckCount,
        Self::SpeculativePrefillTerminalMtpHistoryTokenCount,
        Self::MtpPromptHistoryInitializationFallbackCount,
        Self::MtpFeedbackHistoryReseedCount,
        Self::MtpAcceptedDraftCount,
        Self::MtpRejectedDraftCount,
        Self::MtpOperationalFallbackCount,
    ];

    pub(super) const fn identifier(self) -> &'static str {
        match self {
            Self::PromptTokenCount => "prompt_token_count",
            Self::RestoredPersistentPromptCacheTokenCount => {
                "restored_persistent_prompt_cache_token_count"
            }
            Self::GeneratedTokenCount => "generated_token_count",
            Self::PrefillChunckCount => "prefill_chunck_count",
            Self::PrefillCapacityRejectionCount => "prefill_capacity_rejection_count",
            Self::PrefillCapacityRetryCount => "prefill_capacity_retry_count",
            Self::ExpertWeightMemoryCacheHitCount => "expert_weight_memory_cache_hit_count",
            Self::ExpertWeightMemoryCacheCompleteLayerHitCount => {
                "expert_weight_memory_cache_complete_layer_hit_count"
            }
            Self::ExpertWeightMemoryCacheMissCount => "expert_weight_memory_cache_miss_count",
            Self::ExpertWeightMemoryCacheEvictionCount => {
                "expert_weight_memory_cache_eviction_count"
            }
            Self::ExpertWeightDiskPageLoadCount => "expert_weight_disk_page_load_count",
            Self::ExpertWeightDiskBatchLoadCount => "expert_weight_disk_batch_load_count",
            Self::ExpertPageLogicalPayloadBytes => "expert_page_logical_payload_bytes",
            Self::PositionalFileReadCallCount => "positional_file_read_call_count",
            Self::PositionalFileReadByteCount => "positional_file_read_byte_count",
            Self::PositionalFileReadElapsedNanoseconds => {
                "positional_file_read_elapsed_nanoseconds"
            }
            Self::PositionalFileReadMaximumElapsedNanoseconds => {
                "positional_file_read_maximum_elapsed_nanoseconds"
            }
            Self::PositionalFileReadMaximumConcurrentCount => {
                "positional_file_read_maximum_concurrent_count"
            }
            Self::PositionalFileReadFailureCount => "positional_file_read_failure_count",
            Self::ExpertRoutePredictedExpertCount => "expert_route_predicted_expert_count",
            Self::ExpertRouteMatchedExpertCount => "expert_route_matched_expert_count",
            Self::ExpertRouteCompletelyMatchedLayerCount => {
                "expert_route_completely_matched_layer_count"
            }
            Self::ExpertRouteExaminedLayerCount => "expert_route_examined_layer_count",
            Self::MtpMemoryAdmissionFallbackCount => "mtp_memory_admission_fallback_count",
            Self::MtpAdmittedAttemptCount => "mtp_admitted_attempt_count",
            Self::SpeculativePrefillTargetOnlyPrefixChunckCount => {
                "speculative_prefill_target_only_prefix_chunck_count"
            }
            Self::SpeculativePrefillTargetOnlyPrefixTokenCount => {
                "speculative_prefill_target_only_prefix_token_count"
            }
            Self::SpeculativePrefillTerminalCaptureChunckCount => {
                "speculative_prefill_terminal_capture_chunck_count"
            }
            Self::SpeculativePrefillTerminalMtpHistoryTokenCount => {
                "speculative_prefill_terminal_mtp_history_token_count"
            }
            Self::MtpPromptHistoryInitializationFallbackCount => {
                "mtp_prompt_history_initialization_fallback_count"
            }
            Self::MtpFeedbackHistoryReseedCount => "mtp_feedback_history_reseed_count",
            Self::MtpAcceptedDraftCount => "mtp_accepted_draft_count",
            Self::MtpRejectedDraftCount => "mtp_rejected_draft_count",
            Self::MtpOperationalFallbackCount => "mtp_operational_fallback_count",
        }
    }
}

/// Bounded aggregate for every occurrence of one operation in one report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PerformanceOperationMeasurement {
    pub(super) occurrence_count: u64,
    pub(super) total_elapsed_nanoseconds: u64,
    pub(super) minimum_elapsed_nanoseconds: u64,
    pub(super) maximum_elapsed_nanoseconds: u64,
    pub(super) first_started_offset_nanoseconds: u64,
    pub(super) last_ended_offset_nanoseconds: u64,
}

impl PerformanceOperationMeasurement {
    pub(super) const EMPTY: Self = Self {
        occurrence_count: 0,
        total_elapsed_nanoseconds: 0,
        minimum_elapsed_nanoseconds: u64::MAX,
        maximum_elapsed_nanoseconds: 0,
        first_started_offset_nanoseconds: 0,
        last_ended_offset_nanoseconds: 0,
    };

    #[must_use]
    pub const fn occurrence_count(self) -> u64 {
        self.occurrence_count
    }

    #[must_use]
    pub const fn total_elapsed_nanoseconds(self) -> u64 {
        self.total_elapsed_nanoseconds
    }

    #[must_use]
    pub const fn minimum_elapsed_nanoseconds(self) -> u64 {
        self.minimum_elapsed_nanoseconds
    }

    #[must_use]
    pub const fn maximum_elapsed_nanoseconds(self) -> u64 {
        self.maximum_elapsed_nanoseconds
    }

    #[must_use]
    pub const fn first_started_offset_nanoseconds(self) -> u64 {
        self.first_started_offset_nanoseconds
    }

    #[must_use]
    pub const fn last_ended_offset_nanoseconds(self) -> u64 {
        self.last_ended_offset_nanoseconds
    }

    pub(super) fn record(&mut self, started_offset: Duration, ended_offset: Duration) {
        let started_offset_nanoseconds = duration_nanoseconds_saturating(started_offset);
        let ended_offset_nanoseconds = duration_nanoseconds_saturating(ended_offset);
        let elapsed_nanoseconds =
            ended_offset_nanoseconds.saturating_sub(started_offset_nanoseconds);
        if self.occurrence_count == 0 {
            self.first_started_offset_nanoseconds = started_offset_nanoseconds;
        }
        self.occurrence_count = self.occurrence_count.saturating_add(1);
        self.total_elapsed_nanoseconds = self
            .total_elapsed_nanoseconds
            .saturating_add(elapsed_nanoseconds);
        self.minimum_elapsed_nanoseconds =
            self.minimum_elapsed_nanoseconds.min(elapsed_nanoseconds);
        self.maximum_elapsed_nanoseconds =
            self.maximum_elapsed_nanoseconds.max(elapsed_nanoseconds);
        self.last_ended_offset_nanoseconds = ended_offset_nanoseconds;
    }
}

fn duration_nanoseconds_saturating(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

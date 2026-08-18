//! Stable timing boundaries for model loading and request execution.
//!
//! Enum order indexes a fixed enabled-only accumulator. Outer spans locate
//! latency by request phase, while leaf operations attribute concrete work;
//! overlapping outer spans are serialized but excluded from the attributed sum.

/// One stable, domain-specific operation measured on a model-serving critical path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum PerformanceOperation {
    ArtifactValidation,
    StandaloneMtpArtifactValidation,
    MtpPairingCompatibilityValidation,
    TokenizerInitialization,
    MlxRuntimeInitialization,
    ModelSafetensorsMapping,
    ModelTensorBinding,
    StandaloneMtpTensorBinding,
    StandaloneMtpMaterializationSynchronizationWait,
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
    PersistentPromptCachePublicationValidation,
    PersistentPromptCacheGlobalQuotaEviction,
    PersistentPromptCacheRetentionCleanup,
    PersistentPromptCachePublicationSynchronizationWait,
    PersistentPromptCacheAtomicCommit,
    PagedRouterGraphConstruction,
    RetainedExpertPagePlanning,
    PhaseAwareExpertResidencyPlanning,
    RustExpertStreamingLayerPreparation,
    MandatoryPrefillCompleteLayerMaterializationWait,
    MandatoryDecodeRoutePageMaterializationWait,
    ExpertResidencyCommit,
    GenerationPreparation,
    ExpertRetentionReclamation,
    PagedMoeGraphConstruction,
    PagedMoeOutputMaterializationSynchronizationWait,
    ResidentMoeGraphConstruction,
    FinalLogitsGraphConstruction,
    SpeculativePrefillRequestScopedDraftLoad,
    SpeculativePrefillRequestScopedDraftRelease,
    SpeculativePrefillDraftScoring,
    SpeculativePrefillDraftMemoryAdmission,
    SpeculativePrefillDraftVisionEmbeddingGraphConstruction,
    SpeculativePrefillDraftVisionEmbeddingEvaluationSynchronizationWait,
    SpeculativePrefillSelection,
    SpeculativePrefillSelectionDiskRead,
    SpeculativePrefillSelectionDiskWrite,
    SpeculativePrefillSparseInputAssembly,
    SpeculativePrefillSparseTargetForward,
    TokenSamplingGraphConstruction,
    MtpHeadForwardGraphConstruction,
    MtpHeadStateEvaluationSynchronizationWait,
    MtpPromptHistoryInitializationSpan,
    MtpTargetVerificationSynchronizationWait,
    MtpRejectedDraftStateRestoration,
    MtpTargetRepair,
    MtpPredictorCommitReplay,
    MtpQueuedFrontierRestoration,
    MtpRequestStateCleanup,
    PrefillStateAsyncEvaluationSubmission,
    PrefillStateGraphicsProcessorCompletionWait,
    ExpertPagingDiagnosticLogging,
    DecodeAsyncEvaluationSubmission,
    GeneratedTokenItemSynchronizationWait,
    CompletedForwardMemorySnapshot,
    PromptPrefillAdvanceSpan,
    DecodeAdvanceSpan,
    AttentionForwardSpan,
    SlidingWindowMaskConstruction,
    RotaryEmbeddingApplication,
    RotatingKeyValueStateUpdate,
    SoftplusAttentionGateApplication,
    ExpertAssignmentPreparation,
    GatheredExpertExecution,
    ExpertWeightedReduction,
    RouterScoreSelection,
    SharedExpertExecution,
    MlpForwardSpan,
    GenerationFinalization,
    MlxAllocatorCacheCleanup,
    FinalizedMlxMemorySnapshot,
}

impl PerformanceOperation {
    pub(super) const COUNT: usize = Self::FinalizedMlxMemorySnapshot as usize + 1;
    pub(super) const ALL: [Self; Self::COUNT] = [
        Self::ArtifactValidation,
        Self::StandaloneMtpArtifactValidation,
        Self::MtpPairingCompatibilityValidation,
        Self::TokenizerInitialization,
        Self::MlxRuntimeInitialization,
        Self::ModelSafetensorsMapping,
        Self::ModelTensorBinding,
        Self::StandaloneMtpTensorBinding,
        Self::StandaloneMtpMaterializationSynchronizationWait,
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
        Self::PersistentPromptCachePublicationValidation,
        Self::PersistentPromptCacheGlobalQuotaEviction,
        Self::PersistentPromptCacheRetentionCleanup,
        Self::PersistentPromptCachePublicationSynchronizationWait,
        Self::PersistentPromptCacheAtomicCommit,
        Self::PagedRouterGraphConstruction,
        Self::RetainedExpertPagePlanning,
        Self::PhaseAwareExpertResidencyPlanning,
        Self::RustExpertStreamingLayerPreparation,
        Self::MandatoryPrefillCompleteLayerMaterializationWait,
        Self::MandatoryDecodeRoutePageMaterializationWait,
        Self::ExpertResidencyCommit,
        Self::GenerationPreparation,
        Self::ExpertRetentionReclamation,
        Self::PagedMoeGraphConstruction,
        Self::PagedMoeOutputMaterializationSynchronizationWait,
        Self::ResidentMoeGraphConstruction,
        Self::FinalLogitsGraphConstruction,
        Self::SpeculativePrefillRequestScopedDraftLoad,
        Self::SpeculativePrefillRequestScopedDraftRelease,
        Self::SpeculativePrefillDraftScoring,
        Self::SpeculativePrefillDraftMemoryAdmission,
        Self::SpeculativePrefillDraftVisionEmbeddingGraphConstruction,
        Self::SpeculativePrefillDraftVisionEmbeddingEvaluationSynchronizationWait,
        Self::SpeculativePrefillSelection,
        Self::SpeculativePrefillSelectionDiskRead,
        Self::SpeculativePrefillSelectionDiskWrite,
        Self::SpeculativePrefillSparseInputAssembly,
        Self::SpeculativePrefillSparseTargetForward,
        Self::TokenSamplingGraphConstruction,
        Self::MtpHeadForwardGraphConstruction,
        Self::MtpHeadStateEvaluationSynchronizationWait,
        Self::MtpPromptHistoryInitializationSpan,
        Self::MtpTargetVerificationSynchronizationWait,
        Self::MtpRejectedDraftStateRestoration,
        Self::MtpTargetRepair,
        Self::MtpPredictorCommitReplay,
        Self::MtpQueuedFrontierRestoration,
        Self::MtpRequestStateCleanup,
        Self::PrefillStateAsyncEvaluationSubmission,
        Self::PrefillStateGraphicsProcessorCompletionWait,
        Self::ExpertPagingDiagnosticLogging,
        Self::DecodeAsyncEvaluationSubmission,
        Self::GeneratedTokenItemSynchronizationWait,
        Self::CompletedForwardMemorySnapshot,
        Self::PromptPrefillAdvanceSpan,
        Self::DecodeAdvanceSpan,
        Self::AttentionForwardSpan,
        Self::SlidingWindowMaskConstruction,
        Self::RotaryEmbeddingApplication,
        Self::RotatingKeyValueStateUpdate,
        Self::SoftplusAttentionGateApplication,
        Self::ExpertAssignmentPreparation,
        Self::GatheredExpertExecution,
        Self::ExpertWeightedReduction,
        Self::RouterScoreSelection,
        Self::SharedExpertExecution,
        Self::MlpForwardSpan,
        Self::GenerationFinalization,
        Self::MlxAllocatorCacheCleanup,
        Self::FinalizedMlxMemorySnapshot,
    ];

    pub(super) const fn identifier(self) -> &'static str {
        match self {
            Self::ArtifactValidation => "artifact_validation",
            Self::StandaloneMtpArtifactValidation => "standalone_mtp_artifact_validation",
            Self::MtpPairingCompatibilityValidation => "mtp_pairing_compatibility_validation",
            Self::TokenizerInitialization => "tokenizer_initialization",
            Self::MlxRuntimeInitialization => "mlx_runtime_initialization",
            Self::ModelSafetensorsMapping => "model_safetensors_mapping",
            Self::ModelTensorBinding => "model_tensor_binding",
            Self::StandaloneMtpTensorBinding => "standalone_mtp_tensor_binding",
            Self::StandaloneMtpMaterializationSynchronizationWait => {
                "standalone_mtp_materialization_synchronization_wait"
            }
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
            Self::PersistentPromptCachePublicationValidation => {
                "persistent_prompt_cache_publication_validation"
            }
            Self::PersistentPromptCacheGlobalQuotaEviction => {
                "persistent_prompt_cache_global_quota_eviction"
            }
            Self::PersistentPromptCacheRetentionCleanup => {
                "persistent_prompt_cache_retention_cleanup"
            }
            Self::PersistentPromptCachePublicationSynchronizationWait => {
                "persistent_prompt_cache_publication_synchronization_wait"
            }
            Self::PersistentPromptCacheAtomicCommit => "persistent_prompt_cache_atomic_commit",
            Self::PagedRouterGraphConstruction => "paged_router_graph_construction",
            Self::RetainedExpertPagePlanning => "retained_expert_page_planning",
            Self::PhaseAwareExpertResidencyPlanning => "phase_aware_expert_residency_planning",
            Self::RustExpertStreamingLayerPreparation => "rust_expert_streaming_layer_preparation",
            Self::MandatoryPrefillCompleteLayerMaterializationWait => {
                "mandatory_prefill_complete_layer_materialization_wait"
            }
            Self::MandatoryDecodeRoutePageMaterializationWait => {
                "mandatory_decode_route_page_materialization_wait"
            }
            Self::ExpertResidencyCommit => "expert_residency_commit",
            Self::GenerationPreparation => "generation_preparation",
            Self::ExpertRetentionReclamation => "expert_retention_reclamation",
            Self::PagedMoeGraphConstruction => "paged_moe_graph_construction",
            Self::PagedMoeOutputMaterializationSynchronizationWait => {
                "paged_moe_output_materialization_synchronization_wait"
            }
            Self::ResidentMoeGraphConstruction => "resident_moe_graph_construction",
            Self::FinalLogitsGraphConstruction => "final_logits_graph_construction",
            Self::SpeculativePrefillRequestScopedDraftLoad => {
                "speculative_prefill_request_scoped_draft_load"
            }
            Self::SpeculativePrefillRequestScopedDraftRelease => {
                "speculative_prefill_request_scoped_draft_release"
            }
            Self::SpeculativePrefillDraftScoring => "speculative_prefill_draft_scoring",
            Self::SpeculativePrefillDraftMemoryAdmission => {
                "speculative_prefill_draft_memory_admission"
            }
            Self::SpeculativePrefillDraftVisionEmbeddingGraphConstruction => {
                "speculative_prefill_draft_vision_embedding_graph_construction"
            }
            Self::SpeculativePrefillDraftVisionEmbeddingEvaluationSynchronizationWait => {
                "speculative_prefill_draft_vision_embedding_evaluation_synchronization_wait"
            }
            Self::SpeculativePrefillSelection => "speculative_prefill_selection",
            Self::SpeculativePrefillSelectionDiskRead => "speculative_prefill_selection_disk_read",
            Self::SpeculativePrefillSelectionDiskWrite => {
                "speculative_prefill_selection_disk_write"
            }
            Self::SpeculativePrefillSparseInputAssembly => {
                "speculative_prefill_sparse_input_assembly"
            }
            Self::SpeculativePrefillSparseTargetForward => {
                "speculative_prefill_sparse_target_forward"
            }
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
            Self::MtpTargetRepair => "mtp_target_repair",
            Self::MtpPredictorCommitReplay => "mtp_predictor_commit_replay",
            Self::MtpQueuedFrontierRestoration => "mtp_queued_frontier_restoration",
            Self::MtpRequestStateCleanup => "mtp_request_state_cleanup",
            Self::PrefillStateAsyncEvaluationSubmission => {
                "prefill_state_async_evaluation_submission"
            }
            Self::PrefillStateGraphicsProcessorCompletionWait => {
                "prefill_state_graphics_processor_completion_wait"
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
            Self::SlidingWindowMaskConstruction => "sliding_window_mask_construction",
            Self::RotaryEmbeddingApplication => "rotary_embedding_application",
            Self::RotatingKeyValueStateUpdate => "rotating_key_value_state_update",
            Self::SoftplusAttentionGateApplication => "softplus_attention_gate_application",
            Self::ExpertAssignmentPreparation => "expert_assignment_preparation",
            Self::GatheredExpertExecution => "gathered_expert_execution",
            Self::ExpertWeightedReduction => "expert_weighted_reduction",
            Self::RouterScoreSelection => "router_score_selection",
            Self::SharedExpertExecution => "shared_expert_execution",
            Self::MlpForwardSpan => "mlp_forward_span",
            Self::GenerationFinalization => "generation_finalization",
            Self::MlxAllocatorCacheCleanup => "mlx_allocator_cache_cleanup",
            Self::FinalizedMlxMemorySnapshot => "finalized_mlx_memory_snapshot",
        }
    }

    pub(super) const fn contributes_to_attributed_elapsed(self) -> bool {
        // These operations contain one or more separately recorded leaves. They
        // remain useful timeline evidence but counting them again would inflate
        // attributed time beyond the report's wall-clock duration.
        !matches!(
            self,
            Self::PromptPrefillAdvanceSpan
                | Self::DecodeAdvanceSpan
                | Self::SpeculativePrefillDraftScoring
                | Self::SpeculativePrefillSparseTargetForward
                | Self::MtpPromptHistoryInitializationSpan
                | Self::AttentionForwardSpan
                | Self::MlpForwardSpan
                | Self::GenerationPreparation
        )
    }
}

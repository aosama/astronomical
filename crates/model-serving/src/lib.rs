#![forbid(unsafe_code)]

mod artifact_validation;
mod attention;
mod decoder_cache;
mod deepseek_v4;
mod engine_backed_worker;
mod expert_paging;
mod inference_engine;
mod laguna;
mod memory;
mod model_family_runtime;
mod model_generation_processor;
mod performance_attribution;
mod persistent_cache;
#[path = "prompt_processing_chunk_size_optimizer/mod.rs"]
mod prompt_processing_chunk_size_optimizer;
mod qwen3_5;
mod qwen3_5_moe;
mod safetensors;
mod sparse_experts;
mod strict_json;

#[doc(hidden)]
pub use artifact_validation::validate_required_file_for_tests;
#[doc(hidden)]
pub use artifact_validation::validate_safetensors_profile_partitions_for_tests;
pub use artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, TensorDeclarationOrigin, TensorDtype,
    TensorFeature, TensorInventory, TensorInventoryError, TensorLocation, TensorProfile,
    TensorSemanticRole, TensorSourceId, ValidatedWeightsFile,
};
#[doc(hidden)]
pub use artifact_validation::{
    RawSafetensorsInventoryForTests, RawSafetensorsTensorDescriptorForTests,
};
pub use astronomical_ipc_protocol::ExpertMemoryMode;
#[cfg(feature = "direct-mlx")]
pub use attention::build_causal_sliding_window_mask;
pub use attention::{
    RopeFrequencyError, RotatingAdmissionError, SlidingWindowVisibilityError,
    YarnRopeFrequencyDenominators, compute_default_rope_frequency_denominators,
    compute_yarn_rope_frequency_denominators, rotating_committed_token_count,
    rotating_prefill_transient_token_count, sliding_window_position_is_visible,
    sliding_window_visibility_table,
};
#[cfg(feature = "direct-mlx")]
pub use decoder_cache::{
    ConvolutionState, ConvolutionStateBoundaryCheckpointUpdate, DecoderCacheState,
    DecoderCacheStateAllocationCheckpoint, FullAttentionKeyValueState,
    FullAttentionKeyValueStateAllocationCheckpoint, GatedDeltaRecurrentState,
    RotatingKeyValueState, RotatingKeyValueStateAllocationCheckpoint,
};
pub use decoder_cache::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError,
    DecoderCachePersistedTensorLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};
pub use deepseek_v4::{
    DeepSeekV4UnavailableGenerationProcessor, DeepSeekV4UnavailableInferenceEngine,
    DeepSeekV4UnavailableInferenceRequest, DeepSeekV4UnavailableRequestOutput,
    deepseek_v4_unavailable_reason,
};
pub use engine_backed_worker::{EngineBackedWorker, ModelFactory, WorkerRuntimeError};
pub use expert_paging::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertManifestError, ExpertPageRoutePartition, ExpertResidencyPhase,
    ExpertWeightMemoryCacheStatistics, ExpertWeightPage, PagedDecodeLayerDisposition,
    PhaseAwareExpertResidencyPlan, PhaseAwareExpertResidencyPlanError, QuantizationMode,
    QuantizedExpertLayerPlan, QuantizedExpertPageManifest, QuantizedExpertShardManifest,
    QuantizedExpertSourceInterval, QuantizedExpertTensorRange, QuantizedTensorSource,
    RetainedExpertLayerCache, RetainedExpertLayerCommit, RetainedExpertLayerCommitDelta,
    RetainedExpertLayerCommitError, RetainedExpertLayerCommitOutcome, RetainedExpertPageClass,
    RetainedExpertReclamation, SafetensorsDtype, SafetensorsHeader, SafetensorsHeaderError,
    TensorHeaderEntry, build_quantized_expert_page_manifest_from_plan,
    last_prefill_chunk_demand_weight, parse_safetensors_header, plan_phase_aware_expert_residency,
    validate_expert_ids, validate_quantization_contract, validate_source_intervals,
    validate_virtual_intervals,
};
pub use inference_engine::{
    EngineGenerationStart, EngineLoadResult, ExpertResidencyTelemetry, GeneratedToken,
    GenerationFinalization, InferenceEngine, InferenceEngineError, MlxInferenceEngine,
    MlxInferenceExecution, PreparedInferenceRequest,
    PromptProcessingChunkCandidateMeasurementSummary, PromptProcessingChunkOptimizationContext,
    PromptProcessingChunkOptimizationOutcome,
};
pub use laguna::{
    LagunaAffineProfile, LagunaArtifactValidationError, LagunaArtifactValidator,
    LagunaAttentionDescriptor, LagunaAttentionKind, LagunaAttentionProjection,
    LagunaBlockFp8Profile, LagunaCacheDescriptor, LagunaCanonicalSourceLayout,
    LagunaCanonicalTensorAssemblyKind, LagunaCanonicalTensorDescriptor,
    LagunaCompressedFeedForwardProjection, LagunaCompressedIgnoreScope,
    LagunaCompressedInputActivationDescriptor, LagunaCompressedModuleScope,
    LagunaCompressedStorageDescriptor, LagunaCompressedWeightEncoding, LagunaDefaultRopeDescriptor,
    LagunaDenseFeedForwardDescriptor, LagunaDirectAffineStorageDescriptor,
    LagunaExactStorageSupport, LagunaExecutionDtype, LagunaExecutionError,
    LagunaExpertGateUpLayout, LagunaExpertPagingPlan, LagunaExpertProjection,
    LagunaFeedForwardDescriptor, LagunaFp8InputActivationDescriptor, LagunaFp8KvCacheDescriptor,
    LagunaGatingKind, LagunaGenerationProcessor, LagunaGlobalTensorRole,
    LagunaIndexTotalSizeSemantics, LagunaInferenceRequest, LagunaLayerDescriptor,
    LagunaLayerTensorRole, LagunaModelDescriptor, LagunaMoeDescriptor,
    LagunaNonExecutableMetadataDescriptor, LagunaNormalizationError,
    LagunaNvfp4InputActivationDescriptor, LagunaNvfp4Profile, LagunaOutputEvent,
    LagunaOutputParser, LagunaOutputParserError, LagunaPagingError, LagunaPreparationError,
    LagunaPreparedGeneration, LagunaPromptProcessingChunkSizer,
    LagunaPromptProcessingChunkSizerError, LagunaPromptProcessingExecutionProfile,
    LagunaPromptRenderer, LagunaPromptRendererError, LagunaRawTensorNameRecord,
    LagunaRequestMemoryRequirements, LagunaRequestOutput, LagunaRequestOutputError,
    LagunaRetainedArtifactFiles, LagunaRopeDescriptor, LagunaRouterKind, LagunaRouterSelection,
    LagunaSamplerConfig, LagunaSamplingStrategy, LagunaShardIndex, LagunaShardIndexError,
    LagunaSparseLayerPagingPlan, LagunaStorageDescriptor, LagunaSymmetricPackedAffineProfile,
    LagunaTargetContract, LagunaTargetNormalizer, LagunaTensorAssembly, LagunaTensorComponent,
    LagunaTensorContract, LagunaTensorId, LagunaTensorNameContract,
    LagunaTensorNameNormalizationError, LagunaTensorNameNormalizer, LagunaTensorSource,
    LagunaTensorSourceDescriptor, LagunaTensorSourceRole, LagunaTensorStorageEncoding,
    LagunaTextArtifactDescriptor, LagunaTextArtifactError, LagunaTextArtifactNormalizer,
    LagunaTextArtifactSources, LagunaTokenDecoder, LagunaTokenizer, LagunaTokenizerError,
    LagunaYarnRopeDescriptor, ValidatedLagunaArtifact, apply_router_logit_softcap,
    laguna_decoder_cache_layout, laguna_sliding_prefill_transient_token_count,
    select_laguna_router_experts,
};
#[cfg(feature = "direct-mlx")]
pub use laguna::{
    LagunaDecoderState, LagunaEngine, LagunaExpertWeightPage, LagunaInferenceExecution,
    LagunaModel, LagunaNativeWeights, LagunaServingSettings, LagunaStartupError,
    forward_paged_routed_swiglu, initialize_laguna_execution,
    initialize_laguna_execution_with_serving_settings, initialize_laguna_model,
    initialize_laguna_model_with_serving_settings, load_laguna_expert_page,
    route_laguna_native_experts,
};
pub use memory::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthPhase, AdaptiveRamGrowthProjection, AllocationAdmissionDecision,
    AllocationAdmissionObservation, BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES,
    CompleteResidencyDecision, CompleteResidencyRequirements, ContextAdmissionRequirements,
    ExpertMemoryAdmissionError, ExpertRetentionReclamationPlan, ForwardRecoveryDecision,
    ForwardRecoveryPolicy, ForwardRecoveryRequirements, MemoryAdmissionDecision, MemoryBoundary,
    MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements, MlxActiveMemoryBreakdown,
    MlxMemoryLimitAdjustment, MlxMemoryTelemetry, MlxRamBudget, MlxRamBudgetError,
    MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry, MlxRamBudgetPhase, MlxRamBudgetSnapshot,
    SpeculativePrefillAdmission, combined_persistent_growth_bytes,
    complete_residency_exceeds_ceiling_with_activation_headroom,
    expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure, measured_non_expert_forward_growth_bytes,
    persistent_context_restore_workspace_bytes,
    projected_active_memory_after_complete_expert_replacement,
    required_complete_residency_activation_headroom_bytes, retained_expert_payload_capacity_bytes,
    safe_minimum_active_memory_ceiling_bytes, should_retry_fixed_forward_after_expert_reclamation,
};
#[cfg(feature = "direct-mlx")]
pub use memory::{MlxAllocationBudget, MlxAllocationBudgetError};
#[cfg(feature = "direct-mlx")]
pub use model_family_runtime::ModelFamilyInferenceEngine;
pub use model_family_runtime::{
    ModelFamilyGenerationProcessor, ModelFamilyInferenceRequest, ModelFamilyRequestOutput,
};
pub use model_generation_processor::{
    MalformedModelOutputDiagnostic, ModelGeneratedTokenTranslation, ModelGenerationOutputError,
    ModelGenerationProcessor, PreparedModelGeneration,
};
pub use performance_attribution::{
    GenerationPerformanceAttributionMetadata, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionLog, PerformanceAttributionOutcome,
    PerformanceAttributionReport, PerformanceCounter, PerformanceOperation,
    PerformanceOperationMeasurement,
};
pub use persistent_cache::{
    PERSISTENT_SPECULATIVE_PREFILL_SELECTION_FORMAT_VERSION,
    PERSISTENT_SPECULATIVE_PREFILL_TARGET_STATE_FORMAT_VERSION,
    PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION, PersistentPromptCacheBlockError,
    PersistentPromptCacheBlockHeader, PersistentPromptCacheBlockKey,
    PersistentPromptCacheBlockKeyError, PersistentPromptCacheCounters,
    PersistentPromptCacheLookupDiagnostics, PersistentPromptCacheMissReason,
    PersistentPromptCacheModelContract, PersistentPromptCacheModelContractError,
    PersistentPromptCachePrefixLookup, PersistentPromptCachePrefixLookupResult,
    PersistentSpeculativePrefillPolicyIdentity, PersistentSpeculativePrefillSelectionContract,
    PersistentSpeculativePrefillTargetStateContract, PersistentVisualEmbeddingFileError,
    PersistentVisualEmbeddingFileHeader, PersistentVisualEmbeddingKey,
    PersistentVisualEmbeddingModelContract, longest_reusable_speculative_prefill_target_prefix,
    persistent_prompt_cache_boundary_clamped_prefill_chunck_end,
    persistent_prompt_cache_boundary_completed_prefill_chunck_tokens,
};
#[cfg(feature = "direct-mlx")]
pub use persistent_cache::{
    PersistentPromptCacheClearOutcome, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, PersistentPromptCacheDiskStoreError,
    PersistentPromptCachePublicationOutcome, PersistentPromptCacheStartupCleanupCategory,
    PersistentPromptCacheStartupCleanupEvidence, PersistentSpeculativePrefillPolicyPurgeOutcome,
    RestoredSpeculativePrefillTargetState, build_persistent_prompt_cache_stats_event,
    clear_persistent_prompt_cache_directory,
};
pub use prompt_processing_chunk_size_optimizer::{
    CandidateChunkMeasurementSummary, CandidateMeasurementSource, CandidateMeasurementSummaries,
    PromptProcessingChunkMeasurement, PromptProcessingChunkSizeOptimizer,
    PromptProcessingChunkSizeOptimizerError, PromptProcessingChunkSizeSelection,
    PromptProcessingChunkSizeSelectionReason, PromptProcessingMeasurementContext,
};
pub use qwen3_5::{
    MAXIMUM_MTPLX_RUNTIME_BYTES, ModelWeightStorage, MtpDepthDowngradeReason, MtpDraftDepth,
    MtpDraftDepthError, MtpMemoryAdmission, MtpMemoryCandidate, MtpMemoryProjection,
    MtpMemoryProjectionError, MtpVerificationDecision, MtpVerificationDecisionError, OptiQMetadata,
    OptiQMetadataError, OptiQQuantizationProfile, Qwen3_5ArtifactError,
    Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5ConfigError,
    Qwen3_5DecoderLayerCacheDtypes, Qwen3_5FeedForwardArchitecture, Qwen3_5GenerationProcessor,
    Qwen3_5ImageDimensions, Qwen3_5ImageGrid, Qwen3_5ImageProcessingError, Qwen3_5ImageProcessor,
    Qwen3_5InferenceRequest, Qwen3_5MtpArtifactCapability, Qwen3_5MtpContract,
    Qwen3_5MtpContractError, Qwen3_5MtpSidecarDeclaration, Qwen3_5MtpSidecarDeclarationError,
    Qwen3_5MtpSidecarValidationOutcome, Qwen3_5MtpTargetOnlyReason, Qwen3_5OutputEvent,
    Qwen3_5OutputParser, Qwen3_5OutputParserError, Qwen3_5ProcessedImage, Qwen3_5PromptError,
    Qwen3_5PromptRenderer, Qwen3_5RenderedPrompt, Qwen3_5RequestOutput, Qwen3_5RequestOutputError,
    Qwen3_5SamplerConfig, Qwen3_5SamplingStrategy, Qwen3_5ShardIndex, Qwen3_5TokenDecoder,
    Qwen3_5TokenIds, Qwen3_5Tokenizer, Qwen3_5TokenizerError, Qwen3_5ToolCall, Qwen3_5VisionConfig,
    Qwen3_5VisionInputPlan, Qwen3_5VisionInputPlanError, Qwen3_5VisualEmbeddingRequiredImage,
    Qwen3_5VisualEmbeddingSuffixPlan, Qwen3_5VisualEmbeddingSuffixPlanError,
    ValidatedQwen3_5Artifact, discover_sampler_config, discover_token_ids,
    plan_qwen3_5_visual_embedding_suffix, qwen3_5_decoder_cache_layout,
    qwen3_5_language_tensor_profiles, qwen3_5_mtp_effective_depth_and_reason_for_windows,
    qwen3_5_mtp_effective_depth_for_windows, qwen3_5_mtp_memory_admission,
    qwen3_5_mtp_tensor_names, qwen3_5_mtp_tensor_profiles, qwen3_5_mtp_verification_decision,
    qwen3_5_mtp_verification_transient_array_bytes, qwen3_5_request_enables_thinking,
    qwen3_5_resident_language_tensor_profiles, qwen3_5_vision_tensor_profiles,
    resolve_sampling_seed, translate_qwen3_5_preparation_error, translate_request_output_error,
    validate_context_token_count, validate_qwen3_5_mtp_sidecar_for_tests,
};
#[cfg(feature = "direct-mlx")]
pub use qwen3_5::{
    Qwen3_5Engine, Qwen3_5ExecutionError, Qwen3_5GatedDeltaBoundaryCheckpointResult, Qwen3_5Model,
    Qwen3_5ModelChunkingConfiguration, Qwen3_5MtpForwardOutput, Qwen3_5MtpRequestState,
    Qwen3_5MtpRequestStateAllocationCheckpoint, Qwen3_5MtpRuntimeState,
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector, Qwen3_5PrefillExecutionContext,
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5PromptProcessingChunkSizerError,
    Qwen3_5SpeculativePrefillFailureStageForTests, Qwen3_5SpeculativePrefillSelectionError,
    Qwen3_5TargetForwardOutput, Qwen3_5TargetVerificationProjection,
    Qwen3_5TargetVerificationProjectionDispatch, Qwen3_5VisionModel, Qwen3_5VisionWeights,
    Qwen3_5Weights, RequestDecoderStateStack, RequestDecoderStateStackAllocationCheckpoint,
    RequestDecoderStateStackCheckpoint, VerifiedEmissionQueue, VerifiedTargetFrontier,
    persistent_prompt_cache_publication_advances_parent_chain,
    qwen3_5_aggregate_speculative_prefill_attention_weights, qwen3_5_apply_top_p_mask,
    qwen3_5_depth_one_mtp_window_fits, qwen3_5_full_attention_step,
    qwen3_5_gated_delta_checkpoint_kernel, qwen3_5_gated_delta_kernel,
    qwen3_5_gated_delta_sequence, qwen3_5_gated_delta_sequence_with_boundary_checkpoints,
    qwen3_5_gated_delta_step, qwen3_5_inject_visual_embeddings,
    qwen3_5_mtp_runtime_configuration_after_load, qwen3_5_mtp_runtime_state_after_load,
    qwen3_5_mtp_verification_may_cross_thinking_budget,
    qwen3_5_select_speculative_prefill_token_positions,
    qwen3_5_select_speculative_prefill_token_positions_on_gpu,
    qwen3_5_selected_speculative_prefill_positions_for_range,
    qwen3_5_target_verification_quantized_linear, safe_minimum_mlx_memory_ceiling_bytes,
    target_verification_quantized_linear_kernel,
};
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use qwen3_5_moe::maximum_resident_gate_up_fusion_transient_payload_bytes;
#[cfg(feature = "direct-mlx")]
pub use qwen3_5_moe::{
    ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5MoEPagedPrefillExecutionMode,
    Qwen3_5MoESplitPageRoute, build_quantized_expert_layer_plan, build_source_manifests,
    contiguous_selected_runs, qwen3_5_moe_combine_experts,
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
pub use qwen3_5_moe::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    prefill_recovery_must_demote_complete_resident_owner, retained_expert_fill_budget_bytes,
};
#[cfg(feature = "direct-mlx")]
pub use sparse_experts::{
    SortedExpertAssignments, restore_expert_assignment_order, router_weighted_expert_inputs,
    sort_expert_assignments, sorted_expert_weighted_sum, sorted_expert_weighted_sum_kernel,
    unsorted_expert_weighted_sum,
};
pub use sparse_experts::{
    SparseExpertError, gathered_indices_use_sorted_contract, invert_assignment_order,
};

/// Validates a safetensors shard where some tensors have strict dtype/shape profiles
/// and remaining tensors are accepted by name only.
///
/// Exposed for integration tests that build synthetic multi-tensor shard fixtures.
pub fn validate_bounded_safetensors_with_partial_profiles(
    weights_file: &std::fs::File,
    file_size_bytes: u64,
    weights_file_name: &str,
    profiled_tensor_profiles: &[TensorProfile],
    accepted_extra_tensor_names: &std::collections::HashSet<&str>,
) -> Result<artifact_validation::PartialProfileMetadata, ArtifactValidationError> {
    artifact_validation::validate_bounded_safetensors_with_partial_profiles(
        weights_file,
        file_size_bytes,
        weights_file_name,
        profiled_tensor_profiles,
        accepted_extra_tensor_names,
    )
}

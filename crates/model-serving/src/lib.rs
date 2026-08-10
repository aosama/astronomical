#![forbid(unsafe_code)]

mod artifact_validation;
mod decoder_cache;
mod deepseek_v4;
mod engine_backed_worker;
#[cfg(feature = "direct-mlx")]
mod expert_paging;
mod inference_engine;
mod memory;
mod model_family_runtime;
mod model_generation_processor;
mod performance_attribution;
mod persistent_cache;
#[path = "prefill-chunck-size-optimizer/mod.rs"]
mod prefill_chunck_size_optimizer;
mod qwen3_5;
mod qwen3_5_moe;
mod safetensors;

#[doc(hidden)]
pub use artifact_validation::validate_required_file_for_tests;
pub use artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, TensorDtype, TensorProfile, ValidatedWeightsFile,
};
#[cfg(feature = "direct-mlx")]
pub use decoder_cache::{
    ConvolutionState, ConvolutionStateBoundaryCheckpointUpdate, DecoderCacheState,
    DecoderCacheStateAllocationCheckpoint, FullAttentionKeyValueState,
    FullAttentionKeyValueStateAllocationCheckpoint, GatedDeltaRecurrentState,
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
#[cfg(feature = "direct-mlx")]
pub use expert_paging::{
    ExpertManifestError, ExpertWeightMemoryCache, ExpertWeightMemoryCacheRequestReport,
    ExpertWeightMemoryCacheStatistics, ExpertWeightPage, LiveMetalBudget, MemoryBudgetError,
    MemoryBudgetSnapshot, QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
    QuantizedExpertShardManifest, QuantizedExpertSourceInterval, QuantizedExpertTensorRange,
    QuantizedTensorSource, SafetensorsDtype, SafetensorsHeader, SafetensorsHeaderError,
    TensorHeaderEntry, automatic_expert_weight_memory_cache_maximum_size_bytes,
    build_quantized_expert_page_manifest_from_plan, parse_safetensors_header, validate_expert_ids,
    validate_quantization_contract, validate_source_intervals, validate_virtual_intervals,
};
pub use inference_engine::{
    EngineGenerationStart, EngineLoadResult, GeneratedToken, GenerationFinalization,
    InferenceEngine, InferenceEngineError, MlxInferenceEngine, MlxInferenceExecution,
    PrefillChunckOptimizerCandidateInsight, PrefillChunckOptimizerContextInsight,
    PrefillChunckOptimizerInsight, PreparedInferenceRequest,
};
pub use memory::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthPhase, AdaptiveRamGrowthProjection, MlxActiveMemoryBreakdown,
    MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
};
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
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheDiskStoreError, PersistentPromptCachePublicationOutcome,
    PersistentPromptCacheStartupCleanupCategory, PersistentPromptCacheStartupCleanupEvidence,
    PersistentSpeculativePrefillPolicyPurgeOutcome, RestoredSpeculativePrefillTargetState,
    build_persistent_prompt_cache_stats_event,
};
pub use prefill_chunck_size_optimizer::{
    PrefillChunckOptimizerCandidateEvidence, PrefillChunckOptimizerContextEvidence,
    PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerDecision, PrefillChunckSizeOptimizerDecisionReason,
    PrefillChunckSizeOptimizerError, PrefillChunckSizeOptimizerObservation,
};
pub use qwen3_5::{
    ModelWeightStorage, OptiQMetadata, OptiQMetadataError, OptiQQuantizationProfile,
    Qwen3_5ArtifactError, Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator, Qwen3_5Config,
    Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture, Qwen3_5GenerationProcessor,
    Qwen3_5ImageDimensions, Qwen3_5ImageGrid, Qwen3_5ImageProcessingError, Qwen3_5ImageProcessor,
    Qwen3_5InferenceRequest, Qwen3_5MtpArtifactCapability, Qwen3_5OutputEvent, Qwen3_5OutputParser,
    Qwen3_5OutputParserError, Qwen3_5ProcessedImage, Qwen3_5PromptError, Qwen3_5PromptRenderer,
    Qwen3_5RenderedPrompt, Qwen3_5RequestOutput, Qwen3_5RequestOutputError, Qwen3_5SamplerConfig,
    Qwen3_5SamplingStrategy, Qwen3_5ShardIndex, Qwen3_5TokenDecoder, Qwen3_5TokenIds,
    Qwen3_5Tokenizer, Qwen3_5TokenizerError, Qwen3_5ToolCall, Qwen3_5VisionConfig,
    Qwen3_5VisionInputPlan, Qwen3_5VisionInputPlanError, Qwen3_5VisualEmbeddingRequiredImage,
    Qwen3_5VisualEmbeddingSuffixPlan, Qwen3_5VisualEmbeddingSuffixPlanError,
    ValidatedQwen3_5Artifact, discover_sampler_config, discover_token_ids,
    plan_qwen3_5_visual_embedding_suffix, qwen3_5_decoder_cache_layout,
    qwen3_5_language_tensor_profiles, qwen3_5_mtp_tensor_names, qwen3_5_mtp_tensor_profiles,
    qwen3_5_request_enables_thinking, qwen3_5_resident_language_tensor_profiles,
    qwen3_5_vision_tensor_profiles, resolve_sampling_seed, translate_qwen3_5_preparation_error,
    translate_request_output_error, validate_context_token_count,
};
#[cfg(feature = "direct-mlx")]
pub use qwen3_5::{
    Qwen3_5Engine, Qwen3_5ExecutionError, Qwen3_5GatedDeltaBoundaryCheckpointResult, Qwen3_5Model,
    Qwen3_5ModelChunkingConfiguration, Qwen3_5MtpForwardOutput, Qwen3_5MtpRequestState,
    Qwen3_5MtpRequestStateAllocationCheckpoint, Qwen3_5MtpRuntimeState,
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector, Qwen3_5PrefillChunckSizer,
    Qwen3_5PrefillChunckSizerError, Qwen3_5PrefillExecutionContext,
    Qwen3_5SpeculativePrefillFailureStageForTests, Qwen3_5SpeculativePrefillSelectionError,
    Qwen3_5TargetForwardOutput, Qwen3_5VisionModel, Qwen3_5VisionWeights, Qwen3_5Weights,
    RequestDecoderStateStack, RequestDecoderStateStackAllocationCheckpoint,
    RequestDecoderStateStackCheckpoint, combined_target_and_additional_persistent_growth_bytes,
    context_memory_admission_projected_active_memory_bytes,
    persistent_prompt_cache_publication_advances_parent_chain,
    persistent_prompt_cache_restore_temporary_workspace_bytes,
    qwen3_5_aggregate_speculative_prefill_attention_weights, qwen3_5_apply_top_p_mask,
    qwen3_5_depth_one_mtp_window_fits, qwen3_5_full_attention_step,
    qwen3_5_gated_delta_checkpoint_kernel, qwen3_5_gated_delta_kernel,
    qwen3_5_gated_delta_sequence, qwen3_5_gated_delta_sequence_with_boundary_checkpoints,
    qwen3_5_gated_delta_step, qwen3_5_inject_visual_embeddings,
    qwen3_5_mtp_runtime_state_after_load, qwen3_5_mtp_verification_may_cross_thinking_budget,
    qwen3_5_select_speculative_prefill_token_positions,
    qwen3_5_select_speculative_prefill_token_positions_on_gpu,
    qwen3_5_selected_speculative_prefill_positions_for_range,
    safe_minimum_mlx_memory_ceiling_bytes,
};
#[cfg(feature = "direct-mlx")]
pub use qwen3_5_moe::{
    ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5ExpertWeightMemoryCache,
    Qwen3_5MoEPagedPrefillExecutionMode, Qwen3_5PagedExpertWeights,
    build_quantized_expert_layer_plan, build_source_manifests, contiguous_selected_runs,
    qwen3_5_moe_combine_experts, qwen3_5_moe_remap_expert_page_slots,
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
pub use qwen3_5_moe::{ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION};

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

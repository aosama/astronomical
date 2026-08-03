#![forbid(unsafe_code)]

mod adaptive_ram_growth_guard;
mod bounded_safetensors;
mod bounded_safetensors_header;
mod decoder_cache;
mod engine_backed_worker;
mod engine_backed_worker_construction;
mod engine_backed_worker_fatal;
mod engine_backed_worker_output;
mod engine_backed_worker_protocol;
mod engine_backed_worker_support;
mod error;
mod inference_engine;
mod mlx_memory_telemetry;
mod model_generation_processor;
mod performance_attribution;
mod persistent_cache;
#[path = "prefill-chunck-size-optimizer/mod.rs"]
mod prefill_chunck_size_optimizer;
mod qwen3_5;
mod qwen3_5_moe;
mod required_files;
mod safetensors_dtype;
mod types;
mod validated_artifact;

pub use adaptive_ram_growth_guard::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthPhase, AdaptiveRamGrowthProjection,
};
#[cfg(feature = "direct-mlx")]
pub use decoder_cache::{
    ConvolutionState, DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, DecoderCacheState,
    DecoderCacheStateAllocationCheckpoint, FullAttentionKeyValueState,
    FullAttentionKeyValueStateAllocationCheckpoint, GatedDeltaRecurrentState,
};
pub use decoder_cache::{
    DEFAULT_APPEND_ONLY_ATTENTION_CAPACITY_GROWTH_TOKENS, DecoderCacheLayerLayout,
    DecoderCacheLayout, DecoderCacheLayoutError, DecoderCachePersistedTensorLayout,
    DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};
pub use engine_backed_worker::EngineBackedWorker;
pub use engine_backed_worker_support::{ModelFactory, WorkerRuntimeError};
pub use error::ArtifactValidationError;
pub use inference_engine::{
    EngineGenerationStart, EngineLoadResult, GeneratedToken, GenerationFinalization,
    InferenceEngine, InferenceEngineError, MlxInferenceEngine, MlxInferenceExecution,
    PreparedInferenceRequest,
};
pub use mlx_memory_telemetry::{
    MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
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
    PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT, PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION,
    PersistentPromptCacheBlockError, PersistentPromptCacheBlockHeader,
    PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError,
    PersistentPromptCacheBlockSaveAdmission, PersistentPromptCacheCounters,
    PersistentPromptCacheLookupDiagnostics, PersistentPromptCacheMissReason,
    PersistentPromptCacheModelContract, PersistentPromptCachePrefixLookup,
    PersistentPromptCachePrefixLookupResult, PersistentVisualEmbeddingFileError,
    PersistentVisualEmbeddingFileHeader, PersistentVisualEmbeddingKey,
    PersistentVisualEmbeddingModelContract, persistent_prompt_cache_aligned_prefill_end,
    persistent_prompt_cache_recurrent_snapshot_is_common_prefix_checkpoint,
    persistent_prompt_cache_save_admission,
};
#[cfg(feature = "direct-mlx")]
pub use persistent_cache::{
    PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheDiskStoreError, PersistentPromptCacheWriteQueue,
    PersistentPromptCacheWriteQueueOutcome, PersistentPromptCacheWriteRateLimiter,
    build_persistent_prompt_cache_stats_event, persistent_prompt_cache_write_queue_can_accept,
};
pub use prefill_chunck_size_optimizer::{
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
    Qwen3_5RequestOutput, Qwen3_5RequestOutputError, Qwen3_5SamplerConfig, Qwen3_5SamplingStrategy,
    Qwen3_5ShardIndex, Qwen3_5TokenDecoder, Qwen3_5TokenIds, Qwen3_5Tokenizer,
    Qwen3_5TokenizerError, Qwen3_5ToolCall, Qwen3_5VisionConfig, Qwen3_5VisionInputPlan,
    Qwen3_5VisionInputPlanError, Qwen3_5VisualEmbeddingRequiredImage,
    Qwen3_5VisualEmbeddingSuffixPlan, Qwen3_5VisualEmbeddingSuffixPlanError,
    ValidatedQwen3_5Artifact, discover_sampler_config, discover_token_ids,
    plan_qwen3_5_visual_embedding_suffix, qwen3_5_decoder_cache_layout,
    qwen3_5_language_tensor_profiles, qwen3_5_mtp_tensor_profiles,
    qwen3_5_quantized_mtp_tensor_names, qwen3_5_resident_language_tensor_profiles,
    qwen3_5_vision_tensor_profiles, resolve_sampling_seed, translate_qwen3_5_preparation_error,
    translate_request_output_error, validate_context_token_count,
};
#[cfg(feature = "direct-mlx")]
pub use qwen3_5::{
    Qwen3_5Engine, Qwen3_5ExecutionError, Qwen3_5Model, Qwen3_5MtpForwardOutput,
    Qwen3_5MtpRequestState, Qwen3_5MtpRequestStateAllocationCheckpoint, Qwen3_5MtpRuntimeState,
    Qwen3_5PrefillChunckSizer, Qwen3_5PrefillChunckSizerError, Qwen3_5TargetForwardOutput,
    Qwen3_5VisionModel, Qwen3_5VisionWeights, Qwen3_5Weights, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
    combined_target_and_mtp_persistent_growth_bytes,
    context_memory_admission_projected_active_memory_bytes,
    persistent_prompt_cache_restore_temporary_workspace_bytes, qwen3_5_apply_top_p_mask,
    qwen3_5_depth_one_mtp_window_fits, qwen3_5_full_attention_step, qwen3_5_gated_delta_kernel,
    qwen3_5_gated_delta_sequence, qwen3_5_gated_delta_step, qwen3_5_inject_visual_embeddings,
    qwen3_5_mtp_runtime_state_after_load, qwen3_5_mtp_verification_may_cross_thinking_budget,
    safe_minimum_mlx_memory_ceiling_bytes,
};
#[cfg(feature = "direct-mlx")]
pub use qwen3_5_moe::{
    ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES, AlignedExpertPackBuildRequest,
    AlignedExpertPackError, AlignedExpertPackHeader, AlignedExpertPackPreparationError,
    AlignedExpertPackPreparationInspection, AlignedExpertPackPreparationProgress,
    AlignedExpertPackPreparationReport, AlignedExpertPackPreparer,
    AlignedExpertPackTensorDescriptor, ExpertManifestError, ExpertPager, ExpertPagingError,
    ExpertWeightMemoryCache, ExpertWeightMemoryCacheRequestReport,
    ExpertWeightMemoryCacheStatistics, LiveMetalBudget, MemoryBudgetError, MemoryBudgetSnapshot,
    PagedExpertWeights, QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
    QuantizedExpertShardManifest, QuantizedExpertSourceInterval, QuantizedExpertTensorRange,
    QuantizedTensorSource, Qwen3_5MoEPagedPrefillExecutionMode, SafetensorsDtype,
    SafetensorsHeader, SafetensorsHeaderError, TensorHeaderEntry,
    automatic_expert_weight_memory_cache_maximum_size_bytes, build_aligned_expert_pack,
    build_aligned_expert_pack_metal_io_descriptors, build_quantized_expert_layer_plan,
    build_quantized_expert_page_manifest_from_plan, build_source_manifests,
    contiguous_selected_runs, load_quantized_expert_page, parse_safetensors_header,
    qwen3_5_moe_combine_experts, qwen3_5_moe_remap_expert_page_slots,
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel, read_aligned_expert_pack_header,
    validate_aligned_expert_pack_header, validate_aligned_expert_pack_payload, validate_expert_ids,
    validate_quantization_contract, validate_source_intervals, validate_virtual_intervals,
};
pub use qwen3_5_moe::{ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION};
#[doc(hidden)]
pub use required_files::validate_required_file_for_tests;
pub use types::{RequiredFileProfile, TensorDtype, TensorProfile};
pub use validated_artifact::ValidatedWeightsFile;

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
) -> Result<bounded_safetensors::PartialProfileMetadata, ArtifactValidationError> {
    bounded_safetensors::validate_bounded_safetensors_with_partial_profiles(
        weights_file,
        file_size_bytes,
        weights_file_name,
        profiled_tensor_profiles,
        accepted_extra_tensor_names,
    )
}

/// Validates a safetensors shard where profiled tensors have strict dtype/shape checks
/// and ALL other tensors in the shard are accepted without profiling.
///
/// Used for models with embedded vision tensors where the shard contains both
/// language tensors (which have profiles) and vision tensors (which are accepted
/// as-is). Unlike the partial-profiles variant, this does not require an explicit
/// set of accepted extra tensor names and does not require all profiled tensors
/// to be present in every shard.
pub fn validate_bounded_safetensors_with_permissive_extras(
    weights_file: &std::fs::File,
    file_size_bytes: u64,
    weights_file_name: &str,
    profiled_tensor_profiles: &[TensorProfile],
) -> Result<bounded_safetensors::PartialProfileMetadata, ArtifactValidationError> {
    bounded_safetensors::validate_bounded_safetensors_with_permissive_extras(
        weights_file,
        file_size_bytes,
        weights_file_name,
        profiled_tensor_profiles,
    )
}

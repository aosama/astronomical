// Unsafe is denied workspace-wide. This crate previously carried a source-level
// forbid; it was relaxed to the workspace deny for exactly one module —
// performance_attribution::macos_process_io — whose macOS accounting syscall is
// inseparable from the evidence types it produces (rationale lives there). Any
// new unsafe elsewhere in this crate still fails the build.
mod artifact_validation;
mod attention;
mod decoder_cache;
mod embedding_engine;
mod engine_backed_worker;
mod expert_paging;
mod flux2_klein;
#[cfg(feature = "direct-mlx")]
mod gpu_token_sampling;
mod image_generation_engine;
mod inference_engine;
mod k2_horizon_mova;
mod kernel_capability;
mod memory;
mod model_family_runtime;
mod model_generation_processor;
#[cfg(feature = "direct-mlx")]
mod modernbert;
mod performance_attribution;
mod persistent_cache;
mod qwen3_5;
mod qwen3_5_moe;
mod qwen_image_21;
mod safetensors;
mod sampling_seed;
mod sparse_experts;
mod strict_json;
#[cfg_attr(not(feature = "direct-mlx"), allow(dead_code))]
mod structured_generation;

#[doc(hidden)]
pub use artifact_validation::validate_required_file_for_tests;
pub use artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, TensorDeclarationOrigin, TensorDtype,
    TensorInventory, TensorInventoryError, TensorLocation, TensorProfile, TensorSemanticRole,
    TensorSourceId, ValidatedWeightsFile,
};
#[doc(hidden)]
pub use artifact_validation::{
    RawSafetensorsInventoryForTests, RawSafetensorsTensorDescriptorForTests,
    validate_safetensors_required_profiles_for_tests,
};
pub use astronomical_ipc_protocol::ExpertMemoryMode;
#[cfg(feature = "direct-mlx")]
pub use attention::build_causal_sliding_window_mask;
pub use attention::{
    RopeFrequencyError, SlidingWindowVisibilityError, YarnRopeFrequencyDenominators,
    compute_default_rope_frequency_denominators, compute_yarn_rope_frequency_denominators,
    sliding_window_position_is_visible, sliding_window_visibility_table,
};
#[cfg(feature = "direct-mlx")]
pub use decoder_cache::{
    ConvolutionState, ConvolutionStateBoundaryCheckpointUpdate, DecoderCacheState,
    DecoderCacheStateAllocationCheckpoint, FullAttentionKeyValueState,
    FullAttentionKeyValueStateAllocationCheckpoint, GatedDeltaRecurrentState,
    QuantizedFullAttentionKeyValueState, QuantizedKeyValueViews, QuantizedTensorViews,
    RotatingKeyValueState, RotatingKeyValueStateAllocationCheckpoint,
};
pub use decoder_cache::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError,
    DecoderCachePersistedTensorLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};
pub use embedding_engine::{
    EmbeddingEngine, EmbeddingEngineLoadResult, EmbeddingEngineOutput, EmbeddingUnavailableEngine,
};
pub use engine_backed_worker::{
    EngineBackedWorker, ModelFactory, ModelFactoryRuntime, WorkerRuntimeError,
};
#[cfg(feature = "direct-mlx")]
pub use expert_paging::load_quantized_expert_page;
pub use expert_paging::{
    ExpertManifestError, ExpertPageRoutePartition, ExpertWeightMemoryCacheStatistics,
    ExpertWeightPage, QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
    QuantizedExpertShardManifest, QuantizedExpertSourceInterval, QuantizedExpertTensorRange,
    QuantizedTensorSource, RetainedExpertLayerCommit, RetainedExpertLayerCommitDelta,
    RetainedExpertLayerCommitError, RetainedExpertLayerCommitOutcome, RetainedExpertPageCache,
    RetainedExpertReclamation, SafetensorsDtype, SafetensorsHeader, SafetensorsHeaderError,
    TensorHeaderEntry, build_quantized_expert_page_manifest_from_plan,
    last_prefill_chunk_demand_weight, parse_safetensors_header, validate_expert_ids,
    validate_quantization_contract, validate_source_intervals, validate_virtual_intervals,
};
pub use flux2_klein::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION,
    FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT, FLUX2_KLEIN_PROVIDER_MODEL_ID,
    FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT, Flux2KleinArtifactError, Flux2KleinArtifactProvenance,
    Flux2KleinArtifactValidator, Flux2KleinComponentLoad, Flux2KleinConfigError,
    Flux2KleinDimensionError, Flux2KleinEngineComponents, Flux2KleinFlowSchedule,
    Flux2KleinFlowScheduler, Flux2KleinFlowSchedulerError, Flux2KleinFlowStep,
    Flux2KleinImageDimensions, Flux2KleinImageEncodingError, Flux2KleinImageEngine,
    Flux2KleinLicense, Flux2KleinMemoryAdmission, Flux2KleinMemoryAdmissionError,
    Flux2KleinMemoryGeometry, Flux2KleinOfficialProfile, Flux2KleinPackedLatentLayout,
    Flux2KleinPipelineConfig, Flux2KleinPngEncoder, Flux2KleinResidencyMode,
    Flux2KleinResidencyPlan, Flux2KleinRetainedArtifactFiles, Flux2KleinSchedulerConfig,
    Flux2KleinTensorDescriptor, Flux2KleinTensorInventory, Flux2KleinTextEncoderConfig,
    Flux2KleinTransformerConfig, Flux2KleinVaeConfig, Flux2KleinVaeError, Flux2KleinVaeTile,
    Flux2KleinVaeTilePlan, Flux2KleinVaeTilingConfig, ValidatedFlux2KleinArtifact,
    flux2_klein_inverse_batch_norm_reference, flux2_klein_reference_rgb_u8,
};
#[cfg(feature = "direct-mlx")]
pub use flux2_klein::{
    Flux2KleinBlockGroupEvent, Flux2KleinBlockKind, Flux2KleinComponentOracle,
    Flux2KleinTransformer, Flux2KleinTransformerError, Flux2KleinTransformerGeometry,
    Flux2KleinTransformerGeometryError, Flux2KleinTransformerInputs, Flux2KleinTransformerOutput,
    Flux2KleinTransformerWeights, Flux2KleinVaeDecodeMode, Flux2KleinVaeDecoder,
    apply_rope_for_component_oracle, flux2_klein_euler_update_for_tests,
    flux2_klein_initial_latents_for_tests, flux2_klein_keyed_noise_and_euler_for_tests,
};
#[doc(hidden)]
#[cfg(feature = "direct-mlx")]
pub use gpu_token_sampling::{
    sample_acceptance_coins_for_tests, sample_from_relative_probabilities_for_tests,
};
pub use image_generation_engine::{
    ImageGenerationEngine, ImageGenerationEngineLoadResult, ImageGenerationEngineStep,
    ImageGenerationUnavailableEngine,
};
pub use inference_engine::{
    EngineGenerationStart, EngineLoadResult, ExpertResidencyTelemetry, GeneratedToken,
    GenerationFinalization, InferenceEngine, InferenceEngineError, MlxInferenceEngine,
    MlxInferenceExecution, PreparedInferenceRequest,
};
pub use k2_horizon_mova::K2HorizonMoVAServingSettings;
#[cfg(feature = "direct-mlx")]
pub use k2_horizon_mova::{
    FusedExpertDecodeKernels, K2HorizonMoVAAffineLinear, K2HorizonMoVAEngine,
    K2HorizonMoVAInferenceExecution, K2HorizonMoVAKvState, K2HorizonMoVAStartupError,
    gathered_fused_swiglu, gathered_value_experts, initialize_k2_horizon_mova_execution,
    initialize_k2_horizon_mova_execution_with_serving_settings, initialize_k2_horizon_mova_model,
    initialize_k2_horizon_mova_model_with_serving_settings,
};
pub use k2_horizon_mova::{
    K2HorizonMoVAAffineProfile, K2HorizonMoVAArtifactValidationError,
    K2HorizonMoVAArtifactValidator, K2HorizonMoVAAttentionGateFunc, K2HorizonMoVAConfig,
    K2HorizonMoVAConfigError, K2HorizonMoVAExpertGeometryError, K2HorizonMoVAGenerationProcessor,
    K2HorizonMoVAInferenceRequest, K2HorizonMoVALayerKind, K2HorizonMoVAOutputParser,
    K2HorizonMoVAPromptRenderer, K2HorizonMoVAQuantizationContract, K2HorizonMoVARequestOutput,
    K2HorizonMoVAShardIndex, K2HorizonMoVASparseLayerExpertPayload,
    K2HorizonMoVAThinkingBudgetError, K2HorizonMoVAThinkingBudgetState, K2HorizonMoVATokenizer,
    K2HorizonMoVATokenizerError, K2HorizonMoVAWeightDialect, ValidatedK2HorizonMoVAArtifact,
    expected_stacked_affine_tensor_names, k2_horizon_mova_decoder_cache_layout,
    k2_horizon_mova_expert_layer_geometries, resolve_k2_horizon_mova_thinking_budget,
};
#[cfg(feature = "direct-mlx")]
pub use kernel_capability::SortedExpertWeightedSumProbe;
#[cfg(feature = "direct-mlx")]
pub use kernel_capability::install_forced_worker_verdicts_for_tests;
#[cfg(feature = "direct-mlx")]
pub use kernel_capability::worker_process_kernel_capabilities;
pub use kernel_capability::{
    CustomKernelVerdict, CustomMetalKernelFamily, CustomMetalKernelProbe, KernelCapabilityError,
    KernelUnsupportedReason, WorkerKernelCapabilities, validate_probe_outputs,
};
pub use memory::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthProjection, AdaptiveRamGrowthTransientReserveSource,
    AllocationAdmissionDecision, AllocationAdmissionObservation,
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, CompleteResidencyDecision,
    CompleteResidencyHeadroomBoundary, CompleteResidencyRequirements, ContextAdmissionRequirements,
    CurrentExpertLayerResidency, DecodeExpertCache, ExpertLayerGeometry,
    ExpertLayerResidencyTarget, ExpertMemoryAdmissionError, ExpertReclamationPlan,
    ExpertResidencyPlan, ExpertResidencyPlanError, ForwardRecoveryDecision, ForwardRecoveryPolicy,
    ForwardRecoveryRequirements, MeasuredExpertLayerPayload, MemoryAdmissionDecision,
    MemoryBoundary, MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements,
    MemoryCeilingUtilization, MemoryPhase, MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment,
    MlxMemoryTelemetry, MlxRamBudget, MlxRamBudgetError, MlxRamBudgetMeasurement,
    MlxRamBudgetModelGeometry, MlxRamBudgetSnapshot, PagedExpertReclamationStep,
    PreviousTokenPrefetchCandidate, PreviousTokenPrefetchLayerCapacity, PreviousTokenPrefetchPlan,
    RamBudgetGeometryError, RequestExpertLayerRole, RequestExpertResidency, ResidentExpertWeight,
    RetainedExpertPageClass, RotatingAdmissionError, classify_expert_memory_mode,
    combined_persistent_growth_bytes, complete_layer_indexes_required_before_decode,
    complete_residency_exceeds_ceiling_with_activation_headroom,
    expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure, hot_expert_warm_slot_count,
    measured_non_expert_forward_growth_bytes,
    measured_non_expert_forward_growth_bytes_excluding_expert_page_streaming,
    next_paged_expert_reclamation_step, persistent_context_restore_workspace_bytes,
    plan_expert_residency, plan_previous_token_prefetch,
    projected_active_memory_after_complete_expert_replacement,
    publish_request_stable_residency_plan, request_context_temporary_workspace_bytes,
    required_complete_residency_activation_headroom_bytes,
    retained_complete_layer_ceiling_after_prefill_budget_refresh,
    retained_expert_payload_capacity_bytes, retained_resident_ceiling_after_budget_refresh,
    rotating_committed_token_count, rotating_prefill_transient_token_count,
    safe_minimum_active_memory_ceiling_bytes,
    seated_complete_expert_request_peak_active_memory_bytes,
    seated_complete_expert_request_temporary_workspace_bytes,
    should_commit_mandatory_complete_layer, should_commit_mandatory_routed_page,
    should_enact_planned_expert_release, should_retry_fixed_forward_after_expert_reclamation,
};
#[cfg(feature = "direct-mlx")]
pub use memory::{MlxAllocationAdmission, MlxAllocationAdmissionError};
#[cfg(feature = "direct-mlx")]
pub use model_family_runtime::ModelFamilyInferenceEngine;
pub use model_family_runtime::{
    ModelFamilyGenerationProcessor, ModelFamilyImageEngine, ModelFamilyInferenceRequest,
    ModelFamilyRequestOutput,
};
pub use model_generation_processor::{
    MalformedModelOutputDiagnostic, ModelGeneratedTokenTranslation, ModelGenerationOutputError,
    ModelGenerationProcessor, PreparedModelGeneration,
};
#[cfg(feature = "direct-mlx")]
pub use modernbert::{
    EncodedEmbeddingInput, ModernBertConfiguration, ModernBertEmbeddingEngine,
    encode_embedding_input,
};
pub use performance_attribution::macos_process_io::{
    MacosProcessIoDelta, MacosProcessIoError, MacosProcessIoSnapshot, sample_current_process_io,
    sample_process_io_for_process_id,
};
pub use performance_attribution::{
    GenerationPerformanceAttributionMetadata, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionLog, PerformanceAttributionOutcome,
    PerformanceAttributionReport, PerformanceCounter, PerformanceOperation,
    PerformanceOperationMeasurement, process_io_delta_between_samples,
};
pub use persistent_cache::{
    PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION, PersistentPromptCacheBlockCausalInput,
    PersistentPromptCacheBlockError, PersistentPromptCacheBlockHeader,
    PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError,
    PersistentPromptCacheCounters, PersistentPromptCacheLookupDiagnostics,
    PersistentPromptCacheMissReason, PersistentPromptCacheModelContract,
    PersistentPromptCacheModelContractError, PersistentPromptCachePrefixLookup,
    PersistentPromptCachePrefixLookupResult, PersistentVisualEmbeddingFileError,
    PersistentVisualEmbeddingFileHeader, PersistentVisualEmbeddingKey,
    PersistentVisualEmbeddingModelContract,
    persistent_prompt_cache_boundary_clamped_prefill_chunk_end,
    persistent_prompt_cache_boundary_completed_prefill_chunk_tokens,
};
#[cfg(feature = "direct-mlx")]
pub use persistent_cache::{
    PersistentPromptCacheClearOutcome, PersistentPromptCacheDiskStore,
    PersistentPromptCacheDiskStoreConfig, PersistentPromptCacheDiskStoreError,
    PersistentPromptCachePublicationOutcome, PersistentPromptCacheStartupCleanupCategory,
    PersistentPromptCacheStartupCleanupEvidence, build_persistent_prompt_cache_stats_event,
    clear_persistent_prompt_cache_directory,
};
pub use qwen3_5::{
    ModelWeightStorage, OptiQMetadata, OptiQMetadataError, OptiQQuantizationProfile,
    Qwen3_5ArtifactError, Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator, Qwen3_5Config,
    Qwen3_5ConfigError, Qwen3_5DecoderLayerCacheDtypes, Qwen3_5FeedForwardArchitecture,
    Qwen3_5GenerationProcessor, Qwen3_5ImageDimensions, Qwen3_5ImageGrid,
    Qwen3_5ImageProcessingError, Qwen3_5ImageProcessor, Qwen3_5InferenceRequest,
    Qwen3_5OutputEvent, Qwen3_5OutputParser, Qwen3_5OutputParserError, Qwen3_5ProcessedImage,
    Qwen3_5PromptError, Qwen3_5PromptRenderer, Qwen3_5RamBudgetGeometryError,
    Qwen3_5RenderedPrompt, Qwen3_5RequestOutput, Qwen3_5RequestOutputError, Qwen3_5SamplerConfig,
    Qwen3_5SamplingStrategy, Qwen3_5ShardIndex, Qwen3_5ThinkingBudgetError,
    Qwen3_5ThinkingBudgetState, Qwen3_5TokenDecoder, Qwen3_5TokenIds, Qwen3_5Tokenizer,
    Qwen3_5TokenizerError, Qwen3_5ToolCall, Qwen3_5VisionConfig, Qwen3_5VisionInputPlan,
    Qwen3_5VisionInputPlanError, Qwen3_5VisualEmbeddingRequiredImage,
    Qwen3_5VisualEmbeddingSuffixPlan, Qwen3_5VisualEmbeddingSuffixPlanError,
    Qwen3_5VisualPromptCacheIdentityPlan, Qwen3_5VisualPromptCacheIdentityPlanError,
    ValidatedQwen3_5Artifact, discover_sampler_config, discover_token_ids,
    mlx_ram_budget_model_geometry_from_validated_artifact, plan_qwen3_5_visual_embedding_suffix,
    plan_qwen3_5_visual_prompt_cache_block_inputs, qwen3_5_decoder_cache_layout,
    qwen3_5_language_tensor_profiles, qwen3_5_request_enables_thinking,
    qwen3_5_resident_language_tensor_profiles, qwen3_5_vision_tensor_profiles,
    resolve_sampling_seed, translate_qwen3_5_preparation_error, translate_request_output_error,
    validate_context_token_count,
};

// Qwen-Image-2.1: the family's pure contracts, artifact validation, and geometry constants. The
// MLX components (text encoder, transformer, VAE decoder, pipeline) follow below under `direct-mlx`.
pub use qwen_image_21::{
    CacheMode, FlowMatchSchedule, FlowMatchSchedulerParams, NUM_TRAIN_TIMESTEPS,
    PromptConditioningOutput, QWEN_IMAGE_21_LATENT_CHANNEL_COUNT, QWEN_IMAGE_21_LICENSE_IDENTIFIER,
    QWEN_IMAGE_21_OFFICIAL_MODEL_ID, QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT,
    QWEN_IMAGE_21_PROVIDER_MODEL_ID, QWEN_IMAGE_21_SPATIAL_COMPRESSION_RATIO,
    QWEN_IMAGE_21_SYS_PROMPT, QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH, QWEN_IMAGE_21_VAE_SCALE_FACTOR,
    QwenImage21ArtifactError, QwenImage21ArtifactProvenance, QwenImage21ArtifactValidator,
    QwenImage21ConfigError, QwenImage21License, QwenImage21PipelineConfig,
    QwenImage21RetainedArtifactFiles, QwenImage21Rope, QwenImage21SchedulerConfig,
    QwenImage21TensorDescriptor, QwenImage21TensorInventory, QwenImage21TensorProfile,
    QwenImage21TextEncoderConfig, QwenImage21TransformerConfig, QwenImage21VaeConfig,
    QwenImage21VaeError, SinusoidalTimesteps, TextConditioningError, VAE_SPATIAL_MULTIPLE,
    ValidatedQwenImage21Artifact, build_block_causal_mask, build_image_ids,
    build_prompt_conditioning, build_schedule, build_target_token_mask, cache_is_valid,
    cache_query_slice, cache_write_slice, calculate_dimensions, calculate_shift,
    causal_modulation_row_map, decoded_pixel_dimensions, default_shift, euler_step,
    frequencies_for_tests, latent_spatial_dimensions, normalize_empty_prompt, pack_latents_seq_len,
    prefix_length, prefix_segments, quantized_group_count, quantized_row_count,
    render_t2i_prompt_template, render_ti2i_prompt_template,
    render_ti2i_prompt_template_with_image_count, resolve_generation_dimensions,
    round_half_to_even, round_to_nearest_multiple, special_tokens, text_encoder_tensor_profiles,
    transformer_tensor_profiles, unpack_spatial_dims, vae_tensor_profiles, zero_center_rms_norm,
};
pub use qwen_image_21::{
    QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES, QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
    QwenImage21ComponentLoad, QwenImage21EngineComponents, QwenImage21EngineError,
    QwenImage21EngineRequest, QwenImage21ImageEngine, QwenImage21RenderAdvance,
    QwenImage21RenderRequest, QwenImage21Rendered, qwen_image_21_image_generation_capabilities,
    qwen_image_21_official_model_id, validate_official_request,
};
#[cfg(feature = "direct-mlx")]
pub use qwen_image_21::{
    QwenImage21MlxComponents, QwenImage21Pipeline, QwenImage21TextEncoder, QwenImage21Transformer,
    QwenImage21TransformerRequest, QwenImage21VaeDecoder,
};
#[cfg(feature = "direct-mlx")]
pub use qwen3_5::{
    Qwen3_5Engine, Qwen3_5ExecutionError, Qwen3_5GatedDeltaBoundaryCheckpointResult, Qwen3_5Model,
    Qwen3_5ModelChunkingConfiguration, Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector, Qwen3_5PrefillExecutionContext,
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5PromptProcessingChunkSizerError,
    Qwen3_5TargetForwardOutput, Qwen3_5VisionModel, Qwen3_5VisionPaddingZeroCache,
    Qwen3_5VisionWeights, Qwen3_5Weights, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
    is_gdn_decode_prework_eligible, persistent_prompt_cache_publication_advances_parent_chain,
    qwen3_5_apply_top_p_mask, qwen3_5_full_attention_step, qwen3_5_gated_delta_checkpoint_kernel,
    qwen3_5_gated_delta_kernel, qwen3_5_gated_delta_sequence,
    qwen3_5_gated_delta_sequence_ops_fallback,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints_ops_fallback, qwen3_5_gated_delta_step,
    qwen3_5_gdn_decode_prework, qwen3_5_gdn_decode_prework_kernel,
    qwen3_5_inject_visual_embeddings, safe_minimum_mlx_memory_ceiling_bytes,
};
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use qwen3_5_moe::maximum_resident_gate_up_fusion_transient_payload_bytes;
#[cfg(feature = "direct-mlx")]
pub use qwen3_5_moe::{
    ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5MoECachedPlusStreamedPageRoute,
    Qwen3_5MoEPagedPrefillExecutionMode, build_source_manifests, contiguous_selected_runs,
    qwen3_5_moe_combine_experts, qwen3_5_moe_combine_partial_route_outputs_for_tests,
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel, qwen3_5_moe_unsorted_expert_weighted_sum,
};
pub use qwen3_5_moe::{
    LayerRoutedExpertIds, ObservedExpertRoute, RouteObservationRecord, RouteObservationRing,
    sorted_unique_layer_routed_expert_ids,
};
pub use qwen3_5_moe::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    build_quantized_expert_layer_plan,
};
#[cfg(feature = "direct-mlx")]
pub use qwen3_5_moe::{
    ResidentLayerArraysForTests, ResidentProjectionArraysForTests, resident_layer_arrays_for_tests,
};
pub use sparse_experts::should_use_sorted_expert_reduction;
#[cfg(feature = "direct-mlx")]
pub use sparse_experts::{
    ExpertAssignmentOrder, SortedExpertAssignments, StackedExpertProjection,
    gather_expert_projection, restore_expert_assignment_order, router_weighted_expert_inputs,
    sort_expert_assignments, sorted_expert_weighted_sum, sorted_expert_weighted_sum_kernel,
    unsorted_expert_weighted_sum,
};
pub use sparse_experts::{SparseExpertError, invert_assignment_order};

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

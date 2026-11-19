// Unsafe is denied workspace-wide, and this crate has none of its own. The one
// exception, the macOS accounting syscall behind performance attribution, moved
// with that module into astronomical-gpu-building-blocks (issue #2340).

// Artifact validation is shared by every model family and lives in its own
// crate (issue #2340); the alias keeps `crate::artifact_validation` paths.
pub(crate) use astronomical_artifact_validation as artifact_validation;

// The GPU building blocks every model family shares live in their own crate
// (issue #2340), so a change to one family no longer rebuilds them and they
// cannot reach back into a family. These aliases keep each `crate::<module>`
// path in this crate resolving without touching its many consumers.
use astronomical_gpu_building_blocks::attention;

mod embedding_engine;

mod engine_backed_worker;

mod expert_paging;

#[cfg(feature = "direct-mlx")]
use astronomical_gpu_building_blocks::gpu_token_sampling;

mod kernel_capability;

mod mlx_inference_engine;

mod model_family_runtime;

pub(crate) mod decision_engine;

// Speech lives in its own crate (issue #2340); the aliases keep each
// `crate::speech_*` path in this crate resolving.
pub(crate) use astronomical_speech::speech_engine;

#[cfg(all(test, feature = "direct-mlx"))]
pub(crate) mod synthetic_weights_test_support;

#[cfg(feature = "direct-mlx")]
pub(crate) mod decision_models;

#[cfg(feature = "direct-mlx")]
pub(crate) use astronomical_gpu_building_blocks::linear_adapters;

#[cfg(feature = "direct-mlx")]
pub(crate) mod modernbert;

#[cfg(feature = "direct-mlx")]
pub(crate) use astronomical_speech::speech_models;

pub(crate) use astronomical_gpu_building_blocks::performance_attribution;

pub(crate) mod astro_ssd_prompt_cache;

// The folded ternary family ships its loader and straight text model as
// crate API ahead of the serving-path integration; the acceptance crate
// exercises it through this public surface.

pub mod prism_hadamard_qwen35;

// The jangtq2 family (codebook format, Hadamard rotation, Metal page
// decoder, fused gathered projection kernel) lives in its own workspace
// crate, `astronomical-jangtq2-qwen35` (issue #1427): a family change
// compiles and tests without rebuilding the rest of this crate. The
// streaming integration (page loading, slot writes, dispatch) stays here
// and consumes the family crate's public types.

pub(crate) mod qwen3_5_core;

// The sensitivity probe adapter for the quantization crate (the OptiQ port):
// swaps one Qwen3.5 module at a time between its bf16 weight and a
// simulate-quantized copy and reports KL divergence, using this crate's
// resident model and weights.
#[cfg(feature = "direct-mlx")]
pub mod quantization_probe;

#[cfg(feature = "direct-mlx")]
mod qwen3_5_resident;

mod qwen3_5_streaming;

// The engine core (the inference contract and its errors, MLX memory, the
// astro_decoder_state_cache, the generation processor and structured
// generation) sits below the model families in its own crate (issue #2340).
// Its public surface, modules included, is re-exported whole, so every
// `crate::memory`, `crate::inference_engine` and root path here resolves.
pub use astronomical_engine_core::*;
pub use mlx_inference_engine::MlxInferenceEngine;

// Image generation lives in its own crate (issue #2340). Its public surface,
// the `flux2_klein`, `qwen_image_21` and `image_generation_engine` modules
// included, is re-exported whole, so the worker, this crate's own
// `crate::flux2_klein` paths and the acceptance tests keep resolving.
pub use astronomical_image_generation::*;

// The safetensors header parsing lives in the pure astronomical-memory-policy
// crate; the module bridge keeps every `crate::safetensors::...` path in the
// serving crate resolving without touching its many consumers.
pub(crate) use astronomical_memory_policy::safetensors;

pub use astronomical_gpu_building_blocks::row_batched_matmul;

pub mod speculative_decoding;

use astronomical_gpu_building_blocks::sampling_seed;

use astronomical_gpu_building_blocks::sparse_experts;

#[doc(hidden)]
pub use artifact_validation::validate_required_file_for_tests;

pub use artifact_validation::{ArtifactValidationError, TensorDtype, TensorProfile};

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

pub use decision_engine::{
    DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput, DecisionTrainingStep,
    DecisionUnavailableEngine,
};

#[cfg(feature = "direct-mlx")]
pub use decision_models::clef::ClefDecisionEngine;
#[cfg(feature = "direct-mlx")]
pub use decision_models::kev::{KevDecisionEngine, KevRowLayout};
#[cfg(feature = "direct-mlx")]
pub use decision_models::laya::LayaDecisionEngine;
#[cfg(feature = "direct-mlx")]
pub use decision_models::recorded_training::RecordedTrainingEngine;

pub use embedding_engine::{
    EmbeddingEngine, EmbeddingEngineLoadResult, EmbeddingEngineOutput, EmbeddingUnavailableEngine,
};

pub use engine_backed_worker::{
    EngineBackedWorker, ModelFactory, ModelFactoryRuntime, WorkerRuntimeError,
};

#[cfg(feature = "direct-mlx")]
pub use expert_paging::load_quantized_expert_page;

pub use expert_paging::{
    AstroExpertCacheStatistics, ExpertManifestError, ExpertPageRoutePartition, ExpertWeightPage,
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
    QuantizedExpertShardManifest, QuantizedExpertSourceInterval, QuantizedExpertTensorRange,
    QuantizedTensorSource, RetainedExpertReclamation, SafetensorsDtype, SafetensorsHeader,
    SafetensorsHeaderError, TensorHeaderEntry, build_quantized_expert_page_manifest_from_plan,
    last_prefill_chunk_demand_weight, parse_safetensors_header, validate_expert_ids,
    validate_quantization_contract, validate_source_intervals, validate_virtual_intervals,
};

#[doc(hidden)]
#[cfg(feature = "direct-mlx")]
pub use gpu_token_sampling::{
    build_sampled_token_rows_for_tests, sample_acceptance_coins_for_tests,
    sample_from_relative_probabilities_for_tests,
};

#[cfg(feature = "direct-mlx")]
pub use prism_hadamard_qwen35::engine::{
    PrismHadamardQwen35Engine, PrismHadamardQwen35InferenceExecution,
    PrismHadamardQwen35ServingSettings, PrismHadamardQwen35StartupError,
    initialize_prism_hadamard_qwen35_model_with_serving_settings,
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

#[cfg(feature = "direct-mlx")]
pub use model_family_runtime::ModelFamilyInferenceEngine;

pub use model_family_runtime::{
    ModelFamilyGenerationProcessor, ModelFamilyImageEngine, ModelFamilyInferenceRequest,
    ModelFamilyRequestOutput,
};

#[cfg(feature = "direct-mlx")]
pub use modernbert::{EncodedEmbeddingInput, ModernBertEmbeddingEngine, encode_embedding_input};

#[cfg(feature = "direct-mlx")]
pub use speech_models::kokoro::{KokoroSpeechEngine, KokoroSpeechTrace};

#[cfg(feature = "direct-mlx")]
pub use speech_models::phonon::PhononSpeechEngine;

pub use speech_engine::{
    SpeechEngine, SpeechEngineLoadResult, SpeechEngineStep, SpeechUnavailableEngine, StartedSpeech,
};

pub use performance_attribution::macos_process_io::{
    MacosProcessIoDelta, sample_current_process_io, sample_process_io_for_process_id,
};

pub use performance_attribution::{
    GenerationPerformanceAttributionMetadata, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttribution, PerformanceAttributionLog, PerformanceAttributionOutcome,
    PerformanceAttributionReport, PerformanceCounter, PerformanceOperation,
    PerformanceOperationMeasurement, process_io_delta_between_samples,
};

pub use astro_ssd_prompt_cache::{
    AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheBlockKeyError,
    AstroSsdPromptCacheLookupDiagnostics, AstroSsdPromptCacheMissReason,
    AstroSsdPromptCacheModelContract, AstroSsdPromptCacheModelContractError,
    AstroSsdPromptCachePrefixLookupResult, PersistentVisualEmbeddingFileError,
    PersistentVisualEmbeddingKey, PersistentVisualEmbeddingModelContract,
    astro_ssd_prompt_cache_boundary_clamped_prefill_chunk_end,
    astro_ssd_prompt_cache_boundary_completed_prefill_chunk_tokens,
    astro_ssd_prompt_cache_clamped_prefill_chunk_end,
    astro_ssd_prompt_cache_partial_tail_capture_end,
};

#[cfg(feature = "direct-mlx")]
pub use astro_ssd_prompt_cache::{
    AstroSsdPromptCacheClearOutcome, AstroSsdPromptCacheDiskStore,
    AstroSsdPromptCacheDiskStoreConfig, AstroSsdPromptCacheDiskStoreError,
    AstroSsdPromptCachePublicationOutcome, AstroSsdPromptCacheStartupCleanupCategory,
    AstroSsdPromptCacheStartupCleanupEvidence, build_astro_ssd_prompt_cache_stats_event,
    clear_astro_ssd_prompt_cache_directory,
};

pub use qwen3_5_core::artifacts::{
    Qwen3_5ArtifactError, Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator,
    Qwen3_5MemoryGeometryError, Qwen3_5ShardIndex, ValidatedQwen3_5Artifact,
    expert_layer_plans_for_validated_artifact, model_memory_geometry_from_validated_artifact,
    qwen3_5_language_tensor_profiles,
};

// Restored public surface: these items are the API the crate's own integration

// tests consume (they verify the public seam); everything else that only the

// beside-source unit tests exercised stays crate-private.

pub use crate::artifact_validation::tensor_inventory::{
    TensorDeclarationOrigin, TensorInventory, TensorInventoryError, TensorLocation,
    TensorSemanticRole, TensorSourceId,
};

pub use crate::artifact_validation::types::RequiredFileProfile;

pub use crate::artifact_validation::validated_artifact::ValidatedWeightsFile;

pub use crate::performance_attribution::macos_process_io::MacosProcessIoError;

pub use crate::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;

pub use crate::astro_ssd_prompt_cache::block_format::AstroSsdPromptCacheBlockHeader;

pub use crate::astro_ssd_prompt_cache::block_format_error::AstroSsdPromptCacheBlockError;

pub use crate::astro_ssd_prompt_cache::prefix_lookup::AstroSsdPromptCachePrefixLookup;

pub use crate::astro_ssd_prompt_cache::visual_embedding_format::PersistentVisualEmbeddingFileHeader;

pub use crate::astro_ssd_prompt_cache::visual_embedding_key::PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION;

pub use qwen3_5_core::configuration::{
    Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture,
};
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use qwen3_5_core::model::tree_attention_measurement::TreeAttentionErrorCount;
/// The row-exact draft-tree attention kernel and the row chains it reads, for the
/// acceptance journeys that hold it to the one-row step and time it.
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use qwen3_5_core::model_math::gated_delta_tree::{TreeRowChains, UploadedTreeRowChains};
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use qwen3_5_core::model_math::row_exact_tree_attention::{
    RowExactTreeAttentionInputs, RowExactTreeAttentionKernels, row_exact_tree_attention,
};

pub use qwen3_5_core::decoder::{
    Qwen3_5AstroDecoderStateCacheLayerDtypes, qwen3_5_astro_decoder_state_cache_layout,
};

pub use qwen3_5_core::quantizations::producers::optiq::{
    OptiQMetadata, OptiQMetadataError, resolve_optiq_metadata_relative_path,
};
pub use qwen3_5_core::quantizations::producers::oq::{OqImatrixProvenance, OqImatrixReportError};
pub use qwen3_5_core::quantizations::producers::{
    QuantizationProducer, detect_quantization_producer, detect_quantization_producer_in_directory,
};

#[cfg(any(test, feature = "direct-mlx"))]
pub use qwen3_5_core::text::thinking_budget::Qwen3_5ThinkingBudgetState;

pub use qwen3_5_core::text::{
    Qwen3_5GenerationProcessor, Qwen3_5InferenceRequest, Qwen3_5OutputEvent,
    Qwen3_5OutputParserError, Qwen3_5PromptError, Qwen3_5PromptRenderer, Qwen3_5RenderedPrompt,
    Qwen3_5RequestOutput, Qwen3_5SamplerConfig, Qwen3_5SamplingStrategy, Qwen3_5TokenDecoder,
    Qwen3_5TokenIds, Qwen3_5Tokenizer, Qwen3_5TokenizerError, discover_sampler_config,
    discover_token_ids, qwen3_5_request_enables_thinking, resolve_sampling_seed,
    translate_qwen3_5_preparation_error, translate_request_output_error,
    validate_context_token_count,
};

pub use qwen3_5_core::vision::{
    Qwen3_5ImageProcessor, Qwen3_5ProcessedImage, Qwen3_5VisionConfig,
    Qwen3_5VisualAstroSsdPromptCacheIdentityPlan,
    Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError, Qwen3_5VisualEmbeddingSuffixPlan,
    Qwen3_5VisualEmbeddingSuffixPlanError, plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs,
    plan_qwen3_5_visual_embedding_suffix, qwen3_5_vision_tensor_profiles,
};

#[cfg(any(test, feature = "direct-mlx"))]
pub use qwen3_5_core::vision::vision_input_plan::Qwen3_5VisionInputPlanError;

// Qwen-Image-2.1: the family's pure contracts, artifact validation, and geometry constants. The

// MLX components (text encoder, transformer, VAE decoder, pipeline) follow below under `direct-mlx`.

pub use qwen3_5_core::artifacts::{
    build_quantized_expert_layer_plan, build_quantized_expert_layer_plans,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::decoder::{
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpoint,
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model::differentiable_attention::Qwen3_5AttentionKernel;
#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model::full_attention::qwen3_5_full_attention_step;

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model_math::gated_delta_boundary_checkpoints::{
    Qwen3_5GatedDeltaBoundaryCheckpointResult, qwen3_5_gated_delta_checkpoint_kernel,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints_ops_fallback,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model_math::gated_delta_sequence::{
    qwen3_5_gated_delta_kernel, qwen3_5_gated_delta_sequence,
    qwen3_5_gated_delta_sequence_ops_fallback,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model_math::gated_delta_step::qwen3_5_gated_delta_step;

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model_math::gdn_decode_prework_kernel::{
    is_gdn_decode_prework_eligible, qwen3_5_gdn_decode_prework, qwen3_5_gdn_decode_prework_kernel,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model_math::weights::Qwen3_5Weights;

pub use astronomical_gpu_building_blocks::route_observation::{
    LayerRoutedExpertIds, ObservedExpertRoute,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::text::qwen3_5_apply_repetition_penalty;

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::text::qwen3_5_apply_top_p_mask;

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::vision::{
    Qwen3_5VisionModel, Qwen3_5VisionPaddingZeroCache, Qwen3_5VisionWeights,
    qwen3_5_inject_visual_embeddings,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_resident::{
    Qwen3_5ResidentEngine, Qwen3_5ResidentModel, Qwen3_5ResidentPromptProcessingChunkSizer,
    Qwen3_5ResidentPromptProcessingChunkSizerError,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_resident::{
    Qwen3_5DraftTreeVerification, Qwen3_5DraftedRowsVerification,
    Qwen3_5SpeculativeVerificationOutput, VerificationLogitsMode, VerificationTokenChoice,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_resident::{
    ResidentLayerArraysForTests, ResidentProjectionArraysForTests, resident_layer_arrays_for_tests,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_streaming::{
    ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5MoEAstroExpertCachePlusStreamedPageRoute,
    Qwen3_5MoEPagedExecutionMode, build_source_manifests, contiguous_selected_runs,
    qwen3_5_moe_combine_experts, qwen3_5_moe_combine_partial_route_outputs_for_tests,
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel, qwen3_5_moe_unsorted_expert_weighted_sum,
};

pub use qwen3_5_streaming::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_streaming::{
    DirectExpertFetchCounts, LayerProbeShape, Qwen3_5StreamingEngine, Qwen3_5StreamingModel,
    Qwen3_5StreamingPromptProcessingChunkSizer, Qwen3_5StreamingPromptProcessingChunkSizerError,
    Qwen3_5StreamingTargetForwardOutput, RoutedReplyRecording,
    safe_minimum_mlx_memory_ceiling_bytes,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::astro_ssd_prompt_cache_capture::astro_ssd_prompt_cache_publication_advances_parent_chain;

pub use qwen3_5_streaming::{
    RouteObservationRecord, RouteObservationRing, sorted_unique_layer_routed_expert_ids,
};

#[cfg(feature = "direct-mlx")]
pub use sparse_experts::{
    ExpertAssignmentOrder, SortedExpertAssignments, StackedExpertProjection,
    expert_assignment_order, gather_expert_projection, restore_expert_assignment_order,
    router_weighted_expert_inputs, sort_expert_assignments, sorted_expert_weighted_sum,
    sorted_expert_weighted_sum_kernel, unsorted_expert_weighted_sum,
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

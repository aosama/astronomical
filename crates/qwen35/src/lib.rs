//! The Qwen 3.5 model family (issue #2340): the shared core (configuration,
//! tokenizer, decoder, model math, artifacts and vision), the resident and
//! streaming engines, the Prism Hadamard ternary build, speculative decoding,
//! expert paging, the custom kernel capability probes, the
//! astro_ssd_prompt_cache the family's engines publish to, the owner-thread
//! engine they run on. The quantizer's sensitivity probe for this family lives
//! in `qwen35-quantization-probe`, so this crate never compiles
//! the quantizer.
//!
//! It sits above the engine core, whose contract it implements, and below the
//! decision models (Kev and Clef score on a Qwen 3.5 backbone) and the
//! serving crate's worker. The edge to the serving crate is one way, which
//! `cargo xtask check-architecture` enforces. The serving crate re-exports
//! this crate's root with a glob, so every name at this root stays unique
//! across the crates it re-exports.

pub mod astro_ssd_prompt_cache;

pub mod expert_paging;

pub mod kernel_capability;

pub mod mlx_inference_engine;

// The folded ternary family ships its loader and straight text model as
// crate API ahead of the serving-path integration; the acceptance crate
// exercises it through this public surface.
pub mod prism_hadamard_qwen35;

// The jangtq2 weight family lives in its own crate,
// `jangtq2-qwen35` (issue #1427); the streaming integration
// (page loading, slot writes, dispatch) stays here and consumes its types.
pub mod qwen3_5_core;

// Source-layout contract: the Qwen 3.5 modules import each other only
// through the shared core.
#[cfg(test)]
mod qwen3_5_import_direction_tests;

#[cfg(feature = "direct-mlx")]
pub mod qwen3_5_resident;

pub mod qwen3_5_streaming;

pub mod speculative_decoding;

// The crates below this family, imported at the root so the family's
// `crate::...` paths into them read as they did inside the serving crate.
pub(crate) use ::artifact_validation;
use artifact_validation::*;
use engine_core::*;
use memory_policy::safetensors;
use model_primitives::*;

pub use mlx_inference_engine::MlxInferenceEngine;

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

pub use astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;

pub use astro_ssd_prompt_cache::block_format::AstroSsdPromptCacheBlockHeader;

pub use astro_ssd_prompt_cache::block_format_error::AstroSsdPromptCacheBlockError;

pub use astro_ssd_prompt_cache::prefix_lookup::AstroSsdPromptCachePrefixLookup;

pub use astro_ssd_prompt_cache::visual_embedding_format::PersistentVisualEmbeddingFileHeader;

pub use astro_ssd_prompt_cache::visual_embedding_key::PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION;

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
pub use qwen3_5_core::shared_execution::prompt_processing_chunk_sizer::{
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5PromptProcessingChunkSizerError,
};
#[cfg(feature = "direct-mlx")]
pub use qwen3_5_resident::{Qwen3_5ResidentEngine, Qwen3_5ResidentModel};

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
    Qwen3_5StreamingTargetForwardOutput, RoutedReplyRecording,
    safe_minimum_mlx_memory_ceiling_bytes,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::astro_ssd_prompt_cache_capture::astro_ssd_prompt_cache_publication_advances_parent_chain;

pub use qwen3_5_streaming::{
    RouteObservationRecord, RouteObservationRing, sorted_unique_layer_routed_expert_ids,
};

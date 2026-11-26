//! The Qwen 3.5 model family (issue #2340): the shared core (configuration,
//! tokenizer, decoder, model math, artifacts and vision), the streaming model
//! that serves every text model, the in-memory model the quantization
//! sensitivity probe swaps weights in, the Prism Hadamard ternary build,
//! speculative decoding, the custom kernel capability probes, and the
//! owner-thread engine they run on. The `astro_ssd_prompt_cache` and expert paging they use live in the
//! engine core. The quantizer's sensitivity probe for this family lives
//! in `qwen35-quantization-probe`, so this crate never compiles
//! the quantizer.
//!
//! It sits above the engine core, whose contract it implements, and below the
//! decision models (Kev and Clef score on a Qwen 3.5 backbone) and the
//! serving crate's worker. The edge to the serving crate is one way, which
//! `cargo xtask check-architecture` enforces. The serving crate re-exports
//! this crate's root with a glob, so every name at this root stays unique
//! across the crates it re-exports.

pub mod chat_family_construction;
pub mod kernel_capability;

// The folded ternary family ships its loader and straight text model as
// crate API ahead of the serving-path integration; the acceptance crate
// exercises it through this public surface.
pub mod prism_hadamard_qwen35;

// The jangtq2 weight family lives in its own crate,
// `jangtq2-qwen35` (issue #1427); the streaming integration
// (page loading, slot writes, dispatch) stays here and consumes its types.
pub mod qwen3_5_core;

// Qwen3.8-Flash-Next (`qwen4_exp`), served as a generation of the Qwen 3.5
// layout so it shares the streaming engine (issue #2533). It imports the
// shared core; the core never imports it.
pub mod qwen4_exp;

// Source-layout contract: the Qwen 3.5 modules import each other only
// through the shared core.
#[cfg(test)]
mod qwen3_5_import_direction_tests;

#[cfg(feature = "direct-mlx")]
pub mod qwen3_5_resident;

pub mod qwen3_5_streaming;

// The crates below this family, imported at the root so the family's
// `crate::...` paths into them read as they did inside the serving crate.
pub(crate) use ::artifact_validation;
use artifact_validation::*;
pub use chat_family_construction::*;
use engine_core::*;
use model_primitives::*;

#[cfg(feature = "direct-mlx")]
pub use engine_core::expert_paging::load_quantized_expert_page;

pub use engine_core::expert_paging::{
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

pub use qwen3_5_core::artifacts::{
    Qwen3_5ArtifactError, Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator,
    Qwen3_5MemoryGeometryError, Qwen3_5ShardIndex, ValidatedQwen3_5Artifact,
    expert_layer_plans_for_validated_artifact, model_memory_geometry_from_validated_artifact,
    qwen3_5_language_tensor_profiles,
};

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
pub use text_generation::request::thinking_budget::ThinkingBudgetState;

pub use text_generation::{TextInferenceRequest, TextSamplingStrategy};

pub use qwen3_5_core::text::{
    Qwen3_5GenerationProcessor, Qwen3_5OutputEvent, Qwen3_5OutputParserError, Qwen3_5PromptError,
    Qwen3_5PromptRenderer, Qwen3_5RenderedPrompt, Qwen3_5RequestOutput, Qwen3_5SamplerConfig,
    Qwen3_5TokenDecoder, Qwen3_5TokenIds, Qwen3_5Tokenizer, Qwen3_5TokenizerError,
    discover_sampler_config, discover_token_ids, qwen3_5_request_enables_thinking,
    resolve_sampling_seed, translate_qwen3_5_preparation_error, translate_request_output_error,
    validate_context_token_count,
};

pub use qwen3_5_core::vision::{
    ProcessedImage, Qwen3_5ImageProcessor, Qwen3_5VisionConfig,
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
pub use qwen3_5_core::decoder::request_decoder_state_from_config::request_decoder_state_from_qwen3_5_config;

#[cfg(feature = "direct-mlx")]
pub use text_generation::{
    AstroSsdPromptCacheBoundaryCheckpoint, AstroSsdPromptCacheBoundaryCheckpointCollector,
    RequestDecoderStateStack, RequestDecoderStateStackAllocationCheckpoint,
    RequestDecoderStateStackCheckpoint,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model::differentiable_attention::Qwen3_5AttentionKernel;

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model::full_attention::qwen3_5_full_attention_step;

#[cfg(feature = "direct-mlx")]
pub use text_generation::model::model_chunking_configuration::TextModelChunkingConfiguration;

#[cfg(feature = "direct-mlx")]
pub use text_generation::execution_error::TextExecutionError;

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
pub use text_generation::generation_loop::engine_construction::TextMlxEngineConstructors;
#[cfg(feature = "direct-mlx")]
pub use text_generation::generation_loop::execution_core::TextExecutionCoreSettings;
#[cfg(feature = "direct-mlx")]
pub use text_generation::generation_loop::prompt_processing_chunk_sizer::{
    PromptProcessingChunkSizer, PromptProcessingChunkSizerError,
};
#[cfg(feature = "direct-mlx")]
pub use text_generation::generation_loop::test_controls::{
    TextMlxEngineTestControls, TextModelTestControls,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model::draft_tree_verification::DraftTreeTarget;
#[cfg(feature = "direct-mlx")]
pub use qwen3_5_core::model::draft_tree_verification::Qwen3_5DraftTreeVerification;

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_resident::{
    ResidentLayerArraysForTests, ResidentProjectionArraysForTests, resident_layer_arrays_for_tests,
};

#[cfg(feature = "direct-mlx")]
pub use qwen3_5_streaming::Qwen3_5StreamingEngineTestControls;
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
    DirectExpertFetchCounts, InGraphServingCounts, LayerProbeShape, Qwen3_5StreamingEngine,
    Qwen3_5StreamingModel, Qwen3_5StreamingTargetForwardOutput, ReplyStageCounts,
    RoutedReplyRecording, StreamingEngineSettings, safe_minimum_mlx_memory_ceiling_bytes,
};

#[cfg(feature = "direct-mlx")]
pub use text_generation::astro_ssd_prompt_cache_capture::astro_ssd_prompt_cache_publication_advances_parent_chain;

pub use qwen3_5_streaming::{
    RouteObservationRecord, RouteObservationRing, sorted_unique_layer_routed_expert_ids,
};

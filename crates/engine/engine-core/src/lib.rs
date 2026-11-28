//! The engine core of Astronomical (issue #2340): the contract every
//! inference engine implements and the errors it returns, the MLX-facing
//! memory owners, the astro_decoder_state_cache, the model generation
//! processor and structured generation.
//!
//! It sits below the model families, which implement its contract, and above
//! the model primitives and the memory policy. The edge is one way: this
//! crate depends on no model family and never on
//! `model-serving`, which `cargo xtask check-architecture`
//! enforces. The serving crate re-exports this crate's root with a glob, so
//! every name at this root stays unique across the crates it re-exports.

pub mod astro_decoder_state_cache;

pub mod astro_ssd_prompt_cache;

pub mod expert_paging;

pub mod inference_engine;

pub mod kernel_capability_registry;
pub mod kernel_capability_report;

pub mod memory;

pub mod mlx_inference_engine;

pub mod model_generation_processor;

pub mod speculative_decoding;

pub mod structured_generation;

pub use astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;
pub use astro_ssd_prompt_cache::block_format::AstroSsdPromptCacheBlockHeader;
pub use astro_ssd_prompt_cache::block_format_error::AstroSsdPromptCacheBlockError;
pub use astro_ssd_prompt_cache::prefix_lookup::AstroSsdPromptCachePrefixLookup;
pub use astro_ssd_prompt_cache::visual_embedding_format::PersistentVisualEmbeddingFileHeader;
pub use astro_ssd_prompt_cache::visual_embedding_key::PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION;

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

#[cfg(feature = "direct-mlx")]
pub use astro_decoder_state_cache::{
    AstroDecoderStateCacheState, AstroDecoderStateCacheStateAllocationCheckpoint, ConvolutionState,
    ConvolutionStateBoundaryCheckpointUpdate, FullAttentionKeyValueState,
    FullAttentionKeyValueStateAllocationCheckpoint, GatedDeltaRecurrentState,
    RotatingKeyValueState, RotatingKeyValueStateAllocationCheckpoint,
};

pub use astro_decoder_state_cache::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCacheLayoutError, AstroDecoderStateCachePersistedTensorLayout,
    AstroDecoderStateCacheTensorDtype, AstroDecoderStateCacheTensorLayout,
};

pub use mlx_inference_engine::MlxInferenceEngine;

pub use inference_engine::{
    DynInferenceEngine, GeneratedToken, GenerationFinalization, InferenceEngine,
    InferenceEngineError, MlxInferenceExecution,
};

pub use memory::{
    AllocationAdmissionDecision, AllocationAdmissionObservation, CompleteResidencyDecision,
    CompleteResidencyHeadroomBoundary, CompleteResidencyRequirements, CurrentExpertLayerResidency,
    EnginePlacement, ExpertLayerState, ExpertMemoryAdmissionError, ExpertResidencyPlan,
    ExpertResidencyPlanError, MeasuredExpertLayerPayload, MemoryBoundary,
    MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements, MemoryCeilingUtilization,
    MemoryPhase, MemoryReserveSnapshot, ModelMemoryFacts, ModelMemoryFactsError,
    ModelMemoryGeometry, ModelMemoryGeometryError, RetainedExpertPageClass,
    RetainedPrefixRetentionDecision, RetainedPrefixRetentionRequirements, RotatingAdmissionError,
    StartupResidencyFacts, VISIBLE_ANSWER_RESERVE_TOKENS, bounded_output_reservation_token_count,
    classify_expert_memory_mode, complete_layers_beyond_keep_count,
    decode_warm_floor_eviction_indexes, decode_whole_layer_keep_count,
    effective_mlx_memory_ceiling_after_wired_claim, effective_mlx_memory_ceiling_bytes,
    floor_raise_limit_bytes, floor_raised_ceiling_bytes,
    geometry_safe_minimum_mlx_memory_ceiling_bytes, hot_expert_warm_slot_count,
    kept_layer_indexes_to_seat_before_reply, machine_ceiling_from_physical_ram_bytes,
    minimum_decode_warm_slot_count, observed_expert_memory_mode,
    persistent_context_restore_workspace_bytes, plan_expert_residency,
    rotating_committed_token_count, rotating_prefill_transient_token_count,
    safe_minimum_active_memory_ceiling_bytes, seated_complete_layer_count,
    should_commit_mandatory_complete_layer, should_commit_mandatory_routed_page,
    should_commit_multi_token_union_as_complete_layer, should_complete_partial_layer_in_phase,
    should_complete_partial_layer_on_prefill_read, should_seed_partial_layer_on_prefill_read,
};

#[cfg(feature = "direct-mlx")]
pub use memory::SessionExpertRouteSample;

#[cfg(feature = "direct-mlx")]
pub use memory::{MlxAllocationAdmission, MlxAllocationAdmissionError};

pub use inference_engine::contract::{
    EngineGenerationStart, ExpertResidencyTelemetry, PreparedInferenceRequest,
};

pub use inference_engine::load_result::EngineLoadResult;

pub use memory::telemetry::{
    MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
};

pub use model_generation_processor::{
    MalformedModelOutputDiagnostic, ModelGeneratedTokenTranslation, ModelGenerationOutputError,
    ModelGenerationProcessor, PreparedModelGeneration,
};

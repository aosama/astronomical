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

pub mod expert_paging;

pub mod inference_engine;

pub mod memory;

pub mod model_generation_processor;

pub mod structured_generation;

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

pub use inference_engine::{
    GeneratedToken, GenerationFinalization, InferenceEngine, InferenceEngineError,
    MlxInferenceExecution,
};

pub use memory::{
    AllocationAdmissionDecision, AllocationAdmissionObservation, CompleteResidencyDecision,
    CompleteResidencyHeadroomBoundary, CompleteResidencyRequirements, CurrentExpertLayerResidency,
    EnginePlacement, ExpertLayerState, ExpertMemoryAdmissionError, ExpertResidencyPlan,
    ExpertResidencyPlanError, MeasuredExpertLayerPayload, MemoryBoundary,
    MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements, MemoryCeilingUtilization,
    MemoryPhase, MemoryReserveSnapshot, ModelMemoryFacts, ModelMemoryFactsError,
    ModelMemoryGeometry, ModelMemoryGeometryError, ResidentForwardAdmissionDecision,
    ResidentForwardAdmissionRejection, ResidentPromotionBlocker, ResidentPromotionDecision,
    RetainedExpertPageClass, RetainedPrefixRetentionDecision, RetainedPrefixRetentionRequirements,
    RotatingAdmissionError, StreamingEngineRequirement, StreamingStartupResidencyFacts,
    VISIBLE_ANSWER_RESERVE_TOKENS, bounded_output_reservation_token_count,
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
    should_commit_multi_token_union_as_complete_layer,
    should_complete_partial_layer_on_prefill_read,
    should_promote_demoted_streaming_engine_to_resident, should_seed_partial_layer_on_prefill_read,
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

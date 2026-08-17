//! Laguna-owned artifact, normalization, and text protocol contracts.

mod artifacts;
#[cfg(feature = "direct-mlx")]
mod engine;
mod model;
mod moe;
mod normalization;
mod paging;
mod prompt_processing_chunk_sizer;
mod residency_feedback;
#[cfg(feature = "direct-mlx")]
mod startup;
mod text;

pub use artifacts::{
    LagunaArtifactValidationError, LagunaArtifactValidator, LagunaAttentionProjection,
    LagunaCanonicalSourceLayout, LagunaCanonicalTensorAssemblyKind,
    LagunaCanonicalTensorDescriptor, LagunaExpertGateUpLayout, LagunaExpertProjection,
    LagunaGlobalTensorRole, LagunaIndexTotalSizeSemantics, LagunaLayerTensorRole,
    LagunaNonExecutableMetadataDescriptor, LagunaRawTensorNameRecord, LagunaRetainedArtifactFiles,
    LagunaShardIndex, LagunaShardIndexError, LagunaTensorAssembly, LagunaTensorComponent,
    LagunaTensorContract, LagunaTensorId, LagunaTensorNameContract,
    LagunaTensorNameNormalizationError, LagunaTensorNameNormalizer, LagunaTensorSource,
    LagunaTensorSourceDescriptor, LagunaTensorSourceRole, LagunaTensorStorageEncoding,
    ValidatedLagunaArtifact,
};
#[cfg(feature = "direct-mlx")]
pub use engine::{LagunaEngine, LagunaInferenceExecution};
#[cfg(feature = "direct-mlx")]
pub use model::{LagunaDecoderState, LagunaModel, LagunaNativeWeights};
pub use model::{LagunaExecutionError, laguna_decoder_cache_layout};
#[cfg(feature = "direct-mlx")]
pub use moe::route_laguna_native_experts;
pub use moe::{LagunaRouterSelection, apply_router_logit_softcap, select_laguna_router_experts};
pub use normalization::{
    LagunaAffineProfile, LagunaAttentionDescriptor, LagunaAttentionKind, LagunaBlockFp8Profile,
    LagunaCacheDescriptor, LagunaCompressedFeedForwardProjection, LagunaCompressedIgnoreScope,
    LagunaCompressedInputActivationDescriptor, LagunaCompressedModuleScope,
    LagunaCompressedStorageDescriptor, LagunaCompressedWeightEncoding, LagunaDefaultRopeDescriptor,
    LagunaDenseFeedForwardDescriptor, LagunaDirectAffineStorageDescriptor,
    LagunaExactStorageSupport, LagunaExecutionDtype, LagunaFeedForwardDescriptor,
    LagunaFp8InputActivationDescriptor, LagunaFp8KvCacheDescriptor, LagunaGatingKind,
    LagunaLayerDescriptor, LagunaModelDescriptor, LagunaMoeDescriptor, LagunaNormalizationError,
    LagunaNvfp4InputActivationDescriptor, LagunaNvfp4Profile, LagunaRopeDescriptor,
    LagunaRouterKind, LagunaStorageDescriptor, LagunaSymmetricPackedAffineProfile,
    LagunaTargetContract, LagunaTargetNormalizer, LagunaYarnRopeDescriptor,
};
pub use paging::{
    LagunaExpertPagingPlan, LagunaPagingError, LagunaRequestMemoryRequirements,
    LagunaSparseLayerPagingPlan, laguna_sliding_prefill_transient_token_count,
};
#[cfg(feature = "direct-mlx")]
pub use paging::{LagunaExpertWeightPage, forward_paged_routed_swiglu, load_laguna_expert_page};
pub use prompt_processing_chunk_sizer::{
    LagunaPromptProcessingChunkSizer, LagunaPromptProcessingChunkSizerError,
};
pub use residency_feedback::laguna_retained_expert_budget_after_completed_forward;
#[cfg(feature = "direct-mlx")]
pub use startup::{
    LagunaServingSettings, LagunaStartupError, initialize_laguna_execution,
    initialize_laguna_execution_with_serving_settings, initialize_laguna_model,
    initialize_laguna_model_with_serving_settings,
};
pub use text::{
    LagunaGenerationProcessor, LagunaInferenceRequest, LagunaOutputEvent, LagunaOutputParser,
    LagunaOutputParserError, LagunaPreparationError, LagunaPreparedGeneration,
    LagunaPromptRenderer, LagunaPromptRendererError, LagunaRequestOutput, LagunaRequestOutputError,
    LagunaSamplerConfig, LagunaSamplingStrategy, LagunaTextArtifactDescriptor,
    LagunaTextArtifactError, LagunaTextArtifactNormalizer, LagunaTextArtifactSources,
    LagunaTokenDecoder, LagunaTokenizer, LagunaTokenizerError,
};

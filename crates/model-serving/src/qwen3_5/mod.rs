//! Shared Qwen3.5 behavior and dense model support.

pub(crate) mod artifacts;
mod configuration;
mod decoder;
pub(crate) mod dense;
#[cfg(feature = "direct-mlx")]
pub(crate) mod inference_execution;
#[cfg(feature = "direct-mlx")]
pub(crate) mod model;
pub(crate) mod multi_token_prediction;
pub(crate) mod quantizations;
mod text;
mod vision;

pub use artifacts::{
    MAXIMUM_MTPLX_RUNTIME_BYTES, Qwen3_5ArtifactError, Qwen3_5ArtifactValidationError,
    Qwen3_5ArtifactValidator, Qwen3_5MtpArtifactCapability, Qwen3_5MtpContract,
    Qwen3_5MtpContractError, Qwen3_5MtpSidecarDeclaration, Qwen3_5MtpSidecarDeclarationError,
    Qwen3_5MtpSidecarValidationOutcome, Qwen3_5MtpTargetOnlyReason, Qwen3_5ShardIndex,
    ValidatedQwen3_5Artifact, qwen3_5_language_tensor_profiles, qwen3_5_mtp_tensor_names,
    qwen3_5_mtp_tensor_profiles, qwen3_5_resident_language_tensor_profiles,
    validate_qwen3_5_mtp_sidecar_for_tests,
};
pub use configuration::{
    ModelWeightStorage, Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture,
};
pub use decoder::{Qwen3_5DecoderLayerCacheDtypes, qwen3_5_decoder_cache_layout};
#[cfg(feature = "direct-mlx")]
pub use decoder::{
    Qwen3_5MtpRequestState, Qwen3_5MtpRequestStateAllocationCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
};
#[cfg(feature = "direct-mlx")]
pub use inference_execution::{
    Qwen3_5Engine, Qwen3_5MtpRuntimeState, Qwen3_5PrefillExecutionContext,
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5PromptProcessingChunkSizerError,
    Qwen3_5SpeculativePrefillFailureStageForTests, Qwen3_5SpeculativePrefillSelectionError,
    persistent_prompt_cache_publication_advances_parent_chain, qwen3_5_depth_one_mtp_window_fits,
    qwen3_5_mtp_runtime_configuration_after_load, qwen3_5_mtp_runtime_state_after_load,
    qwen3_5_select_speculative_prefill_token_positions,
    qwen3_5_selected_speculative_prefill_positions_for_range,
    safe_minimum_mlx_memory_ceiling_bytes,
};
#[cfg(feature = "direct-mlx")]
pub use model::{
    Qwen3_5ExecutionError, Qwen3_5GatedDeltaBoundaryCheckpointResult, Qwen3_5Model,
    Qwen3_5ModelChunkingConfiguration, Qwen3_5MtpForwardOutput, Qwen3_5TargetForwardOutput,
    Qwen3_5TargetVerificationProjection, Qwen3_5TargetVerificationProjectionDispatch,
    Qwen3_5Weights, qwen3_5_aggregate_speculative_prefill_attention_weights,
    qwen3_5_full_attention_step, qwen3_5_gated_delta_checkpoint_kernel, qwen3_5_gated_delta_kernel,
    qwen3_5_gated_delta_sequence, qwen3_5_gated_delta_sequence_with_boundary_checkpoints,
    qwen3_5_gated_delta_step, qwen3_5_select_speculative_prefill_token_positions_on_gpu,
    qwen3_5_target_verification_quantized_linear, target_verification_quantized_linear_kernel,
};
pub use multi_token_prediction::{
    MtpDepthDowngradeReason, MtpDraftDepth, MtpDraftDepthError, MtpMemoryAdmission,
    MtpMemoryCandidate, MtpMemoryProjection, MtpMemoryProjectionError, MtpVerificationDecision,
    MtpVerificationDecisionError, qwen3_5_mtp_effective_depth_and_reason_for_windows,
    qwen3_5_mtp_effective_depth_for_windows, qwen3_5_mtp_memory_admission,
    qwen3_5_mtp_verification_decision, qwen3_5_mtp_verification_transient_array_bytes,
};
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use multi_token_prediction::{VerifiedEmissionQueue, VerifiedTargetFrontier};
pub use quantizations::optiq::{OptiQMetadata, OptiQMetadataError, OptiQQuantizationProfile};
#[cfg(feature = "direct-mlx")]
pub use text::qwen3_5_apply_top_p_mask;
pub use text::{
    Qwen3_5GenerationProcessor, Qwen3_5InferenceRequest, Qwen3_5OutputEvent, Qwen3_5OutputParser,
    Qwen3_5OutputParserError, Qwen3_5PromptError, Qwen3_5PromptRenderer, Qwen3_5RenderedPrompt,
    Qwen3_5RequestOutput, Qwen3_5RequestOutputError, Qwen3_5SamplerConfig, Qwen3_5SamplingStrategy,
    Qwen3_5ThinkingBudgetError, Qwen3_5ThinkingBudgetState, Qwen3_5TokenDecoder, Qwen3_5TokenIds,
    Qwen3_5Tokenizer, Qwen3_5TokenizerError, Qwen3_5ToolCall, discover_sampler_config,
    discover_token_ids, qwen3_5_request_enables_thinking, resolve_sampling_seed,
    translate_qwen3_5_preparation_error, translate_request_output_error,
    validate_context_token_count,
};
pub use vision::{
    Qwen3_5ImageDimensions, Qwen3_5ImageGrid, Qwen3_5ImageProcessingError, Qwen3_5ImageProcessor,
    Qwen3_5ProcessedImage, Qwen3_5VisionConfig, Qwen3_5VisionInputPlan,
    Qwen3_5VisionInputPlanError, Qwen3_5VisualEmbeddingRequiredImage,
    Qwen3_5VisualEmbeddingSuffixPlan, Qwen3_5VisualEmbeddingSuffixPlanError,
    Qwen3_5VisualPromptCacheIdentityPlan, Qwen3_5VisualPromptCacheIdentityPlanError,
    plan_qwen3_5_visual_embedding_suffix, plan_qwen3_5_visual_prompt_cache_block_inputs,
    qwen3_5_vision_tensor_profiles,
};
#[cfg(feature = "direct-mlx")]
pub use vision::{Qwen3_5VisionModel, Qwen3_5VisionWeights, qwen3_5_inject_visual_embeddings};
#[cfg(feature = "direct-mlx")]
pub(crate) use vision::{
    qwen3_5_speculative_draft_block_causal_input, qwen3_5_speculative_draft_block_causal_inputs,
};

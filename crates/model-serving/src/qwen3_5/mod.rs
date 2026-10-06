//! Shared Qwen3.5 behavior and dense model support.

pub(crate) mod artifacts;
mod configuration;
mod decoder;
pub(crate) mod dense;
#[cfg(feature = "direct-mlx")]
pub(crate) mod inference_execution;
#[cfg(feature = "direct-mlx")]
pub(crate) mod model;
pub(crate) mod quantizations;
mod text;
mod vision;

pub use artifacts::{
    Qwen3_5ArtifactError, Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator,
    Qwen3_5RamBudgetGeometryError, Qwen3_5ShardIndex, ValidatedQwen3_5Artifact,
    mlx_ram_budget_model_geometry_from_validated_artifact, qwen3_5_language_tensor_profiles,
    qwen3_5_resident_language_tensor_profiles,
};
pub use configuration::{
    ModelWeightStorage, Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture,
};
pub use decoder::{Qwen3_5DecoderLayerCacheDtypes, qwen3_5_decoder_cache_layout};
#[cfg(feature = "direct-mlx")]
pub use decoder::{
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
};
#[cfg(feature = "direct-mlx")]
pub use inference_execution::{
    Qwen3_5Engine, Qwen3_5PrefillExecutionContext, Qwen3_5PromptProcessingChunkSizer,
    Qwen3_5PromptProcessingChunkSizerError,
    persistent_prompt_cache_publication_advances_parent_chain,
    safe_minimum_mlx_memory_ceiling_bytes,
};
#[cfg(feature = "direct-mlx")]
pub use model::{
    Qwen3_5ExecutionError, Qwen3_5GatedDeltaBoundaryCheckpointResult, Qwen3_5Model,
    Qwen3_5ModelChunkingConfiguration, Qwen3_5TargetForwardOutput, Qwen3_5Weights,
    is_gdn_decode_prework_eligible, qwen3_5_full_attention_step,
    qwen3_5_gated_delta_checkpoint_kernel, qwen3_5_gated_delta_kernel,
    qwen3_5_gated_delta_sequence, qwen3_5_gated_delta_sequence_ops_fallback,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints,
    qwen3_5_gated_delta_sequence_with_boundary_checkpoints_ops_fallback, qwen3_5_gated_delta_step,
    qwen3_5_gdn_decode_prework, qwen3_5_gdn_decode_prework_kernel,
};
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
pub use vision::{
    Qwen3_5VisionModel, Qwen3_5VisionPaddingZeroCache, Qwen3_5VisionWeights,
    qwen3_5_inject_visual_embeddings,
};

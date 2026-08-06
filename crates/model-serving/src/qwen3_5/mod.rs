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
    Qwen3_5MtpArtifactCapability, Qwen3_5ShardIndex, ValidatedQwen3_5Artifact,
    qwen3_5_language_tensor_profiles, qwen3_5_mtp_tensor_names, qwen3_5_mtp_tensor_profiles,
    qwen3_5_resident_language_tensor_profiles,
};
pub use configuration::{
    ModelWeightStorage, Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture,
};
pub use decoder::qwen3_5_decoder_cache_layout;
#[cfg(feature = "direct-mlx")]
pub use decoder::{
    Qwen3_5MtpRequestState, Qwen3_5MtpRequestStateAllocationCheckpoint,
    Qwen3_5MtpUnavailableReason, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
};
#[cfg(feature = "direct-mlx")]
pub use inference_execution::{
    Qwen3_5Engine, Qwen3_5MtpRuntimeState, Qwen3_5PrefillChunckSizer,
    Qwen3_5PrefillChunckSizerError, qwen3_5_depth_one_mtp_window_fits,
    qwen3_5_mtp_runtime_state_after_load, qwen3_5_mtp_verification_may_cross_thinking_budget,
    safe_minimum_mlx_memory_ceiling_bytes,
};
#[cfg(feature = "direct-mlx")]
pub use model::{
    Qwen3_5ExecutionError, Qwen3_5Model, Qwen3_5MtpForwardOutput, Qwen3_5TargetForwardOutput,
    Qwen3_5Weights, combined_target_and_mtp_persistent_growth_bytes,
    context_memory_admission_projected_active_memory_bytes,
    persistent_prompt_cache_restore_temporary_workspace_bytes, qwen3_5_full_attention_step,
    qwen3_5_gated_delta_kernel, qwen3_5_gated_delta_sequence, qwen3_5_gated_delta_step,
};
pub use quantizations::optiq::{OptiQMetadata, OptiQMetadataError, OptiQQuantizationProfile};
#[cfg(feature = "direct-mlx")]
pub use text::qwen3_5_apply_top_p_mask;
pub use text::{
    Qwen3_5GenerationProcessor, Qwen3_5InferenceRequest, Qwen3_5OutputEvent, Qwen3_5OutputParser,
    Qwen3_5OutputParserError, Qwen3_5PromptError, Qwen3_5PromptRenderer, Qwen3_5RequestOutput,
    Qwen3_5RequestOutputError, Qwen3_5SamplerConfig, Qwen3_5SamplingStrategy, Qwen3_5TokenDecoder,
    Qwen3_5TokenIds, Qwen3_5Tokenizer, Qwen3_5TokenizerError, Qwen3_5ToolCall,
    discover_sampler_config, discover_token_ids, qwen3_5_request_enables_thinking,
    resolve_sampling_seed, translate_qwen3_5_preparation_error, translate_request_output_error,
    validate_context_token_count,
};
pub use vision::{
    Qwen3_5ImageDimensions, Qwen3_5ImageGrid, Qwen3_5ImageProcessingError, Qwen3_5ImageProcessor,
    Qwen3_5ProcessedImage, Qwen3_5VisionConfig, Qwen3_5VisionInputPlan,
    Qwen3_5VisionInputPlanError, Qwen3_5VisualEmbeddingRequiredImage,
    Qwen3_5VisualEmbeddingSuffixPlan, Qwen3_5VisualEmbeddingSuffixPlanError,
    plan_qwen3_5_visual_embedding_suffix, qwen3_5_vision_tensor_profiles,
};
#[cfg(feature = "direct-mlx")]
pub use vision::{Qwen3_5VisionModel, Qwen3_5VisionWeights, qwen3_5_inject_visual_embeddings};

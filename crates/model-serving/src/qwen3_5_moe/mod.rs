mod artifacts;
mod configuration;
mod decoder;
#[cfg(feature = "direct-mlx")]
pub(crate) mod inference_execution;
#[cfg(feature = "direct-mlx")]
pub use expert_paging::{
    ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES, AlignedExpertPackBuildRequest,
    AlignedExpertPackError, AlignedExpertPackHeader, AlignedExpertPackPreparationError,
    AlignedExpertPackPreparationInspection, AlignedExpertPackPreparationProgress,
    AlignedExpertPackPreparationReport, AlignedExpertPackPreparer,
    AlignedExpertPackTensorDescriptor, ExpertManifestError, ExpertPager, ExpertPagingError,
    ExpertWeightMemoryCache, ExpertWeightMemoryCacheRequestReport,
    ExpertWeightMemoryCacheStatistics, LiveMetalBudget, MemoryBudgetError, MemoryBudgetSnapshot,
    PagedExpertWeights, QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
    QuantizedExpertShardManifest, QuantizedExpertSourceInterval, QuantizedExpertTensorRange,
    QuantizedTensorSource, SafetensorsDtype, SafetensorsHeader, SafetensorsHeaderError,
    TensorHeaderEntry, automatic_expert_weight_memory_cache_maximum_size_bytes,
    build_aligned_expert_pack, build_aligned_expert_pack_metal_io_descriptors,
    build_quantized_expert_layer_plan, build_quantized_expert_page_manifest_from_plan,
    build_source_manifests, contiguous_selected_runs, load_quantized_expert_page,
    parse_safetensors_header, read_aligned_expert_pack_header, validate_aligned_expert_pack_header,
    validate_aligned_expert_pack_payload, validate_expert_ids, validate_quantization_contract,
    validate_source_intervals, validate_virtual_intervals,
};
#[cfg(feature = "direct-mlx")]
pub(crate) mod expert_paging;
mod model;
pub(crate) mod quantizations;
mod text;
mod vision;

pub use artifacts::{
    Qwen3_5MoEArtifactError, Qwen3_5MoEArtifactValidationError, Qwen3_5MoEArtifactValidator,
    Qwen3_5MoEMtpArtifactCapability, Qwen3_5MoEShardIndex, ValidatedQwen3_5MoEArtifact,
    qwen3_5_moe_language_tensor_profiles, qwen3_5_moe_mtp_tensor_profiles,
    qwen3_5_moe_quantized_mtp_tensor_names, qwen3_5_moe_resident_language_tensor_profiles,
};
pub use configuration::{ModelWeightStorage, Qwen3_5MoEConfig, Qwen3_5MoEConfigError};
pub use decoder::qwen3_5_moe_decoder_cache_layout;
#[cfg(feature = "direct-mlx")]
pub use decoder::{
    Qwen3_5MoEMtpRequestState, Qwen3_5MoEMtpRequestStateAllocationCheckpoint,
    Qwen3_5MoEMtpUnavailableReason, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
};
#[cfg(feature = "direct-mlx")]
pub use inference_execution::{
    Qwen3_5MoEEngine, Qwen3_5MoEMtpRuntimeState, Qwen3_5MoEPrefillChunckSizer,
    Qwen3_5MoEPrefillChunckSizerError, qwen3_5_moe_depth_one_mtp_window_fits,
    qwen3_5_moe_mtp_runtime_state_after_load,
    qwen3_5_moe_mtp_verification_may_cross_thinking_budget, safe_minimum_mlx_memory_ceiling_bytes,
};
#[cfg(feature = "direct-mlx")]
pub(in crate::qwen3_5_moe) use model::reclaim_retained_experts_for_request_memory_pressure;
#[cfg(feature = "direct-mlx")]
pub use model::{
    Qwen3_5MoEExecutionError, Qwen3_5MoEModel, Qwen3_5MoEMtpForwardOutput,
    Qwen3_5MoETargetForwardOutput, Qwen3_5MoEWeights,
};
#[cfg(feature = "direct-mlx")]
pub use model::{
    Qwen3_5MoEPagedPrefillExecutionMode, combined_target_and_mtp_persistent_growth_bytes,
    context_memory_admission_projected_active_memory_bytes,
    persistent_prompt_cache_restore_temporary_workspace_bytes, qwen3_5_moe_combine_experts,
    qwen3_5_moe_full_attention_step, qwen3_5_moe_fused_four_bit_affine_gate_up_swiglu,
    qwen3_5_moe_fused_four_bit_affine_gate_up_swiglu_kernel, qwen3_5_moe_gated_delta_kernel,
    qwen3_5_moe_gated_delta_sequence, qwen3_5_moe_gated_delta_step,
    qwen3_5_moe_remap_expert_page_slots, qwen3_5_moe_restore_expert_assignment_order,
    qwen3_5_moe_route_experts, qwen3_5_moe_sort_expert_assignments,
    qwen3_5_moe_sorted_expert_weighted_sum, qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
pub use quantizations::optiq::{OptiQMetadata, OptiQMetadataError, OptiQQuantizationProfile};
#[cfg(feature = "direct-mlx")]
pub use text::qwen3_5_moe_apply_top_p_mask;
pub use text::{
    Qwen3_5MoEGenerationProcessor, Qwen3_5MoEInferenceRequest, Qwen3_5MoEOutputEvent,
    Qwen3_5MoEOutputParser, Qwen3_5MoEOutputParserError, Qwen3_5MoEPromptError,
    Qwen3_5MoEPromptRenderer, Qwen3_5MoERequestOutput, Qwen3_5MoERequestOutputError,
    Qwen3_5MoESamplerConfig, Qwen3_5MoESamplingStrategy, Qwen3_5MoETokenDecoder,
    Qwen3_5MoETokenIds, Qwen3_5MoETokenizer, Qwen3_5MoETokenizerError, Qwen3_5MoEToolCall,
    discover_sampler_config, discover_token_ids, resolve_sampling_seed,
    translate_qwen3_5_moe_preparation_error, translate_request_output_error,
    validate_context_token_count,
};
pub use vision::{
    Qwen3_5MoEImageDimensions, Qwen3_5MoEImageGrid, Qwen3_5MoEImageProcessingError,
    Qwen3_5MoEImageProcessor, Qwen3_5MoEProcessedImage, Qwen3_5MoEVisionConfig,
    Qwen3_5MoEVisionInputPlan, Qwen3_5MoEVisionInputPlanError,
    Qwen3_5MoEVisualEmbeddingRequiredImage, Qwen3_5MoEVisualEmbeddingSuffixPlan,
    Qwen3_5MoEVisualEmbeddingSuffixPlanError, plan_qwen3_5_moe_visual_embedding_suffix,
    qwen3_5_moe_vision_tensor_profiles,
};
#[cfg(feature = "direct-mlx")]
pub use vision::{
    Qwen3_5MoEVisionModel, Qwen3_5MoEVisionWeights, qwen3_5_moe_inject_visual_embeddings,
};

/// Model identity constants retained for test compatibility.
/// These match the original hardcoded values from the per-model profile era.
pub const ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
pub const ORNITH_1_0_35B_OPTIQ_4BIT_REVISION: &str = "ce62c23d34b91d84f838e0b292d517dbe4b9b60f";

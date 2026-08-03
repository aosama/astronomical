pub(crate) mod artifacts;
#[cfg(feature = "direct-mlx")]
pub(crate) mod expert_paging;
#[cfg(feature = "direct-mlx")]
pub(crate) mod model;

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
pub(crate) use model::feed_forward_weights::bind_qwen3_5_moe_feed_forward_weights;
#[cfg(feature = "direct-mlx")]
pub(crate) use model::reclaim_retained_experts_for_request_memory_pressure;
#[cfg(feature = "direct-mlx")]
pub use model::{
    Qwen3_5MoEPagedPrefillExecutionMode, qwen3_5_moe_combine_experts,
    qwen3_5_moe_remap_expert_page_slots, qwen3_5_moe_restore_expert_assignment_order,
    qwen3_5_moe_route_experts, qwen3_5_moe_sort_expert_assignments,
    qwen3_5_moe_sorted_expert_weighted_sum, qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};

/// Model identity constants retained for sparse-artifact test fixtures.
pub const ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
pub const ORNITH_1_0_35B_OPTIQ_4BIT_REVISION: &str = "ce62c23d34b91d84f838e0b292d517dbe4b9b60f";

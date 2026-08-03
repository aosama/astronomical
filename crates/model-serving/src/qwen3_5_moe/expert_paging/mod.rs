//! Expert paging for Astronomical's Qwen3.5-MoE inference engine.
//!
//! This module provides fail-closed, bounded expert-streaming for loading MoE
//! expert weights from SSD on demand. It retains complete layers when the live
//! memory budget permits and otherwise reads routed experts through bounded
//! `pread()`-style input/output.
//!
//! Architecture:
//! - `safetensors_header`: Pure Rust safetensors header parsing without MLX dependency
//! - `quantized_expert_manifest`: Types and page manifest builder for selected experts
//! - `quantized_expert_layer_plan`: Startup-validated layer-plan construction and
//!   source-manifest building
//! - `quantized_expert_validation`: Pure validation functions (expert IDs, quantization
//!   contract, source/virtual intervals)
//! - `bounded_expert_reader`: High-level loading pipeline that takes manifests and
//!   produces MLX arrays via the runtime-integration bounded reader
//! - `memory_budget`: Fail-closed Metal memory budget enforcement using MLX counters
//! - `expert_pager`: Coordination layer that builds layer plans at startup and
//!   loads selected expert weights per decode step

mod aligned_expert_pack;
mod aligned_expert_pack_layout;
mod aligned_expert_pack_loader;
mod aligned_expert_pack_positional_io;
mod aligned_expert_pack_preparer;
pub mod bounded_expert_reader;
mod complete_layer_retention;
pub mod expert_cache;
mod expert_cache_capacity;
mod expert_cache_eviction;
mod expert_cache_page_assembly;
mod expert_cache_pressure;
mod expert_cache_statistics;
pub mod expert_pager;
mod expert_pager_construction;
pub mod memory_budget;
mod paged_expert_weights;
pub mod quantized_expert_layer_plan;
pub mod quantized_expert_manifest;
pub mod quantized_expert_validation;
pub mod safetensors_header;

pub use aligned_expert_pack::{
    ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES, AlignedExpertPackBuildRequest,
    AlignedExpertPackError, AlignedExpertPackHeader, AlignedExpertPackTensorDescriptor,
    build_aligned_expert_pack, read_aligned_expert_pack_header,
    validate_aligned_expert_pack_header, validate_aligned_expert_pack_payload,
};
pub use aligned_expert_pack_loader::build_aligned_expert_pack_metal_io_descriptors;
pub use aligned_expert_pack_preparer::{
    AlignedExpertPackPreparationError, AlignedExpertPackPreparationInspection,
    AlignedExpertPackPreparationProgress, AlignedExpertPackPreparationReport,
    AlignedExpertPackPreparer,
};
#[allow(unused_imports)] // Public API re-exports; will be used by external consumers
pub use bounded_expert_reader::load_quantized_expert_page;
pub use expert_cache::ExpertWeightMemoryCache;
pub use expert_cache_statistics::{
    ExpertWeightMemoryCacheRequestReport, ExpertWeightMemoryCacheStatistics,
};
pub use expert_pager::{ExpertPager, ExpertPagingError, PagedExpertWeights};
pub use memory_budget::{
    LiveMetalBudget, MemoryBudgetError, MemoryBudgetSnapshot,
    automatic_expert_weight_memory_cache_maximum_size_bytes,
};
pub use quantized_expert_layer_plan::{
    build_quantized_expert_layer_plan, build_source_manifests, contiguous_selected_runs,
};
pub use quantized_expert_manifest::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
    QuantizedExpertShardManifest, QuantizedExpertSourceInterval, QuantizedExpertTensorRange,
    QuantizedTensorSource, build_quantized_expert_page_manifest_from_plan,
};
pub use quantized_expert_validation::{
    validate_expert_ids, validate_quantization_contract, validate_source_intervals,
    validate_virtual_intervals,
};
pub use safetensors_header::{
    SafetensorsDtype, SafetensorsHeader, SafetensorsHeaderError, TensorHeaderEntry,
    parse_safetensors_header,
};

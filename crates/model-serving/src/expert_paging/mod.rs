//! Architecture-neutral expert storage, bounded loading, and retention policy.

use std::fmt::Debug;

mod bounded_expert_reader;
mod expert_cache;
mod expert_cache_capacity;
mod expert_cache_eviction;
mod expert_cache_pressure;
mod expert_cache_statistics;
pub mod memory_budget;
pub mod quantized_expert_manifest;
pub mod quantized_expert_validation;
pub mod safetensors_header;
mod source_manifests;

/// Family-owned expert page payload admitted by the shared retention policy.
pub trait ExpertWeightPage: Debug {
    /// Returns the exact MLX payload bytes owned by this page.
    fn resident_payload_byte_count(&self) -> u64;
}

pub(crate) use bounded_expert_reader::load_quantized_expert_page;
pub use expert_cache::ExpertWeightMemoryCache;
pub use expert_cache_statistics::{
    ExpertWeightMemoryCacheRequestReport, ExpertWeightMemoryCacheStatistics,
};
pub use memory_budget::{
    LiveMetalBudget, MemoryBudgetError, MemoryBudgetSnapshot,
    automatic_expert_weight_memory_cache_maximum_size_bytes,
};
pub use quantized_expert_manifest::{
    ExpertManifestError, QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertPageManifest,
    QuantizedExpertShardManifest, QuantizedExpertSourceInterval, QuantizedExpertTensorRange,
    QuantizedTensorSource, build_quantized_expert_cache_population_manifest_from_plan,
    build_quantized_expert_page_manifest_from_plan,
};
pub use quantized_expert_validation::{
    validate_expert_ids, validate_quantization_contract, validate_source_intervals,
    validate_virtual_intervals,
};
pub use safetensors_header::{
    SafetensorsDtype, SafetensorsHeader, SafetensorsHeaderError, TensorHeaderEntry,
    parse_safetensors_header,
};
pub use source_manifests::{build_source_manifests, contiguous_selected_runs};

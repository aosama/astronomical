//! Architecture-neutral expert storage, bounded loading, and retention policy.

mod expert_cache_statistics;
pub mod memory_budget;
pub mod quantized_expert_manifest;
pub mod quantized_expert_validation;
pub mod safetensors_header;
mod source_manifests;

pub use expert_cache_statistics::ExpertWeightMemoryCacheStatistics;
pub use memory_budget::{
    LiveMetalBudget, MemoryBudgetError, MemoryBudgetSnapshot,
    automatic_expert_weight_memory_cache_maximum_size_bytes,
    maximum_possible_expert_route_payload_bytes,
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
pub use source_manifests::{build_source_manifests, contiguous_selected_runs};

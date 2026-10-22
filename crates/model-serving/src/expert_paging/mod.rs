//! Expert-storage owner in the serving crate: the retained cache slot
//! tables, the quantized expert manifests (which embed model-family
//! configuration types), and the bounded MLX reader stay beside the
//! bindings and the families. The pure safetensors-header parsing (over
//! `astronomical-memory-policy`'s `safetensors` and `strict_json`) and the
//! cache statistics are re-exported from the policy crate; the module
//! re-export keeps `crate::expert_paging::safetensors_header::...` paths
//! resolving for their consumers.

#[cfg(feature = "direct-mlx")]
mod bounded_expert_reader;
pub mod quantized_expert_manifest;
pub mod quantized_expert_validation;
mod retained_expert_page_cache;
mod source_manifests;

#[cfg(test)]
mod quantized_expert_page_manifest_tests;
#[cfg(test)]
mod route_observation_tests;

pub use astronomical_memory_policy::expert_paging::ExpertWeightMemoryCacheStatistics;
// Module re-export: qwen3_5_core reaches the parser by module path
// (`crate::expert_paging::safetensors_header::parse_safetensors_header`).
pub use astronomical_memory_policy::expert_paging::safetensors_header;
pub use quantized_expert_manifest::{
    ExpertManifestError, ExpertPageRoutePartition, QuantizationMode, QuantizedExpertLayerPlan,
    QuantizedExpertPageManifest, QuantizedExpertShardManifest, QuantizedExpertSourceInterval,
    QuantizedExpertTensorRange, QuantizedTensorSource,
    build_quantized_expert_page_manifest_from_plan,
};
pub use quantized_expert_validation::{
    validate_expert_ids, validate_quantization_contract, validate_source_intervals,
    validate_virtual_intervals,
};
pub use retained_expert_page_cache::{
    RetainedExpertLayerCommit, RetainedExpertLayerCommitDelta, RetainedExpertLayerCommitError,
    RetainedExpertLayerCommitOutcome, RetainedExpertPageCache, RetainedExpertReclamation,
    last_prefill_chunk_demand_weight,
};
pub use safetensors_header::{
    SafetensorsDtype, SafetensorsHeader, SafetensorsHeaderError, TensorHeaderEntry,
    parse_safetensors_header,
};

/// Family-owned expert payload retained by the shared RAM policy.
pub trait ExpertWeightPage: std::fmt::Debug {
    fn resident_payload_byte_count(&self) -> u64;
}
#[cfg(feature = "direct-mlx")]
pub use bounded_expert_reader::load_quantized_expert_page;
#[cfg(feature = "direct-mlx")]
pub use source_manifests::{build_source_manifests, contiguous_selected_runs};

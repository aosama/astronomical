//! Architecture-neutral expert storage, bounded loading, and retention mechanism.

#[cfg(feature = "direct-mlx")]
mod bounded_expert_reader;
mod expert_cache_statistics;
pub mod quantized_expert_manifest;
pub mod quantized_expert_validation;
mod retained_expert_page_cache;
pub mod safetensors_header;
mod source_manifests;
mod streaming_expert_pack_pages;
#[cfg(feature = "direct-mlx")]
mod streaming_expert_pack_plans;
pub mod streaming_expert_packs;

pub use expert_cache_statistics::ExpertWeightMemoryCacheStatistics;
pub use retained_expert_page_cache::{
    RetainedExpertLayerCommit, RetainedExpertLayerCommitDelta, RetainedExpertLayerCommitError,
    RetainedExpertLayerCommitOutcome, RetainedExpertPageCache, RetainedExpertReclamation,
    last_prefill_chunk_demand_weight,
};

/// Family-owned expert payload retained by the shared RAM policy.
pub trait ExpertWeightPage: std::fmt::Debug {
    fn resident_payload_byte_count(&self) -> u64;
}
#[cfg(feature = "direct-mlx")]
pub use bounded_expert_reader::load_quantized_expert_page;
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
pub use safetensors_header::{
    SafetensorsDtype, SafetensorsHeader, SafetensorsHeaderError, TensorHeaderEntry,
    parse_safetensors_header,
};
#[cfg(feature = "direct-mlx")]
pub use source_manifests::{build_source_manifests, contiguous_selected_runs};
#[cfg(feature = "direct-mlx")]
pub use streaming_expert_pack_pages::assemble_streaming_expert_page_tensors;
pub use streaming_expert_pack_pages::build_streaming_expert_page_manifest;
#[cfg(feature = "direct-mlx")]
pub use streaming_expert_pack_plans::build_streaming_expert_layer_plans;
pub use streaming_expert_packs::{
    StreamingExpertPackError, StreamingExpertPackSources, detect_streaming_expert_pack_sources,
};

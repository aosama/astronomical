//! Architecture-neutral expert storage, bounded loading, and retention mechanism.

#[cfg(feature = "direct-mlx")]
mod bounded_expert_reader;
mod expert_cache_statistics;
mod paged_decode_layer_disposition;
mod phase_aware_expert_residency;
pub mod quantized_expert_manifest;
pub mod quantized_expert_validation;
mod retained_expert_layer_cache;
pub mod safetensors_header;
mod source_manifests;

pub use expert_cache_statistics::ExpertWeightMemoryCacheStatistics;
pub use paged_decode_layer_disposition::PagedDecodeLayerDisposition;
pub use phase_aware_expert_residency::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPhase, PhaseAwareExpertResidencyPlan, PhaseAwareExpertResidencyPlanError,
    RetainedExpertPageClass, plan_phase_aware_expert_residency,
};
pub use retained_expert_layer_cache::{
    RetainedExpertLayerCache, RetainedExpertLayerCommit, RetainedExpertLayerCommitDelta,
    RetainedExpertLayerCommitError, RetainedExpertLayerCommitOutcome, RetainedExpertReclamation,
    last_prefill_chunk_demand_weight,
};

/// Family-owned expert payload retained by the shared RAM policy.
pub trait ExpertWeightPage: std::fmt::Debug {
    fn resident_payload_byte_count(&self) -> u64;
}
#[cfg(feature = "direct-mlx")]
pub(crate) use bounded_expert_reader::load_quantized_expert_page;
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

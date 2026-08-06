//! Bounded multi-range expert page reader for MLX safetensors loading.
//!
//! This module implements the high-level loading pipeline that takes a
//! `QuantizedExpertPageManifest`, constructs bounded readers for each source
//! shard, and produces MLX arrays containing only the selected expert weights.
//! The actual I/O vtable and interval mapping is handled by
//! `MlxRuntime::load_safetensors_from_bounded_ranges` in the runtime-integration
//! crate.

use std::{collections::HashMap, sync::Arc};

use astronomical_runtime_integration::{MlxArray, PositionalFileReadMetrics};

use super::quantized_expert_manifest::{
    ExpertManifestError, QuantizedExpertPageManifest, QuantizedExpertShardManifest,
};

/// Load tensors for a single quantized expert page manifest.
///
/// For each source manifest in the page, constructs a bounded reader that
/// serves a synthetic safetensors header and maps virtual payload offsets
/// to exact byte ranges in the source file. Returns a HashMap from tensor
/// name to MLX array.
///
/// This is the primary entry point for expert paging: given a page manifest
/// (which describes which expert rows to load from which shard files), it
/// produces the actual MLX arrays containing only the selected expert weights.
pub fn load_quantized_expert_page(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    page_manifest: &QuantizedExpertPageManifest,
    expert_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
) -> Result<HashMap<String, MlxArray>, ExpertManifestError> {
    let mut loaded_tensors: HashMap<String, MlxArray> = HashMap::new();
    for source_manifest in &page_manifest.source_manifests {
        let shard_tensors = load_shard_manifest(
            runtime,
            source_manifest,
            expert_file_read_metrics.as_ref().map(Arc::clone),
        )?;
        for (tensor_name, array) in shard_tensors {
            if loaded_tensors.contains_key(&tensor_name) {
                return Err(ExpertManifestError::OverlappingSourceIntervals {
                    source_file_offset: 0,
                });
            }
            loaded_tensors.insert(tensor_name, array);
        }
    }
    Ok(loaded_tensors)
}

/// Load tensors from a single shard manifest using the bounded reader pattern.
fn load_shard_manifest(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    source_manifest: &QuantizedExpertShardManifest,
    expert_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
) -> Result<HashMap<String, MlxArray>, ExpertManifestError> {
    let synthetic_header_bytes = source_manifest.rebased_safetensors_header()?;
    let source_file = std::fs::File::open(&source_manifest.source_file).map_err(|source| {
        ExpertManifestError::SafetensorsHeader(
            super::safetensors_header::SafetensorsHeaderError::Io(source),
        )
    })?;

    // Convert manifest intervals to the runtime-integration interval type.
    let intervals: Vec<astronomical_runtime_integration::BoundedReadInterval> = source_manifest
        .source_intervals
        .iter()
        .map(
            |interval| astronomical_runtime_integration::BoundedReadInterval {
                virtual_payload_offset: interval.virtual_payload_offset,
                source_file_offset: interval.source_file_offset,
                source_byte_count: interval.source_byte_count,
            },
        )
        .collect();

    let load_result = runtime
        .load_safetensors_from_bounded_ranges(
            source_file,
            synthetic_header_bytes,
            intervals,
            source_manifest.payload_byte_count,
            expert_file_read_metrics,
        )
        .map_err(|source| ExpertManifestError::ReaderError {
            description: source.to_string(),
        })?;

    // Extract all tensors from the load result by iterating the manifest's
    // tensor ranges and looking up each one by name.
    let mut result = HashMap::new();
    for tensor_range in &source_manifest.tensor_ranges {
        let array = load_result
            .tensor(&tensor_range.tensor_name)
            .map_err(|source| ExpertManifestError::ReaderError {
                description: source.to_string(),
            })?;
        result.insert(tensor_range.tensor_name.clone(), array);
    }
    Ok(result)
}

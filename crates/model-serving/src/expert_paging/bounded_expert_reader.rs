//! Bounded SafeTensors loading for one Rust-selected expert page.
//!
//! # Why this exists
//!
//! A model shard can contain far more expert data than one forward needs. Asking
//! the ordinary SafeTensors loader to open the whole shard would make a routed
//! top-K decode miss behave like a whole-shard load. The manifest built earlier
//! in the pipeline instead describes only the exact source byte intervals for the
//! selected experts. This module turns that validated description into lazy MLX
//! arrays without teaching this generic reader anything about Qwen projections.
//!
//! # Ownership and ordering
//!
//! 1. The caller has already validated tensor geometry and selected experts.
//! 2. Each source shard receives a synthetic SafeTensors header whose offsets
//!    refer to a compact *virtual* payload.
//! 3. The bounded MLX reader presents exact ranges from the real file as that
//!    virtual payload. Gaps and unselected experts are never part of the request.
//! 4. Arrays remain lazy until the model evaluates them. Keeping file ownership
//!    inside the bounded load result is therefore required for correctness.
//! 5. The returned name-to-array map is consumed by the architecture-specific
//!    page builder, which gives the arrays their gate/up/down meaning.
//!
//! The optional positional-read metrics count logical reads issued by this path.
//! They are not proof of equal physical SSD traffic; macOS may satisfy reads from
//! its file cache.

use std::{collections::HashMap, sync::Arc};

use astronomical_runtime_integration::{MlxArray, PositionalFileReadMetrics};

use super::quantized_expert_manifest::{
    ExpertManifestError, QuantizedExpertPageManifest, QuantizedExpertShardManifest,
};

/// Loads only the validated tensor ranges described by one expert-page manifest.
///
/// This is public so architecture-neutral qualification tools can compare the
/// production bounded-reader baseline with alternative expert-pack layouts.
pub fn load_quantized_expert_page(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    page_manifest: &QuantizedExpertPageManifest,
    expert_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
) -> Result<HashMap<String, MlxArray>, ExpertManifestError> {
    // One logical page may span several SafeTensors shards. Merge their tensors
    // only after each shard has independently passed bounded-header construction.
    // Duplicate names indicate an invalid manifest: silently replacing an array
    // here could pair one projection's weight with another projection's metadata.
    let mut loaded_tensors = HashMap::new();
    for source_manifest in &page_manifest.source_manifests {
        let shard_tensors = load_shard_manifest(
            runtime,
            source_manifest,
            expert_file_read_metrics.as_ref().map(Arc::clone),
        )?;
        for (tensor_name, tensor) in shard_tensors {
            if loaded_tensors.insert(tensor_name, tensor).is_some() {
                return Err(ExpertManifestError::OverlappingSourceIntervals {
                    source_file_offset: 0,
                });
            }
        }
    }
    Ok(loaded_tensors)
}

fn load_shard_manifest(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    source_manifest: &QuantizedExpertShardManifest,
    expert_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
) -> Result<HashMap<String, MlxArray>, ExpertManifestError> {
    // The synthetic header names only selected tensors and rebases them into a
    // dense virtual address space. `source_intervals` below is the translation
    // table from that virtual space back to disjoint offsets in the real shard.
    let synthetic_header_bytes = source_manifest.rebased_safetensors_header()?;
    // Open a fresh descriptor for this lazy load. The runtime retains the file for
    // as long as arrays can still trigger reads, so this local variable going out
    // of scope does not invalidate deferred MLX evaluation.
    let source_file = std::fs::File::open(&source_manifest.source_file).map_err(|source| {
        ExpertManifestError::SafetensorsHeader(
            super::safetensors_header::SafetensorsHeaderError::Io(source),
        )
    })?;
    let source_intervals = source_manifest
        .source_intervals
        .iter()
        .map(
            |source_interval| astronomical_runtime_integration::BoundedReadInterval {
                virtual_payload_offset: source_interval.virtual_payload_offset,
                source_file_offset: source_interval.source_file_offset,
                source_byte_count: source_interval.source_byte_count,
            },
        )
        .collect();
    let load_result = runtime
        .load_safetensors_from_bounded_ranges(
            source_file,
            synthetic_header_bytes,
            source_intervals,
            source_manifest.payload_byte_count,
            expert_file_read_metrics,
        )
        .map_err(|source| ExpertManifestError::ReaderError {
            description: source.to_string(),
        })?;
    // Do not return every tensor the synthetic loader happens to expose. Iterate
    // the validated plan so the architecture layer receives exactly the names it
    // requested and a missing lazy array becomes a typed error at this boundary.
    source_manifest
        .tensor_ranges
        .iter()
        .map(|tensor_range| {
            load_result
                .tensor(&tensor_range.tensor_name)
                .map(|tensor| (tensor_range.tensor_name.clone(), tensor))
                .map_err(|source| ExpertManifestError::ReaderError {
                    description: source.to_string(),
                })
        })
        .collect()
}

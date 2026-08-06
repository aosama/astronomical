//! Exact selected-expert manifests for quantized Qwen3.5-MoE MoE pages.
//!
//! The oQ6e artifact has a checkpoint-specific contract: each projection is
//! represented by a packed U32 `weight` tensor plus BF16 `scales` and `biases`,
//! and a layer's projection family may be split across shard files. This module
//! keeps those source files explicit so native reads remain bounded and never
//! repack the checkpoint on disk.
//!

use std::collections::BTreeMap;
use std::path::PathBuf;

use thiserror::Error;

use super::quantized_expert_validation::{
    validate_expert_ids, validate_source_intervals, validate_virtual_intervals,
};
use super::safetensors_header::{SafetensorsDtype, SafetensorsHeaderError};
use super::source_manifests::build_source_manifests;

/// Typed failures during manifest construction and validation.
#[derive(Debug, Error)]
pub enum ExpertManifestError {
    #[error("safetensors header error: {0}")]
    SafetensorsHeader(#[from] SafetensorsHeaderError),
    #[error("quantized expert page is missing index entry for tensor {tensor_name:?}")]
    MissingTensorEntry { tensor_name: String },
    #[error("quantized expert page is missing {tensor_name:?} in shard header")]
    MissingShardTensor { tensor_name: String },
    #[error(
        "quantized expert tensor {tensor_name:?} must use {expected_dtype} but has {actual_dtype}"
    )]
    WrongDtype {
        tensor_name: String,
        expected_dtype: SafetensorsDtype,
        actual_dtype: SafetensorsDtype,
    },
    #[error(
        "quantized expert tensor {tensor_name:?} must use float16, bfloat16, or float32 but has {actual_dtype}"
    )]
    UnsupportedAffineParameterDtype {
        tensor_name: String,
        actual_dtype: SafetensorsDtype,
    },
    #[error("quantized expert tensor {tensor_name:?} must have a non-empty rank-three shape")]
    InvalidShape { tensor_name: String },
    #[error("quantized projection {projection_name:?} scales and biases must share a shape")]
    ScalesBiasesShapeMismatch { projection_name: String },
    #[error("quantized projection {projection_name:?} weight/scales batch dimensions differ")]
    WeightScalesBatchMismatch { projection_name: String },
    #[error(
        "quantized projection {projection_name:?} has an invalid packed weight width: expected {expected_packed_width}, got {actual_packed_width}"
    )]
    InvalidPackedWidth {
        projection_name: String,
        expected_packed_width: usize,
        actual_packed_width: usize,
    },
    #[error(
        "quantized expert projections must share one expert capacity, found {found_capacities:?}"
    )]
    InconsistentExpertCapacity { found_capacities: Vec<usize> },
    #[error("expert IDs must be non-negative integers")]
    NegativeExpertId,
    #[error("expert IDs must be unique and strictly ascending")]
    NonAscendingExpertIds,
    #[error("quantized expert page must select at least one expert")]
    EmptyExpertIds,
    #[error(
        "selected expert IDs exceed the quantized layer's expert capacity: max {expert_capacity}, got {max_selected_id}"
    )]
    ExpertIdExceedsCapacity {
        max_selected_id: usize,
        expert_capacity: usize,
    },
    #[error("expert page slot {page_slot} exceeds the UInt32 range")]
    PageSlotExceedsU32 { page_slot: usize },
    #[error("quantization bits must be a positive integer")]
    InvalidBits,
    #[error("quantization group_size must be a positive integer")]
    InvalidGroupSize,
    #[error(
        "quantized expert groups must pack into whole U32 elements: bits={bits}, group_size={group_size}"
    )]
    GroupsNotPackedIntoU32 { bits: i32, group_size: i32 },
    #[error("quantized expert pages currently support affine quantization only, got mode={mode:?}")]
    UnsupportedQuantizationMode { mode: String },
    #[error(
        "source intervals must not overlap: interval at offset {source_file_offset} overlaps previous"
    )]
    OverlappingSourceIntervals { source_file_offset: u64 },
    #[error(
        "source interval exceeds the source file: offset {source_file_offset} + {source_byte_count} bytes exceeds {source_file_size_bytes} bytes"
    )]
    SourceIntervalExceedsFile {
        source_file_offset: u64,
        source_byte_count: usize,
        source_file_size_bytes: u64,
    },
    #[error(
        "virtual intervals must be contiguous: expected offset {expected_virtual_offset}, got {actual_virtual_offset}"
    )]
    NonContiguousVirtualIntervals {
        expected_virtual_offset: u64,
        actual_virtual_offset: u64,
    },
    #[error(
        "virtual intervals do not cover the declared payload: expected {declared_bytes}, got {actual_bytes}"
    )]
    VirtualIntervalsShortfall {
        declared_bytes: u64,
        actual_bytes: u64,
    },
    #[error("source tensors disagree about source-file size for {source_file:?}")]
    SourceFileSizeDisagreement { source_file: PathBuf },
    #[error("reader error: {description}")]
    ReaderError { description: String },
    #[error("failed to serialize a bounded expert safetensors header: {0}")]
    HeaderSerialization(#[from] serde_json::Error),
}

/// One selected expert run copied from one original quantized tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedExpertSourceInterval {
    pub tensor_name: String,
    pub expert_start: usize,
    pub expert_count: usize,
    pub source_file_offset: u64,
    pub source_byte_count: usize,
    pub virtual_payload_offset: u64,
}

/// One compact virtual tensor for selected quantized expert parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedExpertTensorRange {
    pub tensor_name: String,
    pub projection_name: String,
    pub parameter_name: String,
    pub dtype: SafetensorsDtype,
    pub shape: Vec<usize>,
    pub virtual_payload_offset: u64,
    pub byte_count: usize,
}

/// Native-readable compact safetensors view for one source shard.
#[derive(Clone, Debug)]
pub struct QuantizedExpertShardManifest {
    pub source_file: PathBuf,
    pub tensor_ranges: Vec<QuantizedExpertTensorRange>,
    pub source_intervals: Vec<QuantizedExpertSourceInterval>,
    pub payload_byte_count: u64,
}

impl QuantizedExpertShardManifest {
    /// Serialize selected tensors with offsets relative to this shard view.
    /// Produces a complete safetensors header (8-byte length prefix + JSON)
    /// suitable for consumption by the bounded reader.
    pub fn rebased_safetensors_header(&self) -> Result<Vec<u8>, ExpertManifestError> {
        let mut header_mapping = serde_json::Map::with_capacity(self.tensor_ranges.len());
        for tensor_range in &self.tensor_ranges {
            let mut tensor_entry = serde_json::Map::new();
            tensor_entry.insert(
                "dtype".to_owned(),
                serde_json::Value::String(tensor_range.dtype.as_str().to_owned()),
            );
            tensor_entry.insert(
                "shape".to_owned(),
                serde_json::Value::Array(
                    tensor_range
                        .shape
                        .iter()
                        .map(|&d| serde_json::Value::from(d as u64))
                        .collect(),
                ),
            );
            tensor_entry.insert(
                "data_offsets".to_owned(),
                serde_json::Value::Array(vec![
                    serde_json::Value::from(tensor_range.virtual_payload_offset),
                    serde_json::Value::from(
                        tensor_range.virtual_payload_offset + tensor_range.byte_count as u64,
                    ),
                ]),
            );
            header_mapping.insert(
                tensor_range.tensor_name.clone(),
                serde_json::Value::Object(tensor_entry),
            );
        }
        let encoded_header = serde_json::to_vec(&serde_json::Value::Object(header_mapping))?;
        let mut header_bytes = Vec::with_capacity(8 + encoded_header.len());
        header_bytes.extend_from_slice(&(encoded_header.len() as u64).to_le_bytes());
        header_bytes.extend_from_slice(&encoded_header);
        Ok(header_bytes)
    }
}

/// Complete selected expert page, possibly assembled from multiple shards.
#[derive(Clone, Debug)]
pub struct QuantizedExpertPageManifest {
    pub expert_ids: Vec<usize>,
    /// Dense lookup consumed by MLX take_axis: global expert ID -> compact page slot.
    /// IDs absent from this compact page retain the u32::MAX sentinel.
    pub page_slot_by_global_expert_id: Vec<u32>,
    pub source_manifests: Vec<QuantizedExpertShardManifest>,
    pub payload_byte_count: u64,
}

/// Validated source metadata for one full quantized expert tensor.
#[derive(Clone, Debug)]
pub struct QuantizedTensorSource {
    pub tensor_name: String,
    pub projection_name: String,
    pub parameter_name: String,
    pub quantization_bits: i32,
    pub quantization_group_size: i32,
    pub source_file: PathBuf,
    pub source_file_size_bytes: u64,
    pub dtype: SafetensorsDtype,
    pub full_shape: Vec<usize>,
    pub tensor_payload_offset: u64,
    pub bytes_per_expert: usize,
    pub expert_capacity: usize,
}

/// Startup-validated tensor geometry reused by every decode-time page.
#[derive(Clone, Debug)]
pub struct QuantizedExpertLayerPlan {
    pub layer_prefix: String,
    pub tensor_sources: Vec<QuantizedTensorSource>,
    pub expert_capacity: usize,
    pub quantization_bits: i32,
    pub quantization_group_size: i32,
    pub quantization_mode: QuantizationMode,
}

/// Supported quantization modes for expert weight loading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationMode {
    Affine,
    NativeBfloat16,
}

/// Build a selected page using immutable startup-validated layer metadata.
pub fn build_quantized_expert_page_manifest_from_plan(
    layer_plan: &QuantizedExpertLayerPlan,
    expert_ids: &[usize],
) -> Result<QuantizedExpertPageManifest, ExpertManifestError> {
    let normalized_expert_ids = validate_expert_ids(expert_ids, layer_plan.expert_capacity)?;
    let source_manifests =
        build_source_manifests(&layer_plan.tensor_sources, &normalized_expert_ids)?;
    let payload_byte_count = source_manifests.iter().map(|m| m.payload_byte_count).sum();
    let page_slot_by_global_expert_id =
        build_page_slot_by_global_expert_id(&normalized_expert_ids, layer_plan.expert_capacity)?;
    Ok(QuantizedExpertPageManifest {
        expert_ids: normalized_expert_ids,
        page_slot_by_global_expert_id,
        source_manifests,
        payload_byte_count,
    })
}

/// Build a cache-population page that returns one `[1, ...]` tensor set per expert.
///
/// This manifest keeps expert tensors independent in MLX, so the cache can store
/// complete one-expert arrays without retaining lazy slices from a larger page.
/// It still batches cold I/O into one bounded safetensors load per source shard.
pub fn build_quantized_expert_cache_population_manifest_from_plan(
    layer_plan: &QuantizedExpertLayerPlan,
    expert_ids: &[usize],
) -> Result<QuantizedExpertPageManifest, ExpertManifestError> {
    let normalized_expert_ids = validate_expert_ids(expert_ids, layer_plan.expert_capacity)?;
    let mut builders_by_source_file: BTreeMap<PathBuf, CachePopulationShardManifestBuilder> =
        BTreeMap::new();

    for &expert_id in &normalized_expert_ids {
        let single_expert_manifest =
            build_quantized_expert_page_manifest_from_plan(layer_plan, &[expert_id])?;
        for source_manifest in single_expert_manifest.source_manifests {
            builders_by_source_file
                .entry(source_manifest.source_file.clone())
                .or_insert_with(|| {
                    CachePopulationShardManifestBuilder::new(source_manifest.source_file.clone())
                })
                .append_expert_manifest(expert_id, source_manifest);
        }
    }

    let source_manifests = builders_by_source_file
        .into_values()
        .map(CachePopulationShardManifestBuilder::finish)
        .collect::<Result<Vec<_>, _>>()?;
    let payload_byte_count = source_manifests
        .iter()
        .map(|source_manifest| source_manifest.payload_byte_count)
        .sum();
    let page_slot_by_global_expert_id =
        build_page_slot_by_global_expert_id(&normalized_expert_ids, layer_plan.expert_capacity)?;

    Ok(QuantizedExpertPageManifest {
        expert_ids: normalized_expert_ids,
        page_slot_by_global_expert_id,
        source_manifests,
        payload_byte_count,
    })
}

fn build_page_slot_by_global_expert_id(
    normalized_expert_ids: &[usize],
    expert_capacity: usize,
) -> Result<Vec<u32>, ExpertManifestError> {
    // Build one small, fixed-capacity lookup per loaded page instead of a HashMap
    // used once for every routed assignment. The page manifest already owns this
    // host-side metadata, so execution can upload it as one MLX array and gather
    // all assignment slots on the graphics processor.
    //
    // The sentinel makes holes explicit. Production execution verifies that the
    // sorted routed IDs exactly equal this manifest's expert IDs before building
    // the lazy gather, so an absent expert must never reach take_axis.
    let mut page_slot_by_global_expert_id = vec![u32::MAX; expert_capacity];
    for (page_slot, expert_id) in normalized_expert_ids.iter().copied().enumerate() {
        // Page tensors are assembled in normalized expert-ID order. This index is
        // therefore the first dimension consumed by gathered expert matrix work.
        page_slot_by_global_expert_id[expert_id] = u32::try_from(page_slot)
            .map_err(|_| ExpertManifestError::PageSlotExceedsU32 { page_slot })?;
    }
    Ok(page_slot_by_global_expert_id)
}

struct CachePopulationShardManifestBuilder {
    source_file: PathBuf,
    tensor_ranges: Vec<QuantizedExpertTensorRange>,
    source_intervals: Vec<QuantizedExpertSourceInterval>,
    payload_byte_count: u64,
}

impl CachePopulationShardManifestBuilder {
    fn new(source_file: PathBuf) -> Self {
        Self {
            source_file,
            tensor_ranges: Vec::new(),
            source_intervals: Vec::new(),
            payload_byte_count: 0,
        }
    }

    fn append_expert_manifest(
        &mut self,
        expert_id: usize,
        source_manifest: QuantizedExpertShardManifest,
    ) {
        let appended_payload_offset = self.payload_byte_count;
        for tensor_range in source_manifest.tensor_ranges {
            self.tensor_ranges.push(QuantizedExpertTensorRange {
                tensor_name: format!("expert_{expert_id}.{}", tensor_range.tensor_name),
                projection_name: tensor_range.projection_name,
                parameter_name: tensor_range.parameter_name,
                dtype: tensor_range.dtype,
                shape: tensor_range.shape,
                virtual_payload_offset: appended_payload_offset
                    + tensor_range.virtual_payload_offset,
                byte_count: tensor_range.byte_count,
            });
        }
        for source_interval in source_manifest.source_intervals {
            self.source_intervals.push(QuantizedExpertSourceInterval {
                tensor_name: source_interval.tensor_name,
                expert_start: source_interval.expert_start,
                expert_count: source_interval.expert_count,
                source_file_offset: source_interval.source_file_offset,
                source_byte_count: source_interval.source_byte_count,
                virtual_payload_offset: appended_payload_offset
                    + source_interval.virtual_payload_offset,
            });
        }
        self.payload_byte_count += source_manifest.payload_byte_count;
    }

    fn finish(self) -> Result<QuantizedExpertShardManifest, ExpertManifestError> {
        let mut source_intervals_sorted_by_source = self.source_intervals;
        source_intervals_sorted_by_source.sort_by_key(|interval| interval.source_file_offset);
        validate_source_intervals(&source_intervals_sorted_by_source, 0)?;
        validate_virtual_intervals(&source_intervals_sorted_by_source, self.payload_byte_count)?;
        Ok(QuantizedExpertShardManifest {
            source_file: self.source_file,
            tensor_ranges: self.tensor_ranges,
            source_intervals: source_intervals_sorted_by_source,
            payload_byte_count: self.payload_byte_count,
        })
    }
}

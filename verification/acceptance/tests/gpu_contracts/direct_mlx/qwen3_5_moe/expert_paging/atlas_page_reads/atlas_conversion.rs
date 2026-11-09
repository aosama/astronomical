//! Converts a few layers of an installed checkpoint into a test-local Expert
//! Atlas: the routed-expert tensors re-laid out expert-major, so one expert
//! is one contiguous record.
//!
//! Stock safetensors stores every expert of `gate_proj.weight` together, then
//! every expert of `gate_proj.scales`, and so on, so one routed expert is nine
//! slices spread across gigabytes of a shard. The atlas writes those nine
//! slices back to back. Records start on 16 KiB boundaries so a record read is
//! direct (uncached). Bytes are copied unchanged; nothing is decoded.

use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use astronomical_model_serving::{
    QuantizedExpertShardManifest, SafetensorsDtype, build_quantized_expert_layer_plan,
    build_quantized_expert_page_manifest_from_plan,
};

use super::super::large_sparse_moe_fixture::StreamingProbeFixture;
use super::progress_report::ProgressReporter;
use super::uncached_file_io::{
    DIRECT_READ_ALIGNMENT_BYTES, PageAlignedBuffer, available_bytes_on_volume, create_uncached,
    open_uncached, read_cold_span,
};

/// Free space kept beyond the atlas itself, so a full scratch volume fails
/// the probe up front rather than the machine's other writers.
const SCRATCH_VOLUME_RESERVE_BYTES: u64 = 8 * 1_000_000_000;

/// One of the nine tensors inside every record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AtlasSlot {
    pub(super) tensor_name_suffix: String,
    pub(super) dtype: SafetensorsDtype,
    pub(super) per_expert_shape: Vec<usize>,
    pub(super) offset_in_record: u64,
    pub(super) bytes_per_expert: u64,
}

pub(super) struct AtlasTestFile {
    pub(super) path: PathBuf,
    /// Model layer index of each atlas layer, in atlas order.
    pub(super) layer_indexes: Vec<usize>,
    pub(super) expert_count: usize,
    pub(super) record_byte_count: u64,
    pub(super) record_stride_bytes: u64,
    pub(super) slots: Vec<AtlasSlot>,
}

impl AtlasTestFile {
    /// `atlas_layer_position` indexes `layer_indexes`, not the model's layers.
    pub(super) fn record_offset(&self, atlas_layer_position: usize, expert_id: usize) -> u64 {
        (atlas_layer_position * self.expert_count + expert_id) as u64 * self.record_stride_bytes
    }
}

/// One expert-batched tensor of one layer: where its first expert starts.
struct ExpertBatchedTensorSource {
    source_file_path: PathBuf,
    tensor_name: String,
    first_expert_file_offset: u64,
    bytes_per_expert: u64,
    dtype: SafetensorsDtype,
    per_expert_shape: Vec<usize>,
}

pub(super) fn write_atlas_test_file(
    fixture: &StreamingProbeFixture,
    layer_indexes: &[usize],
    atlas_path: &Path,
) -> AtlasTestFile {
    let expert_count = fixture.config.expert_count() as usize;
    let layer_tensor_sources: Vec<Vec<ExpertBatchedTensorSource>> = layer_indexes
        .iter()
        .map(|&layer_index| expert_batched_tensor_sources(fixture, layer_index))
        .collect();
    let atlas = plan_atlas(
        atlas_path,
        layer_indexes,
        expert_count,
        &layer_tensor_sources[0],
    );
    for (layer_index, tensor_sources) in layer_indexes.iter().zip(&layer_tensor_sources) {
        assert_layer_matches_atlas_slots(*layer_index, tensor_sources, &atlas.slots);
    }
    assert_scratch_volume_fits(&atlas);
    let atlas_file = create_uncached(atlas_path);
    let mut progress = ProgressReporter::new("atlas-conversion", layer_indexes.len());
    let conversion_started_at = Instant::now();
    for (atlas_layer_position, tensor_sources) in layer_tensor_sources.iter().enumerate() {
        let mut layer_records =
            PageAlignedBuffer::new(atlas.record_stride_bytes * expert_count as u64);
        for (tensor_source, slot) in tensor_sources.iter().zip(&atlas.slots) {
            scatter_tensor_into_records(tensor_source, slot, &atlas, layer_records.bytes_mut());
        }
        atlas_file
            .write_all_at(
                layer_records.bytes_mut(),
                atlas.record_offset(atlas_layer_position, 0),
            )
            .expect("the atlas layer should write");
        progress.record(atlas_layer_position + 1, conversion_started_at);
    }
    atlas_file
        .sync_all()
        .expect("the atlas should reach storage");
    atlas
}

fn plan_atlas(
    atlas_path: &Path,
    layer_indexes: &[usize],
    expert_count: usize,
    first_layer_sources: &[ExpertBatchedTensorSource],
) -> AtlasTestFile {
    let mut offset_in_record = 0;
    let mut slots = Vec::with_capacity(first_layer_sources.len());
    for tensor_source in first_layer_sources {
        slots.push(AtlasSlot {
            tensor_name_suffix: tensor_name_suffix(&tensor_source.tensor_name),
            dtype: tensor_source.dtype,
            per_expert_shape: tensor_source.per_expert_shape.clone(),
            offset_in_record,
            bytes_per_expert: tensor_source.bytes_per_expert,
        });
        offset_in_record += tensor_source.bytes_per_expert;
    }
    AtlasTestFile {
        path: atlas_path.to_path_buf(),
        layer_indexes: layer_indexes.to_vec(),
        expert_count,
        record_byte_count: offset_in_record,
        record_stride_bytes: offset_in_record.div_ceil(DIRECT_READ_ALIGNMENT_BYTES)
            * DIRECT_READ_ALIGNMENT_BYTES,
        slots,
    }
}

/// Every layer shares the first layer's slot layout, or the records would
/// overflow into each other: a mixed-precision checkpoint (per-layer bits)
/// fails here instead of producing corrupt records.
fn assert_layer_matches_atlas_slots(
    layer_index: usize,
    tensor_sources: &[ExpertBatchedTensorSource],
    atlas_slots: &[AtlasSlot],
) {
    assert_eq!(
        tensor_sources.len(),
        atlas_slots.len(),
        "layer {layer_index} tensor count"
    );
    for (tensor_source, slot) in tensor_sources.iter().zip(atlas_slots) {
        assert_eq!(
            tensor_name_suffix(&tensor_source.tensor_name),
            slot.tensor_name_suffix,
            "layer {layer_index} slot order"
        );
        assert_eq!(
            tensor_source.dtype, slot.dtype,
            "layer {layer_index} {} dtype",
            slot.tensor_name_suffix
        );
        assert_eq!(
            tensor_source.per_expert_shape, slot.per_expert_shape,
            "layer {layer_index} {} shape",
            slot.tensor_name_suffix
        );
        assert_eq!(
            tensor_source.bytes_per_expert, slot.bytes_per_expert,
            "layer {layer_index} {} size",
            slot.tensor_name_suffix
        );
    }
}

fn assert_scratch_volume_fits(atlas: &AtlasTestFile) {
    let atlas_byte_count =
        atlas.record_stride_bytes * (atlas.expert_count * atlas.layer_indexes.len()) as u64;
    let scratch_directory = atlas.path.parent().expect("the atlas lives in a directory");
    let available_byte_count = available_bytes_on_volume(scratch_directory);
    assert!(
        available_byte_count >= atlas_byte_count + SCRATCH_VOLUME_RESERVE_BYTES,
        "the atlas needs {:.1} GB of scratch space plus {:.0} GB reserve; {:.1} GB are free",
        atlas_byte_count as f64 / 1e9,
        SCRATCH_VOLUME_RESERVE_BYTES as f64 / 1e9,
        available_byte_count as f64 / 1e9
    );
}

/// The production manifest for expert 0 alone: each of its tensors is one
/// slice that starts its tensor, so the slice's source offset is where the
/// whole expert-batched tensor begins. (A complete-layer manifest cannot
/// serve here: it merges tensors that sit back to back in a shard into one
/// read interval.)
fn expert_batched_tensor_sources(
    fixture: &StreamingProbeFixture,
    layer_index: usize,
) -> Vec<ExpertBatchedTensorSource> {
    let layer_prefix = format!("language_model.model.layers.{layer_index}.mlp");
    let layer_plan = build_quantized_expert_layer_plan(
        &fixture.model_directory,
        &fixture.weight_map,
        &layer_prefix,
        &fixture.config,
    )
    .expect("the layer plan should build");
    let first_expert_manifest = build_quantized_expert_page_manifest_from_plan(&layer_plan, &[0])
        .expect("the expert-0 manifest should build");
    let mut tensor_sources: Vec<ExpertBatchedTensorSource> = first_expert_manifest
        .source_manifests
        .iter()
        .flat_map(shard_tensor_sources)
        .collect();
    tensor_sources.sort_by_key(|tensor_source| tensor_name_suffix(&tensor_source.tensor_name));
    tensor_sources
}

fn shard_tensor_sources(
    shard_manifest: &QuantizedExpertShardManifest,
) -> Vec<ExpertBatchedTensorSource> {
    let mut tensor_sources = Vec::with_capacity(shard_manifest.tensor_ranges.len());
    for tensor_range in &shard_manifest.tensor_ranges {
        let first_slice_interval = shard_manifest
            .source_intervals
            .iter()
            .find(|interval| interval.virtual_payload_offset == tensor_range.virtual_payload_offset)
            .expect("every tensor's expert-0 slice has its own interval");
        assert_eq!(
            first_slice_interval.source_byte_count, tensor_range.byte_count,
            "an expert-0 slice is one interval"
        );
        tensor_sources.push(ExpertBatchedTensorSource {
            source_file_path: shard_manifest.source_file.clone(),
            tensor_name: tensor_range.tensor_name.clone(),
            first_expert_file_offset: first_slice_interval.source_file_offset,
            bytes_per_expert: tensor_range.byte_count as u64,
            dtype: tensor_range.dtype,
            per_expert_shape: tensor_range.shape[1..].to_vec(),
        });
    }
    tensor_sources
}

fn scatter_tensor_into_records(
    tensor_source: &ExpertBatchedTensorSource,
    slot: &AtlasSlot,
    atlas: &AtlasTestFile,
    layer_records: &mut [u8],
) {
    let source_file = open_uncached(&tensor_source.source_file_path);
    let mut expert_batched_tensor = read_cold_span(
        &source_file,
        tensor_source.first_expert_file_offset,
        tensor_source.bytes_per_expert * atlas.expert_count as u64,
    );
    let expert_batched_tensor_bytes = expert_batched_tensor.bytes();
    let bytes_per_expert = slot.bytes_per_expert as usize;
    for expert_id in 0..atlas.expert_count {
        let source_start = expert_id * bytes_per_expert;
        let destination_start =
            (expert_id as u64 * atlas.record_stride_bytes + slot.offset_in_record) as usize;
        layer_records[destination_start..destination_start + bytes_per_expert].copy_from_slice(
            &expert_batched_tensor_bytes[source_start..source_start + bytes_per_expert],
        );
    }
}

/// `...switch_mlp.gate_proj.weight` -> `gate_proj.weight`, the name shared by
/// every layer, so slot order is identical across layers.
pub(super) fn tensor_name_suffix(tensor_name: &str) -> String {
    let mut name_parts = tensor_name.rsplit('.');
    let parameter_name = name_parts.next().unwrap_or_default();
    let projection_name = name_parts.next().unwrap_or_default();
    format!("{projection_name}.{parameter_name}")
}

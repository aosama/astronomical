//! Per-expert `.apack` streaming sources wired into the expert pager.
//!
//! A converted streaming revision publishes `manifest.json` (format version 3)
//! beside one pack file per `(layer, expert)`. These tests prove that pager
//! detection validates that inventory against the startup layer plan, that page
//! manifests route reads through the per-expert files with the same compact
//! virtual layout and payload accounting as the shard path, and that a
//! manifest inconsistent with the plan geometry fails the load loudly.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use astronomical_model_serving::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype,
    StreamingExpertPackError, build_streaming_expert_page_manifest,
    detect_streaming_expert_pack_sources,
};
use serde_json::json;
use tempfile::TempDir;

const PACK_SEGMENT_ALIGNMENT_BYTES: u64 = 64 * 1024;
const PACK_HEADER_BYTES: u64 = PACK_SEGMENT_ALIGNMENT_BYTES;
const PACK_MAGIC: [u8; 8] = *b"ASTEPR01";
const EXPERT_CAPACITY: usize = 3;
const WEIGHT_BYTES_PER_EXPERT: usize = 10;
const SCALES_BYTES_PER_EXPERT: usize = 6;

/// One affine projection's weight and scales over three experts. The two
/// tensors use different byte counts so segment alignment padding is exercised.
fn synthetic_layer_plan(layer_prefix: &str) -> QuantizedExpertLayerPlan {
    let tensor_source = |projection_name: &str, parameter_name: &str, bytes_per_expert: usize| {
        QuantizedTensorSource {
            tensor_name: format!("switch_mlp.{projection_name}.{parameter_name}"),
            projection_name: projection_name.to_owned(),
            parameter_name: parameter_name.to_owned(),
            quantization_bits: 6,
            quantization_group_size: 64,
            source_file: PathBuf::from("model.safetensors"),
            source_file_size_bytes: 0,
            dtype: SafetensorsDtype::Uint32,
            full_shape: vec![EXPERT_CAPACITY, 2, 8],
            tensor_payload_offset: 0,
            bytes_per_expert,
            expert_capacity: EXPERT_CAPACITY,
        }
    };
    QuantizedExpertLayerPlan {
        layer_prefix: layer_prefix.to_owned(),
        tensor_sources: vec![
            tensor_source("gate_proj", "weight", WEIGHT_BYTES_PER_EXPERT),
            tensor_source("gate_proj", "scales", SCALES_BYTES_PER_EXPERT),
        ],
        expert_capacity: EXPERT_CAPACITY,
        quantization_bits: 6,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}

/// Segment layout derived from the plan, mirroring the preparer: 64 KiB header,
/// then one 64 KiB-aligned segment per ordered tensor. Returns per-tensor
/// `(segment_offset, segment_byte_count)` plus the full file byte count.
fn segment_layout(layer_plan: &QuantizedExpertLayerPlan) -> (Vec<(u64, u64)>, u64) {
    let mut segments = Vec::with_capacity(layer_plan.tensor_sources.len());
    let mut next_segment_offset = PACK_HEADER_BYTES;
    for tensor_source in &layer_plan.tensor_sources {
        let segment_byte_count = tensor_source.bytes_per_expert as u64;
        segments.push((next_segment_offset, segment_byte_count));
        next_segment_offset += segment_byte_count.div_ceil(PACK_SEGMENT_ALIGNMENT_BYTES)
            * PACK_SEGMENT_ALIGNMENT_BYTES;
    }
    (segments, next_segment_offset)
}

/// Writes one valid per-expert pack: 64 KiB header region with magic, header
/// payload length, and identity JSON, then aligned tensor segments carrying
/// bytes derived from the expert ID so replays can assert exact slices.
fn write_expert_pack(
    expert_file_path: &Path,
    layer_index: usize,
    expert_id: usize,
    expert_capacity: usize,
    segments: &[(u64, u64)],
    expected_file_byte_count: u64,
) {
    // The header probe schema requires the quantization fields the pager
    // cross-checks against the config-derived plan; the fixture carries the
    // same 6-bit group-64 contract the plan declares.
    let header_json = json!({
        "format_version": 3_u32,
        "layer_index": layer_index,
        "expert_id": expert_id,
        "expert_capacity": expert_capacity,
        "quantization_bits": 6,
        "quantization_group_size": 64,
    });
    let header_payload = serde_json::to_vec(&header_json).expect("header JSON");
    let mut pack_bytes = vec![0_u8; expected_file_byte_count as usize];
    pack_bytes[..8].copy_from_slice(&PACK_MAGIC);
    pack_bytes[8..16].copy_from_slice(&(header_payload.len() as u64).to_le_bytes());
    pack_bytes[16..16 + header_payload.len()].copy_from_slice(&header_payload);
    for (segment_index, &(segment_offset, segment_byte_count)) in segments.iter().enumerate() {
        for byte_index in 0..segment_byte_count {
            let value = ((expert_id * 31 + segment_index * 7 + byte_index as usize) % 251) as u8;
            pack_bytes[(segment_offset + byte_index) as usize] = value;
        }
    }
    fs::write(expert_file_path, pack_bytes).expect("write expert pack file");
}

/// Assembles one valid converted-revision directory with one decoder layer.
struct StreamingRevision {
    model_directory: TempDir,
    layer_plan: QuantizedExpertLayerPlan,
    expert_file_paths: Vec<PathBuf>,
}

fn write_streaming_revision() -> StreamingRevision {
    let model_directory = TempDir::new().expect("model directory");
    let layer_plan = synthetic_layer_plan("language_model.model.layers.0.mlp");
    let (segments, expected_file_byte_count) = segment_layout(&layer_plan);
    let mut expert_file_paths = Vec::with_capacity(EXPERT_CAPACITY);
    for expert_id in 0..EXPERT_CAPACITY {
        let expert_file_path = model_directory
            .path()
            .join(format!("layers/0/{expert_id}.apack"));
        fs::create_dir_all(expert_file_path.parent().expect("layer directory"))
            .expect("layer directory");
        write_expert_pack(
            &expert_file_path,
            0,
            expert_id,
            EXPERT_CAPACITY,
            &segments,
            expected_file_byte_count,
        );
        expert_file_paths.push(expert_file_path);
    }
    let expert_file_entries: Vec<_> = expert_file_paths
        .iter()
        .enumerate()
        .map(|(expert_id, expert_file_path)| {
            json!({
                "layer_index": 0,
                "expert_id": expert_id,
                "file_name": expert_file_path
                    .strip_prefix(model_directory.path())
                    .expect("relative expert file name"),
                "expected_file_byte_count": expected_file_byte_count,
                "content_sha256": "unused-by-detection",
            })
        })
        .collect();
    fs::write(
        model_directory.path().join("manifest.json"),
        serde_json::to_vec(&json!({
            "format_version": 3_u32,
            "model_id": "Ornith-1.5-35B-A3B-OptiQ-4bit-expert-streaming",
            "source_model_id": "Ornith-1.5-35B-A3B-OptiQ-4bit",
            "model_revision": "main",
            "expert_capacity": EXPERT_CAPACITY,
            "quantization_mode": "affine",
            "quantization_bits": 6,
            "quantization_group_size": 64,
            "resident_files": [],
            "expert_files": expert_file_entries,
        }))
        .expect("manifest JSON"),
    )
    .expect("write manifest");
    StreamingRevision {
        model_directory,
        layer_plan,
        expert_file_paths,
    }
}

#[test]
fn should_detect_a_converted_revision_and_route_pages_through_expert_files() {
    let streaming_revision = write_streaming_revision();

    let detected_sources = detect_streaming_expert_pack_sources(
        streaming_revision.model_directory.path(),
        &[streaming_revision.layer_plan.clone()],
    )
    .expect("a consistent conversion passes detection")
    .expect("a manifest means streaming sources exist");
    assert_eq!(detected_sources.decoder_layer_count(), 1);
    assert_eq!(
        detected_sources.expert_file_paths(0),
        streaming_revision.expert_file_paths
    );

    let page_manifest = build_streaming_expert_page_manifest(
        &streaming_revision.layer_plan,
        detected_sources.expert_file_paths(0),
        &[0, 2],
    )
    .expect("routed expert IDs within capacity build a page");

    // Expert IDs are normalized ascending, so page slots follow that order.
    assert_eq!(page_manifest.expert_ids, vec![0, 2]);
    assert_eq!(page_manifest.page_slot_by_global_expert_id[0], 0);
    assert_eq!(page_manifest.page_slot_by_global_expert_id[2], 1);
    let payload_byte_count = ((WEIGHT_BYTES_PER_EXPERT + SCALES_BYTES_PER_EXPERT) * 2) as u64;
    assert_eq!(page_manifest.payload_byte_count, payload_byte_count);

    // One shard manifest per routed expert file, each with the page's tensors.
    assert_eq!(page_manifest.source_manifests.len(), 2);
    for (page_slot, expert_id) in page_manifest.expert_ids.iter().enumerate() {
        let shard_manifest = &page_manifest.source_manifests[page_slot];
        assert_eq!(
            shard_manifest.source_file,
            streaming_revision.expert_file_paths[*expert_id]
        );
        // Tensor-major compact layout within each file's own virtual space:
        // weight first, scales right after; the interval points at the aligned
        // segment inside this expert file, and page-slot ordering comes from the
        // assembly concatenation order, not the per-file virtual offsets.
        let weight_interval = &shard_manifest.source_intervals[0];
        let scales_interval = &shard_manifest.source_intervals[1];
        assert_eq!(weight_interval.source_file_offset, PACK_HEADER_BYTES);
        assert_eq!(weight_interval.source_byte_count, WEIGHT_BYTES_PER_EXPERT);
        assert_eq!(weight_interval.virtual_payload_offset, 0);
        assert_eq!(
            scales_interval.source_file_offset,
            PACK_HEADER_BYTES + PACK_SEGMENT_ALIGNMENT_BYTES
        );
        assert_eq!(scales_interval.source_byte_count, SCALES_BYTES_PER_EXPERT);
        assert_eq!(
            scales_interval.virtual_payload_offset,
            weight_interval.virtual_payload_offset + WEIGHT_BYTES_PER_EXPERT as u64
        );
        assert_eq!(shard_manifest.payload_byte_count, payload_byte_count / 2);
    }
}

#[test]
fn should_route_expert_pack_intervals_to_the_exact_expert_slice_bytes() {
    let streaming_revision = write_streaming_revision();
    let detected_sources = detect_streaming_expert_pack_sources(
        streaming_revision.model_directory.path(),
        &[streaming_revision.layer_plan.clone()],
    )
    .expect("consistent revision")
    .expect("streaming sources");
    let page_manifest = build_streaming_expert_page_manifest(
        &streaming_revision.layer_plan,
        detected_sources.expert_file_paths(0),
        &[1],
    )
    .expect("single-expert page");

    let shard_manifest = &page_manifest.source_manifests[0];
    assert_eq!(
        shard_manifest.source_file,
        streaming_revision.expert_file_paths[1]
    );
    let mut pack_file = fs::File::open(&shard_manifest.source_file).expect("expert pack");
    for (segment_index, source_interval) in shard_manifest.source_intervals.iter().enumerate() {
        let mut read_bytes = vec![0_u8; source_interval.source_byte_count];
        pack_file
            .seek(SeekFrom::Start(source_interval.source_file_offset))
            .expect("seek to segment");
        pack_file.read_exact(&mut read_bytes).expect("read segment");
        for (byte_index, &read_byte) in read_bytes.iter().enumerate() {
            let expected_byte = ((1 * 31 + segment_index * 7 + byte_index) % 251) as u8;
            assert_eq!(read_byte, expected_byte, "expert slice byte mismatch");
        }
    }
}

#[test]
fn should_fall_back_to_the_shard_path_without_a_streaming_manifest() {
    let streaming_revision = write_streaming_revision();
    fs::remove_file(
        streaming_revision
            .model_directory
            .path()
            .join("manifest.json"),
    )
    .expect("remove manifest");

    let detected_sources = detect_streaming_expert_pack_sources(
        streaming_revision.model_directory.path(),
        &[streaming_revision.layer_plan],
    )
    .expect("a directory without manifest keeps the shard path");
    assert!(detected_sources.is_none());
}

#[test]
fn should_fail_loudly_when_a_declared_expert_file_size_disagrees_with_the_plan() {
    let streaming_revision = write_streaming_revision();
    let manifest_path = streaming_revision
        .model_directory
        .path()
        .join("manifest.json");
    let mut manifest_json: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).expect("manifest bytes"))
            .expect("valid manifest JSON");
    manifest_json["expert_files"][0]["expected_file_byte_count"] = json!(1_u64);
    fs::write(
        &manifest_path,
        serde_json::to_vec(&manifest_json).expect("manifest JSON"),
    )
    .expect("rewrite manifest");

    let detection_error = detect_streaming_expert_pack_sources(
        streaming_revision.model_directory.path(),
        &[streaming_revision.layer_plan],
    )
    .expect_err("a size disagreement must fail the load");
    assert!(matches!(
        detection_error,
        StreamingExpertPackError::ExpertFileSizeDisagreement { .. }
    ));
}

#[test]
fn should_fail_loudly_when_an_expert_file_does_not_carry_the_pack_magic() {
    let streaming_revision = write_streaming_revision();
    let corrupted_pack_path = &streaming_revision.expert_file_paths[0];
    let mut corrupted_bytes = fs::read(corrupted_pack_path).expect("read pack");
    // Corrupt the magic in place so the file size gate cannot mask the header
    // check: only the first eight bytes change.
    corrupted_bytes[..8].copy_from_slice(b"XXXXXXXX");
    fs::write(corrupted_pack_path, &corrupted_bytes).expect("write corrupted pack");

    let detection_error = detect_streaming_expert_pack_sources(
        streaming_revision.model_directory.path(),
        &[streaming_revision.layer_plan],
    )
    .expect_err("corrupted pack magic must fail the load");
    assert!(matches!(
        detection_error,
        StreamingExpertPackError::PackHeaderMagic { .. }
    ));
}

#[test]
fn should_fail_loudly_when_the_manifest_is_missing_one_expert_file() {
    let streaming_revision = write_streaming_revision();
    fs::remove_file(&streaming_revision.expert_file_paths[2]).expect("remove expert file");

    let detection_error = detect_streaming_expert_pack_sources(
        streaming_revision.model_directory.path(),
        &[streaming_revision.layer_plan],
    )
    .expect_err("missing expert files must fail the load");
    assert!(matches!(
        detection_error,
        StreamingExpertPackError::ExpertFileOpen { .. }
    ));
}

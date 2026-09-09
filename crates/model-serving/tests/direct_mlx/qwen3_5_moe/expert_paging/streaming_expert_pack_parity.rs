//! Byte parity between the shard-path page manifest and the per-expert pack
//! page manifest for the same routed experts.
//!
//! A converted streaming revision must produce page arrays indistinguishable
//! from the shard path: same tensor names, same shapes, same evaluated bytes.
//! A rank or slot-order difference here would surface at serving time as a
//! fatal gather_qmm dimension mismatch, so both expert-count regimes are
//! pinned: a multi-expert page and the single-expert page where the shard
//! path still carries a leading page-slot axis of one.

use std::fs;
use std::path::PathBuf;

use astronomical_model_serving::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype,
    assemble_streaming_expert_page_tensors, build_quantized_expert_page_manifest_from_plan,
    build_streaming_expert_page_manifest, load_quantized_expert_page,
};
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};
use tempfile::TempDir;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const PACK_SEGMENT_ALIGNMENT_BYTES: u64 = 64 * 1024;
const PACK_HEADER_BYTES: u64 = PACK_SEGMENT_ALIGNMENT_BYTES;
const PACK_MAGIC: [u8; 8] = *b"ASTEPR01";
const EXPERT_CAPACITY: usize = 4;
const PACKED_WIDTH: usize = 8;
const GROUP_COUNT: usize = 4;
const WEIGHT_BYTES_PER_EXPERT: usize = PACKED_WIDTH * 4;
const SCALES_BYTES_PER_EXPERT: usize = GROUP_COUNT * 4;

/// Deterministic byte for one expert's tensor slice.
fn parity_byte(expert_id: usize, segment_index: usize, byte_index: u64) -> u8 {
    ((expert_id * 31 + segment_index * 7 + byte_index as usize) % 251) as u8
}

fn parity_layer_plan(shard_path: &std::path::Path) -> QuantizedExpertLayerPlan {
    let make_source = |parameter_name: &str, bytes_per_expert: usize| QuantizedTensorSource {
        tensor_name: format!("switch_mlp.gate_proj.{parameter_name}"),
        projection_name: "gate_proj".to_owned(),
        parameter_name: parameter_name.to_owned(),
        quantization_bits: 4,
        quantization_group_size: 64,
        source_file: shard_path.to_path_buf(),
        source_file_size_bytes: 0,
        dtype: if parameter_name == "weight" {
            SafetensorsDtype::Uint32
        } else {
            SafetensorsDtype::Float32
        },
        full_shape: if parameter_name == "weight" {
            vec![EXPERT_CAPACITY, PACKED_WIDTH]
        } else {
            vec![EXPERT_CAPACITY, GROUP_COUNT]
        },
        tensor_payload_offset: if parameter_name == "weight" {
            0
        } else {
            (WEIGHT_BYTES_PER_EXPERT * EXPERT_CAPACITY) as u64
        },
        bytes_per_expert,
        expert_capacity: EXPERT_CAPACITY,
    };
    QuantizedExpertLayerPlan {
        layer_prefix: "language_model.model.layers.0.mlp".to_owned(),
        tensor_sources: vec![
            make_source("weight", WEIGHT_BYTES_PER_EXPERT),
            make_source("scales", SCALES_BYTES_PER_EXPERT),
        ],
        expert_capacity: EXPERT_CAPACITY,
        quantization_bits: 4,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}

struct ParityFixture {
    _model_directory: TempDir,
    layer_plan: QuantizedExpertLayerPlan,
    expert_pack_paths: Vec<PathBuf>,
}

/// Writes the shard in expert-major layout (matching the plan's tensor
/// payload offsets and expert strides) plus one valid per-expert pack file
/// carrying the same slice bytes behind a 64 KiB header.
fn write_parity_fixture() -> ParityFixture {
    let model_directory = TempDir::new().expect("model directory");
    let shard_path = model_directory.path().join("model.safetensors");

    let mut shard_bytes = Vec::new();
    for expert_id in 0..EXPERT_CAPACITY {
        for byte_index in 0..WEIGHT_BYTES_PER_EXPERT as u64 {
            shard_bytes.push(parity_byte(expert_id, 0, byte_index));
        }
    }
    for expert_id in 0..EXPERT_CAPACITY {
        for byte_index in 0..SCALES_BYTES_PER_EXPERT as u64 {
            shard_bytes.push(parity_byte(expert_id, 1, byte_index));
        }
    }
    fs::write(&shard_path, &shard_bytes).expect("write shard fixture");

    let mut expert_pack_paths = Vec::with_capacity(EXPERT_CAPACITY);
    for expert_id in 0..EXPERT_CAPACITY {
        let expert_file_path = model_directory.path().join(format!("{expert_id}.apack"));
        let header_json = serde_json::json!({
            "format_version": 3_u32,
            "layer_index": 0_usize,
            "expert_id": expert_id,
            "expert_capacity": EXPERT_CAPACITY,
        });
        let header_payload =
            serde_json::to_vec(&header_json).expect("pack header JSON should serialize");
        let expected_file_byte_count = PACK_HEADER_BYTES + 2 * PACK_SEGMENT_ALIGNMENT_BYTES;
        let mut pack_bytes = vec![0_u8; expected_file_byte_count as usize];
        pack_bytes[..8].copy_from_slice(&PACK_MAGIC);
        pack_bytes[8..16].copy_from_slice(&(header_payload.len() as u64).to_le_bytes());
        pack_bytes[16..16 + header_payload.len()].copy_from_slice(&header_payload);
        for byte_index in 0..WEIGHT_BYTES_PER_EXPERT as u64 {
            pack_bytes[PACK_HEADER_BYTES as usize + byte_index as usize] =
                parity_byte(expert_id, 0, byte_index);
        }
        let scales_segment_offset = PACK_HEADER_BYTES + PACK_SEGMENT_ALIGNMENT_BYTES;
        for byte_index in 0..SCALES_BYTES_PER_EXPERT as u64 {
            pack_bytes[scales_segment_offset as usize + byte_index as usize] =
                parity_byte(expert_id, 1, byte_index);
        }
        fs::write(&expert_file_path, &pack_bytes).expect("write expert pack fixture");
        expert_pack_paths.push(expert_file_path);
    }

    let layer_plan = parity_layer_plan(&shard_path);
    ParityFixture {
        _model_directory: model_directory,
        layer_plan,
        expert_pack_paths,
    }
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("direct MLX test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn assert_parity(routed_expert_ids: &[usize]) {
    let fixture = write_parity_fixture();
    let runtime = test_runtime();

    let shard_page_manifest =
        build_quantized_expert_page_manifest_from_plan(&fixture.layer_plan, routed_expert_ids)
            .expect("shard page manifest should build from the validated plan");
    let shard_tensors = load_quantized_expert_page(&runtime, &shard_page_manifest, None)
        .expect("shard page should load from bounded ranges");

    let pack_page_manifest = build_streaming_expert_page_manifest(
        &fixture.layer_plan,
        &fixture.expert_pack_paths,
        routed_expert_ids,
    )
    .expect("pack page manifest should build from the validated plan");
    let mut pack_loaded_tensors = load_quantized_expert_page(&runtime, &pack_page_manifest, None)
        .expect("pack page should load from bounded ranges");
    let pack_tensors = assemble_streaming_expert_page_tensors(
        &runtime,
        &mut pack_loaded_tensors,
        &fixture.layer_plan,
        routed_expert_ids.len(),
    )
    .expect("pack page should assemble into canonical tensors");

    assert_eq!(
        shard_tensors.len(),
        pack_tensors.len(),
        "both paths must expose the same tensor set"
    );
    for (tensor_name, shard_array) in &shard_tensors {
        let pack_array = pack_tensors
            .get(tensor_name)
            .unwrap_or_else(|| panic!("pack path is missing {tensor_name}"));
        assert_eq!(
            shard_array.shape(),
            pack_array.shape(),
            "tensor {tensor_name} shape must match between shard and pack pages"
        );
        match shard_array.dtype() {
            MlxDtype::UInt32 => {
                let shard_values = shard_array
                    .to_vec_u32()
                    .expect("shard page weight should evaluate");
                let pack_values = pack_array
                    .to_vec_u32()
                    .expect("pack page weight should evaluate");
                assert_eq!(
                    shard_values, pack_values,
                    "tensor {tensor_name} values must be bit-identical"
                );
            }
            MlxDtype::Float32 => {
                let shard_values = shard_array
                    .to_vec_f32()
                    .expect("shard page scales should evaluate");
                let pack_values = pack_array
                    .to_vec_f32()
                    .expect("pack page scales should evaluate");
                assert_eq!(
                    shard_values, pack_values,
                    "tensor {tensor_name} values must be bit-identical"
                );
            }
            dtype => panic!("parity fixture did not expect dtype {dtype:?}"),
        }
    }
}

#[test]
fn should_produce_bit_identical_multi_expert_pages_from_packs_and_shards() {
    assert_parity(&[1, 3]);
}

#[test]
fn should_produce_bit_identical_single_expert_pages_from_packs_and_shards() {
    assert_parity(&[2]);
}

use std::path::PathBuf;

use crate::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedExpertPageManifest, QuantizedTensorSource,
    SafetensorsDtype,
};

fn retained_page_manifest() -> QuantizedExpertPageManifest {
    QuantizedExpertPageManifest {
        expert_ids: vec![1, 3],
        page_slot_by_global_expert_id: vec![u32::MAX, 0, u32::MAX, 1, u32::MAX],
        source_manifests: Vec::new(),
        payload_byte_count: 200,
    }
}

#[test]
fn should_identify_only_routed_experts_missing_from_a_retained_page() {
    let retained_page_manifest = retained_page_manifest();

    assert_eq!(
        retained_page_manifest.missing_expert_ids(&[0, 1, 3, 4]),
        vec![0, 4]
    );
}

#[test]
fn should_report_complete_route_coverage_for_a_retained_page() {
    let retained_page_manifest = retained_page_manifest();

    assert!(retained_page_manifest.contains_every_expert(&[1, 3]));
    assert!(!retained_page_manifest.contains_every_expert(&[1, 2, 3]));
}

#[test]
fn should_distinguish_complete_and_partial_expert_pages() {
    let partial_page_manifest = retained_page_manifest();
    let complete_page_manifest = QuantizedExpertPageManifest {
        expert_ids: vec![0, 1, 2],
        page_slot_by_global_expert_id: vec![0, 1, 2],
        source_manifests: Vec::new(),
        payload_byte_count: 300,
    };

    assert!(!partial_page_manifest.contains_all_experts());
    assert!(complete_page_manifest.contains_all_experts());
}

#[test]
fn should_partition_route_assignments_without_duplicate_execution() {
    let retained_page_manifest = retained_page_manifest();

    let route_partition = retained_page_manifest.partition_route_assignments(&[3, 0, 1, 4, 3, 1]);

    assert_eq!(
        route_partition.retained_assignment_positions,
        vec![0, 2, 4, 5]
    );
    assert_eq!(route_partition.retained_expert_ids, vec![1, 3]);
    assert_eq!(route_partition.missing_assignment_positions, vec![1, 3]);
    assert_eq!(route_partition.missing_expert_ids, vec![0, 4]);
    assert_eq!(
        route_partition.retained_assignment_positions.len()
            + route_partition.missing_assignment_positions.len(),
        6
    );
}

#[test]
fn should_derive_exact_per_expert_and_complete_layer_payload_from_tensor_geometry() {
    let tensor_source = |tensor_name: &str, bytes_per_expert: usize| QuantizedTensorSource {
        tensor_name: tensor_name.to_owned(),
        projection_name: "fictional_projection".to_owned(),
        parameter_name: "weight".to_owned(),
        quantization_bits: 4,
        quantization_group_size: 64,
        source_file: PathBuf::from("fictional-model.safetensors"),
        source_file_size_bytes: 1_000,
        dtype: SafetensorsDtype::Uint32,
        full_shape: vec![4, 2, 2],
        tensor_payload_offset: 0,
        bytes_per_expert,
        expert_stride_bytes: bytes_per_expert,
        expert_capacity: 4,
    };
    let layer_plan = QuantizedExpertLayerPlan {
        layer_prefix: "fictional.layers.0".to_owned(),
        tensor_sources: vec![
            tensor_source("gate.weight", 10),
            tensor_source("up.weight", 5),
        ],
        jangtq2_projection_rotations: Vec::new(),
        expert_capacity: 4,
        quantization_bits: 4,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    };

    assert_eq!(
        layer_plan
            .expert_payload_byte_count()
            .expect("fictional exact expert geometry should be valid"),
        15
    );
    assert_eq!(
        layer_plan
            .complete_expert_payload_byte_count()
            .expect("fictional complete geometry should agree"),
        60
    );
}

/// One 100-byte-per-expert tensor of eight experts, stored either back to
/// back (a stock shard) or one 4096-byte record apart (an Atlas layer file).
fn eight_expert_tensor_source(
    source_file: &str,
    expert_stride_bytes: usize,
) -> QuantizedTensorSource {
    QuantizedTensorSource {
        tensor_name: "layers.0.mlp.switch_mlp.gate_proj.weight".to_owned(),
        projection_name: "gate_proj".to_owned(),
        parameter_name: "weight".to_owned(),
        quantization_bits: 4,
        quantization_group_size: 64,
        source_file: PathBuf::from(source_file),
        source_file_size_bytes: 1 << 20,
        dtype: SafetensorsDtype::Uint32,
        full_shape: vec![8, 25],
        tensor_payload_offset: 512,
        bytes_per_expert: 100,
        expert_stride_bytes,
        expert_capacity: 8,
    }
}

fn source_interval_triples(
    tensor_source: &QuantizedTensorSource,
    expert_ids: &[usize],
) -> Vec<(usize, u64, usize, u64)> {
    let source_manifests = super::source_manifests::build_source_manifests(
        std::slice::from_ref(tensor_source),
        expert_ids,
    )
    .expect("the manifest builds");
    source_manifests[0]
        .source_intervals
        .iter()
        .map(|interval| {
            (
                interval.expert_start,
                interval.source_file_offset,
                interval.source_byte_count,
                interval.virtual_payload_offset,
            )
        })
        .collect()
}

#[test]
fn should_merge_adjacent_experts_into_one_read_in_a_stock_shard() {
    let stock_source = eight_expert_tensor_source("model-00001.safetensors", 100);
    assert_eq!(
        source_interval_triples(&stock_source, &[2, 3, 4, 7]),
        vec![(2, 712, 300, 0), (7, 1_212, 100, 300)]
    );
}

#[test]
fn should_read_each_atlas_expert_from_its_own_record() {
    let atlas_source = eight_expert_tensor_source("experts/layer-000.atlas", 4_096);
    assert_eq!(
        source_interval_triples(&atlas_source, &[2, 3, 7]),
        vec![
            (2, 512 + 2 * 4_096, 100, 0),
            (3, 512 + 3 * 4_096, 100, 100),
            (7, 512 + 7 * 4_096, 100, 200)
        ]
    );
}

#[test]
fn should_read_a_single_atlas_expert_from_its_record_at_the_page_start() {
    let atlas_source = eight_expert_tensor_source("experts/layer-000.atlas", 4_096);
    assert_eq!(
        source_interval_triples(&atlas_source, &[0]),
        vec![(0, 512, 100, 0)]
    );
}

#[test]
fn should_mark_only_atlas_layer_files_as_atlas_sources() {
    assert!(eight_expert_tensor_source("experts/layer-000.atlas", 4_096).is_expert_atlas_source());
    assert!(!eight_expert_tensor_source("model-00001.safetensors", 100).is_expert_atlas_source());
    assert!(!eight_expert_tensor_source("resident.safetensors", 4_096).is_expert_atlas_source());
}

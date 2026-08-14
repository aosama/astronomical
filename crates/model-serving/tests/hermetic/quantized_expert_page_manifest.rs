use std::path::PathBuf;

use astronomical_model_serving::{
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
        expert_capacity: 4,
    };
    let layer_plan = QuantizedExpertLayerPlan {
        layer_prefix: "fictional.layers.0".to_owned(),
        tensor_sources: vec![
            tensor_source("gate.weight", 10),
            tensor_source("up.weight", 5),
        ],
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

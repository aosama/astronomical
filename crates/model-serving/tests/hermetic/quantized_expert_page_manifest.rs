use astronomical_model_serving::QuantizedExpertPageManifest;

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

use crate::budget::ram_values::MlxRamBudgetModelGeometry;

#[test]
fn should_derive_the_geometry_floor_from_core_context_and_one_routed_page() {
    let model_geometry = MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 2_000_000_000,
        complete_expert_payload_bytes: 18_000_000_000,
        largest_complete_expert_layer_bytes: 450_000_000,
        largest_routed_expert_page_bytes: 90_000_000,
        sequence_state_bytes_per_token: 320,
    };

    assert_eq!(
        crate::geometry_floor::geometry_safe_minimum_mlx_memory_ceiling_bytes(&model_geometry),
        2_000_000_000 + 1_000_000_000 + 450_000_000 + 90_000_000
    );
}

#[test]
fn should_keep_a_dense_model_floor_below_its_core_plus_reserve() {
    let dense_model_geometry = MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 3_000_000_000,
        complete_expert_payload_bytes: 0,
        largest_complete_expert_layer_bytes: 0,
        largest_routed_expert_page_bytes: 0,
        sequence_state_bytes_per_token: 160,
    };

    assert_eq!(
        crate::geometry_floor::geometry_safe_minimum_mlx_memory_ceiling_bytes(
            &dense_model_geometry
        ),
        3_000_000_000 + 1_000_000_000
    );
}

#[test]
fn should_ignore_disk_scale_of_fully_paged_experts_in_the_floor() {
    // The floor must describe what the NEXT request needs under streaming, not
    // the residency a prior ceiling happened to produce. A 20 GB-on-disk
    // sparse artifact whose core is small must therefore report a small floor.
    let paged_model_geometry = MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 1_400_000_000,
        complete_expert_payload_bytes: 19_000_000_000,
        largest_complete_expert_layer_bytes: 475_000_000,
        largest_routed_expert_page_bytes: 95_000_000,
        sequence_state_bytes_per_token: 0,
    };

    let geometry_floor_bytes =
        crate::geometry_floor::geometry_safe_minimum_mlx_memory_ceiling_bytes(
            &paged_model_geometry,
        );

    assert!(geometry_floor_bytes < 3_500_000_000);
}

#[test]
fn should_saturate_instead_of_overflowing_on_hostile_geometry() {
    let saturated_model_geometry = MlxRamBudgetModelGeometry {
        model_core_payload_bytes: u64::MAX,
        complete_expert_payload_bytes: 0,
        largest_complete_expert_layer_bytes: u64::MAX,
        largest_routed_expert_page_bytes: u64::MAX,
        sequence_state_bytes_per_token: 0,
    };

    assert_eq!(
        crate::geometry_floor::geometry_safe_minimum_mlx_memory_ceiling_bytes(
            &saturated_model_geometry
        ),
        u64::MAX
    );
}

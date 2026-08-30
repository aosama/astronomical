use astronomical_model_serving::{
    CompleteResidencyDecision, CompleteResidencyHeadroomBoundary, CompleteResidencyRequirements,
    MemoryBoundary, MlxRamBudgetModelGeometry,
};

fn sparse_geometry() -> MlxRamBudgetModelGeometry {
    MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 100,
        complete_expert_payload_bytes: 1_000,
        largest_complete_expert_layer_bytes: 50,
        largest_routed_expert_page_bytes: 5,
        sequence_state_bytes_per_token: 0,
    }
}

#[test]
fn should_return_no_paging_ceiling_when_the_artifact_has_no_headroom_gap() {
    let geometry_without_experts = MlxRamBudgetModelGeometry {
        complete_expert_payload_bytes: 0,
        ..sparse_geometry()
    };
    assert_eq!(
        CompleteResidencyHeadroomBoundary::from_model_geometry(geometry_without_experts, 40)
            .paging_ceiling_bytes(),
        None
    );
    assert_eq!(
        CompleteResidencyHeadroomBoundary::from_model_geometry(sparse_geometry(), 0)
            .paging_ceiling_bytes(),
        None
    );
}

#[test]
fn should_place_the_paging_ceiling_where_static_weights_fit_and_headroom_does_not() {
    let required_headroom_bytes = 40;
    let boundary = CompleteResidencyHeadroomBoundary::from_model_geometry(
        sparse_geometry(),
        required_headroom_bytes,
    );
    let paging_ceiling_bytes = boundary
        .paging_ceiling_bytes()
        .expect("a sparse artifact with headroom must expose a paging ceiling");
    assert_eq!(paging_ceiling_bytes, 1_039);

    let rejected = CompleteResidencyRequirements {
        current_active_memory_bytes: 100,
        retained_paged_expert_payload_bytes: 0,
        complete_expert_payload_bytes: 1_000,
        required_headroom_bytes,
        active_memory_ceiling_bytes: paging_ceiling_bytes,
    }
    .decide();
    assert!(matches!(
        rejected,
        CompleteResidencyDecision::DoesNotFit {
            boundary: MemoryBoundary::CompleteResidency,
            ..
        }
    ));

    let admitted = CompleteResidencyRequirements {
        current_active_memory_bytes: 100,
        retained_paged_expert_payload_bytes: 0,
        complete_expert_payload_bytes: 1_000,
        required_headroom_bytes,
        active_memory_ceiling_bytes: 100 + 1_000 + required_headroom_bytes,
    }
    .decide();
    assert!(matches!(admitted, CompleteResidencyDecision::Admit { .. }));
}

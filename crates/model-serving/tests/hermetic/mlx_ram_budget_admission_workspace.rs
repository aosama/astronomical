//! Admission-composition regression coverage for the learned activation reserve.
//!
//! The measured production failure (2026-10-10): the 36.83 GB Ornith-8-bit
//! artifact under a 32 GB ceiling taught the RAM budget a ~9.9 GB chunk-shaped
//! activation observation at the 2,048-token operation bucket, and the
//! follow-up turn's admission inflated it by the chunk ratio to the
//! 8,192-token operation bound, capped it at the whole ceiling, and rejected
//! every later request with `generation context exceeds available GPU wired
//! memory` until the model reloaded. These tests pin the numbers admission
//! actually charges so the rejection cascade cannot creep back in.

use astronomical_model_serving::{
    ContextAdmissionRequirements, MemoryAdmissionDecision, MemoryPhase, MlxRamBudget,
    MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry,
};

/// A geometry whose static three-layer floor is small enough that learned
/// evidence dominates the promise, mirroring the production artifact shape.
fn production_failure_geometry() -> MlxRamBudgetModelGeometry {
    MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 2_360_000_000,
        complete_expert_payload_bytes: 36_238_786_560,
        largest_complete_expert_layer_bytes: 10_000_000,
        largest_routed_expert_page_bytes: 28_311_552,
        sequence_state_bytes_per_token: 20_480,
    }
}

#[test]
fn should_reclaim_instead_of_rejecting_the_follow_up_context_after_chunk_shaped_activation_evidence()
 {
    let mut mlx_ram_budget = MlxRamBudget::new(32_000_000_000, production_failure_geometry())
        .expect("positive ceiling should construct");
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MemoryPhase::Prefill,
        context_token_count: 2_048,
        measured_context_and_activation_bytes: 9_900_000_000,
        observed_activation_headroom_bytes: 9_000_000_000,
        exact_temporary_workspace_bytes: 0,
    });

    // The paged follow-up turn plans its activation reserve at the
    // 8,192-token SSD-streaming operation bound. The workspace must stay at
    // the measured observation; scaling it by the chunk ratio would cap it at
    // the whole ceiling, and workspace == ceiling makes every nonzero current
    // active memory project above the ceiling — the rejection cascade.
    let admission_workspace =
        mlx_ram_budget.context_admission_workspace_snapshot(50_591, 8_192, 0, 0);
    let follow_up_requirements = ContextAdmissionRequirements {
        current_active_memory_bytes: 28_909_997_702,
        context_growth_bytes: 1_036_103_680,
        expert_page_reservation_bytes: 855_638_016,
        temporary_workspace_bytes: usize::try_from(admission_workspace.activation_headroom_bytes)
            .unwrap_or(usize::MAX),
        retained_expert_payload_bytes: 26_307_526_656,
        active_memory_ceiling_bytes: 32_000_000_000,
        complete_experts_are_resident: false,
    };

    assert_eq!(
        follow_up_requirements.temporary_workspace_bytes, 9_000_000_000,
        "the reserve must stay at the measured observation instead of inflating toward the ceiling"
    );
    assert_eq!(
        follow_up_requirements.decide(),
        MemoryAdmissionDecision::Reclaim {
            required_bytes: 7_801_739_398
        },
        "the follow-up turn must be servable by reclaiming retained experts, never rejected"
    );
}

#[test]
fn should_resolve_the_follow_up_reserve_from_evidence_at_the_operation_scope() {
    let mut mlx_ram_budget = MlxRamBudget::new(32_000_000_000, production_failure_geometry())
        .expect("positive ceiling should construct");
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MemoryPhase::Prefill,
        context_token_count: 2_048,
        measured_context_and_activation_bytes: 9_900_000_000,
        observed_activation_headroom_bytes: 9_000_000_000,
        exact_temporary_workspace_bytes: 0,
    });
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MemoryPhase::Prefill,
        context_token_count: 8_192,
        measured_context_and_activation_bytes: 3_400_000_000,
        observed_activation_headroom_bytes: 3_000_000_000,
        exact_temporary_workspace_bytes: 0,
    });

    // Once the 8,192-token scope carries its own measured evidence, the
    // follow-up promise resolves from the honest at-or-below maximum: the
    // smaller-scope observation must not override the larger known lower
    // bucket, and neither value may be multiplied further.
    let admission_workspace =
        mlx_ram_budget.context_admission_workspace_snapshot(50_591, 8_192, 0, 0);

    assert_eq!(admission_workspace.activation_headroom_bytes, 9_000_000_000);
}

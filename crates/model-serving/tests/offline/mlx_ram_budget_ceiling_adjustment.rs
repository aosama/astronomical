//! Learned-evidence behavior across live MLX memory-ceiling changes.
//!
//! Activation observations embed the ceiling regime they were measured under:
//! a lowered ceiling's expert demote/stream churn inflates the observed
//! transients into a permanently over-sized reserve (issue #1108). A ceiling
//! change therefore clears activation evidence while persistent context-window
//! evidence survives, because KV bytes per token do not depend on the ceiling.

use astronomical_model_serving::{
    MemoryPhase, MlxRamBudget, MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry,
};

fn small_layer_geometry() -> MlxRamBudgetModelGeometry {
    MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 2_360_000_000,
        complete_expert_payload_bytes: 36_238_786_560,
        largest_complete_expert_layer_bytes: 10_000_000,
        largest_routed_expert_page_bytes: 28_311_552,
        sequence_state_bytes_per_token: 0,
    }
}

#[test]
fn should_forget_learned_activation_evidence_when_the_live_ceiling_changes() {
    let mut mlx_ram_budget = MlxRamBudget::new(39_000_000_000, small_layer_geometry())
        .expect("positive ceiling should construct");
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MemoryPhase::Prefill,
        context_token_count: 4_096,
        measured_context_and_activation_bytes: 2_000_000_000,
        observed_activation_headroom_bytes: 0,
        exact_temporary_workspace_bytes: 0,
    });
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MemoryPhase::Prefill,
        context_token_count: 4_096,
        measured_context_and_activation_bytes: 6_000_000_000,
        observed_activation_headroom_bytes: 6_000_000_000,
        exact_temporary_workspace_bytes: 0,
    });
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MemoryPhase::Decode,
        context_token_count: 4_096,
        measured_context_and_activation_bytes: 500_000_000,
        observed_activation_headroom_bytes: 500_000_000,
        exact_temporary_workspace_bytes: 0,
    });

    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Prefill, 4_096),
        6_000_000_000
    );
    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Decode, 1),
        500_000_000
    );

    mlx_ram_budget
        .update_mlx_active_memory_ceiling_bytes(19_000_000_000)
        .expect("positive ceiling should update");

    // The lowered ceiling's paging churn inflated the learned transients; after
    // the change only the static floors remain until fresh forwards re-learn
    // under the new regime (issue #1108).
    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Prefill, 4_096),
        3 * small_layer_geometry().largest_complete_expert_layer_bytes,
    );
    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Decode, 1),
        small_layer_geometry().largest_complete_expert_layer_bytes,
    );
    // Persistent KV bytes per token do not depend on the ceiling, so
    // context-window evidence survives the change.
    assert!(mlx_ram_budget.has_context_window_measurement());
    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(4_096),
        2_000_000_000 + 64_000_000,
    );
}

#[test]
fn should_keep_learned_activation_evidence_when_the_ceiling_value_is_unchanged() {
    let mut mlx_ram_budget = MlxRamBudget::new(19_000_000_000, small_layer_geometry())
        .expect("positive ceiling should construct");
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MemoryPhase::Prefill,
        context_token_count: 4_096,
        measured_context_and_activation_bytes: 6_000_000_000,
        observed_activation_headroom_bytes: 6_000_000_000,
        exact_temporary_workspace_bytes: 0,
    });

    mlx_ram_budget
        .update_mlx_active_memory_ceiling_bytes(19_000_000_000)
        .expect("same-value ceiling update should succeed");

    // A same-value write (for example a failed-raise restore path) is not a
    // regime change, so valid evidence from forwards already measured under
    // this ceiling stays.
    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Prefill, 4_096),
        6_000_000_000
    );
}

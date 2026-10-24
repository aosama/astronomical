//! Boundary tests for the RAM budget owner's learned context reserve,
//! activation coexistence cap, prefill stream slot floor, and the retained
//! budget refinement for one admitted forward.

use crate::budget::ram::MlxRamBudget;
use crate::budget::ram_values::{MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry};
use crate::phase::MemoryPhase;

/// One mebibyte of sequence state per token: large enough that the geometry
/// projection dwarfs a small learned measurement.
const SEQUENCE_STATE_BYTES_PER_TOKEN: u64 = 1_048_576;

fn geometry_with_sequence_state(sequence_state_bytes_per_token: u64) -> MlxRamBudgetModelGeometry {
    MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 1_000_000_000,
        complete_expert_payload_bytes: 4_000_000_000,
        largest_complete_expert_layer_bytes: 100_000_000,
        largest_routed_expert_page_bytes: 25_000_000,
        sequence_state_bytes_per_token,
    }
}

fn budget_with_geometry(model_geometry: MlxRamBudgetModelGeometry) -> MlxRamBudget {
    MlxRamBudget::with_bootstrap_context_window_reserve_bytes(8_000_000_000, model_geometry, 0)
        .expect("a nonzero ceiling with valid geometry should construct")
}

fn idle_measurement(context_token_count: u64) -> MlxRamBudgetMeasurement {
    MlxRamBudgetMeasurement {
        phase: MemoryPhase::Idle,
        context_token_count,
        measured_context_and_activation_bytes: 0,
        observed_activation_headroom_bytes: 0,
        exact_temporary_workspace_bytes: 0,
    }
}

#[test]
fn should_floor_the_context_reserve_with_the_geometry_projection_beyond_every_measurement() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // One tiny measurement exists (bucket 0, buffered to 64 MB); the request
    // sits far beyond the highest measured bucket. The within-range branch
    // never lets the reserve fall below the geometry's sequence-state
    // projection, so the beyond-measured branch must honor the same floor:
    // a learned measurement that disagrees with the geometry projection may
    // not silently under-reserve context for the unmeasured span.
    let mut mlx_ram_budget =
        budget_with_geometry(geometry_with_sequence_state(SEQUENCE_STATE_BYTES_PER_TOKEN));
    mlx_ram_budget.record_measurement(idle_measurement(1_024));

    let projected_sequence_state_bytes = 100_000 * SEQUENCE_STATE_BYTES_PER_TOKEN;
    let beyond_measured_reserve_bytes = mlx_ram_budget.context_window_reserve_bytes(100_000);

    assert!(
        beyond_measured_reserve_bytes >= projected_sequence_state_bytes,
        "the beyond-measured reserve {} must honor the same geometry-projection \
         floor as the within-range branch ({projected_sequence_state_bytes})",
        beyond_measured_reserve_bytes
    );
}

#[test]
fn should_keep_the_geometry_projection_floor_within_the_measured_range() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut mlx_ram_budget =
        budget_with_geometry(geometry_with_sequence_state(SEQUENCE_STATE_BYTES_PER_TOKEN));
    mlx_ram_budget.record_measurement(idle_measurement(1_024));

    // 512 tokens sit inside the measured bucket, where the reserve is
    // floored at the projection today.
    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(512),
        512 * SEQUENCE_STATE_BYTES_PER_TOKEN
    );
}

#[test]
fn should_project_only_the_unmeasured_tokens_above_the_highest_measurement() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // With zero sequence state per token the beyond-measured branch must
    // return exactly the buffered high-water evidence: nothing is projected
    // for the measured span, and the unmeasured span adds zero.
    let mut mlx_ram_budget = budget_with_geometry(geometry_with_sequence_state(0));
    mlx_ram_budget.record_measurement(idle_measurement(1_024));

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(1_000_000),
        64_000_000
    );
}

#[test]
fn should_cap_activation_headroom_at_what_the_model_core_can_coexist_with() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // A learned activation observation of 500 MB under a model core that
    // leaves only 10 bytes of the ceiling is paper: the reserve caps at the
    // coexistence bound so it stays grantable.
    let model_geometry = MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 8_000_000_000 - 10,
        complete_expert_payload_bytes: 4_000_000_000,
        largest_complete_expert_layer_bytes: 100_000_000,
        largest_routed_expert_page_bytes: 25_000_000,
        sequence_state_bytes_per_token: 0,
    };
    let mut mlx_ram_budget = budget_with_geometry(model_geometry);
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MemoryPhase::Decode,
        context_token_count: 1_024,
        measured_context_and_activation_bytes: 600_000_000,
        observed_activation_headroom_bytes: 500_000_000,
        exact_temporary_workspace_bytes: 0,
    });

    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Decode, 1),
        10,
        "the learned 500 MB observation caps at the 10 bytes that can coexist \
         with the model core"
    );
}

#[test]
fn should_return_zero_activation_headroom_when_the_model_fills_the_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let model_geometry = MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 8_000_000_000,
        complete_expert_payload_bytes: 4_000_000_000,
        largest_complete_expert_layer_bytes: 100_000_000,
        largest_routed_expert_page_bytes: 25_000_000,
        sequence_state_bytes_per_token: 0,
    };
    let mlx_ram_budget = budget_with_geometry(model_geometry);

    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Prefill, 2_048),
        0,
        "no activation can coexist with a model that fills the ceiling; the \
         reserve must be zero, not an ungrantable paper number"
    );
}

#[test]
fn should_keep_the_declared_prefill_stream_slot_at_least_one_packed_complete_layer() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut mlx_ram_budget = budget_with_geometry(geometry_with_sequence_state(0));
    mlx_ram_budget.set_prefill_complete_layer_stream_slot_bytes(10);

    let planned_budget = mlx_ram_budget.plan(MemoryPhase::Prefill, 1_024, 2_048, 0);
    assert_eq!(
        planned_budget.complete_layer_stream_slot_bytes, 100_000_000,
        "a declared slot below one packed complete layer is raised to the layer"
    );

    mlx_ram_budget.set_prefill_complete_layer_stream_slot_bytes(250_000_000);
    let planned_budget = mlx_ram_budget.plan(MemoryPhase::Prefill, 1_024, 2_048, 0);
    assert_eq!(
        planned_budget.complete_layer_stream_slot_bytes, 250_000_000,
        "a declared slot above one packed complete layer is kept"
    );
}

#[test]
fn should_refine_the_retained_expert_budget_at_the_exact_admitted_boundaries() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mlx_ram_budget = budget_with_geometry(geometry_with_sequence_state(0));
    let mlx_active_memory_ceiling_bytes = 8_000_000_000_u64;

    // Projected active lands exactly on the ceiling: retention stands.
    assert_eq!(
        mlx_ram_budget.retained_expert_budget_for_admitted_forward(
            mlx_active_memory_ceiling_bytes - 40,
            60,
            40,
        ),
        60
    );
    // One byte over: exactly one retained byte must yield.
    assert_eq!(
        mlx_ram_budget.retained_expert_budget_for_admitted_forward(
            mlx_active_memory_ceiling_bytes - 40,
            60,
            41,
        ),
        59
    );
    // A shortfall past everything retained saturates at zero, never wraps.
    assert_eq!(
        mlx_ram_budget.retained_expert_budget_for_admitted_forward(100, 60, u64::MAX),
        0
    );
    // Headroom below the ceiling grows the allowed retention one for one.
    assert_eq!(
        mlx_ram_budget.retained_expert_budget_for_admitted_forward(100, 60, 0),
        60 + mlx_active_memory_ceiling_bytes - 100
    );
}

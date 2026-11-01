//! Boundary tests for the RAM budget owner's projected and learned context reserve,
//! activation coexistence cap, prefill stream slot floor, and the retained
//! budget refinement for one admitted forward.

use crate::budget::ram::MlxRamBudget;
use crate::budget::ram_values::{MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry};
use crate::phase::MemoryPhase;

/// One mebibyte of sequence state per token: large enough that the geometry
/// projection of a 2,048-token context alone exceeds the bootstrap floor.
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

/// Bytes the decoder cache layout of the measured Qwen3.5 sparse model
/// appends per context token.
const MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN: u64 = 20_480;

const SAFETY_BUFFER_BYTES: u64 = 64_000_000;

fn prefill_measurement_of_persistent_context(
    forward_token_count: u64,
    persistent_context_bytes: u64,
) -> MlxRamBudgetMeasurement {
    MlxRamBudgetMeasurement {
        phase: MemoryPhase::Prefill,
        forward_token_count,
        measured_context_and_activation_bytes: persistent_context_bytes,
        observed_activation_headroom_bytes: 0,
        exact_temporary_workspace_bytes: 0,
    }
}

/// The production bootstrap floor, so these tests see exactly when the floor
/// stops applying.
fn budget_with_production_bootstrap_floor(sequence_state_bytes_per_token: u64) -> MlxRamBudget {
    MlxRamBudget::new(
        8_000_000_000,
        geometry_with_sequence_state(sequence_state_bytes_per_token),
    )
    .expect("a nonzero ceiling with valid geometry should construct")
}

#[test]
fn should_keep_the_bootstrap_floor_before_any_measurement() {
    let mlx_ram_budget =
        budget_with_production_bootstrap_floor(MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN);

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(5_242),
        crate::budget::ram::BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES
    );
}

#[test]
fn should_let_the_projection_exceed_the_bootstrap_floor_before_any_measurement() {
    let mlx_ram_budget = budget_with_production_bootstrap_floor(SEQUENCE_STATE_BYTES_PER_TOKEN);

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(2_048),
        2_048 * SEQUENCE_STATE_BYTES_PER_TOKEN
    );
}

#[test]
fn should_replace_the_bootstrap_floor_with_projection_plus_learned_overhead_after_a_measurement() {
    // A 2,048-token chunk left its own sequence state plus 50 MB of fixed
    // state behind. A 5,242-token request then needs its whole projection,
    // the 50 MB, and the buffer, not the 1 GB floor: the measurement no longer
    // has to come from a forward as long as the queried context.
    let mut mlx_ram_budget =
        budget_with_production_bootstrap_floor(MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN);
    mlx_ram_budget.record_measurement(prefill_measurement_of_persistent_context(
        2_048,
        2_048 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 50_000_000,
    ));

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(5_242),
        5_242 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 50_000_000 + SAFETY_BUFFER_BYTES
    );
}

#[test]
fn should_learn_zero_overhead_when_a_forward_grows_less_than_its_projection() {
    let mut mlx_ram_budget =
        budget_with_production_bootstrap_floor(MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN);
    mlx_ram_budget.record_measurement(prefill_measurement_of_persistent_context(2_048, 1_000));

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(5_242),
        5_242 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + SAFETY_BUFFER_BYTES
    );
}

#[test]
fn should_keep_the_overhead_high_water_when_a_later_forward_measures_less() {
    let mut mlx_ram_budget =
        budget_with_production_bootstrap_floor(MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN);
    mlx_ram_budget.record_measurement(prefill_measurement_of_persistent_context(
        2_048,
        2_048 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 80_000_000,
    ));
    mlx_ram_budget.record_measurement(prefill_measurement_of_persistent_context(
        1,
        MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 3_000_000,
    ));

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(1_024),
        1_024 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 80_000_000 + SAFETY_BUFFER_BYTES
    );
}

#[test]
fn should_saturate_instead_of_wrapping_for_an_extreme_context_length() {
    let mut mlx_ram_budget = budget_with_production_bootstrap_floor(SEQUENCE_STATE_BYTES_PER_TOKEN);
    mlx_ram_budget.record_measurement(prefill_measurement_of_persistent_context(1, u64::MAX));

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(u64::MAX),
        u64::MAX
    );
}

#[test]
fn should_learn_the_fixed_state_beyond_the_projection_from_the_live_decoder_payload() {
    // 5,242 tokens of projected state plus 60 MB of fixed recurrent state and
    // slab rounding: the reserve for a longer context carries that 60 MB on
    // top of its own projection.
    let mut mlx_ram_budget =
        budget_with_production_bootstrap_floor(MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN);
    mlx_ram_budget.record_context_state_payload(
        5_242,
        5_242 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 60_000_000,
    );

    assert!(mlx_ram_budget.has_context_window_measurement());
    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(5_542),
        5_542 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 60_000_000 + SAFETY_BUFFER_BYTES
    );
}

#[test]
fn should_learn_zero_overhead_from_a_live_payload_smaller_than_its_projection() {
    let mut mlx_ram_budget =
        budget_with_production_bootstrap_floor(MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN);
    mlx_ram_budget.record_context_state_payload(5_242, 0);

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(5_242),
        5_242 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + SAFETY_BUFFER_BYTES
    );
}

#[test]
fn should_keep_the_larger_overhead_across_live_payload_and_forward_measurements() {
    let mut mlx_ram_budget =
        budget_with_production_bootstrap_floor(MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN);
    mlx_ram_budget.record_context_state_payload(
        1_024,
        1_024 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 60_000_000,
    );
    mlx_ram_budget.record_measurement(prefill_measurement_of_persistent_context(
        2_048,
        2_048 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 10_000_000,
    ));
    mlx_ram_budget
        .record_context_state_payload(2_048, 2_048 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN);

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(4_096),
        4_096 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 60_000_000 + SAFETY_BUFFER_BYTES
    );
}

#[test]
fn should_cap_activation_headroom_at_what_the_model_core_can_coexist_with() {
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
        forward_token_count: 1_024,
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

/// A geometry whose experts cannot all be seated under the 8 GB test
/// ceiling, so prefill keeps streaming at least one complete layer.
fn streaming_geometry() -> MlxRamBudgetModelGeometry {
    MlxRamBudgetModelGeometry {
        complete_expert_payload_bytes: 9_000_000_000,
        ..geometry_with_sequence_state(0)
    }
}

#[test]
fn should_keep_the_declared_prefill_stream_slot_at_least_one_packed_complete_layer() {
    let mut mlx_ram_budget = budget_with_geometry(streaming_geometry());
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
fn should_reserve_no_prefill_stream_slot_when_every_expert_fits() {
    // 4 GB of experts beside a 1 GB core fit the 8 GB ceiling with the
    // activation floor, so prefill streams nothing and needs no slot.
    let mut mlx_ram_budget = budget_with_geometry(geometry_with_sequence_state(0));
    mlx_ram_budget.set_prefill_complete_layer_stream_slot_bytes(250_000_000);

    let planned_budget = mlx_ram_budget.plan(MemoryPhase::Prefill, 1_024, 2_048, 0);
    assert_eq!(planned_budget.complete_layer_stream_slot_bytes, 0);
    assert!(planned_budget.retained_expert_budget_bytes >= 4_000_000_000);
}

#[test]
fn should_refine_the_retained_expert_budget_at_the_exact_admitted_boundaries() {
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

/// A measured budget whose zero-token reserve is exactly 50 MB of learned
/// fixed state plus the safety buffer.
fn measured_budget_with_fifty_megabyte_overhead() -> MlxRamBudget {
    let mut mlx_ram_budget =
        budget_with_production_bootstrap_floor(MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN);
    mlx_ram_budget.record_measurement(prefill_measurement_of_persistent_context(
        2_048,
        2_048 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 50_000_000,
    ));
    mlx_ram_budget
}

#[test]
fn should_reserve_the_live_prefix_a_between_request_sample_still_holds() {
    // A 15,512-token retained live prefix outweighs the zero-token reserve,
    // so the context owner must hold the prefix and the expert budget shrinks
    // by exactly the difference.
    let mlx_ram_budget = measured_budget_with_fifty_megabyte_overhead();
    let live_prefix_payload_bytes = 15_512 * MEASURED_SEQUENCE_STATE_BYTES_PER_TOKEN + 50_000_000;
    let fresh_plan = mlx_ram_budget.plan(MemoryPhase::Decode, 0, 1, 0);

    let holding_plan = mlx_ram_budget.plan_holding_live_context_state(
        MemoryPhase::Decode,
        0,
        1,
        0,
        live_prefix_payload_bytes,
    );

    assert_eq!(
        holding_plan.context_window_reserve_bytes,
        live_prefix_payload_bytes
    );
    assert_eq!(
        fresh_plan.retained_expert_budget_bytes - holding_plan.retained_expert_budget_bytes,
        live_prefix_payload_bytes - fresh_plan.context_window_reserve_bytes
    );
}

#[test]
fn should_keep_the_planned_reserve_when_no_live_state_is_held() {
    let mlx_ram_budget = measured_budget_with_fifty_megabyte_overhead();

    assert_eq!(
        mlx_ram_budget.plan_holding_live_context_state(MemoryPhase::Decode, 4_096, 1, 0, 0),
        mlx_ram_budget.plan(MemoryPhase::Decode, 4_096, 1, 0),
        "zero live state must leave the plan unchanged"
    );
}

#[test]
fn should_keep_the_planned_reserve_at_the_exact_live_payload_boundary() {
    // Live payload equal to the planned reserve: the floor neither adds nor
    // removes a byte; one byte more raises the reserve by exactly one byte.
    let mlx_ram_budget = measured_budget_with_fifty_megabyte_overhead();
    let planned_reserve_bytes = mlx_ram_budget
        .plan(MemoryPhase::Decode, 0, 1, 0)
        .context_window_reserve_bytes;

    let at_boundary = mlx_ram_budget.plan_holding_live_context_state(
        MemoryPhase::Decode,
        0,
        1,
        0,
        planned_reserve_bytes,
    );
    let one_byte_over = mlx_ram_budget.plan_holding_live_context_state(
        MemoryPhase::Decode,
        0,
        1,
        0,
        planned_reserve_bytes + 1,
    );

    assert_eq!(
        at_boundary.context_window_reserve_bytes,
        planned_reserve_bytes
    );
    assert_eq!(
        one_byte_over.context_window_reserve_bytes,
        planned_reserve_bytes + 1
    );
}

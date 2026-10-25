//! Tests for the adaptive RAM growth guard's analytic floor, which must equal
//! the activation promise request admission already made for the operation.

use crate::budget::adaptive_growth::{AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile};
use crate::budget::ram::MlxRamBudget;
use crate::budget::ram_values::MlxRamBudgetModelGeometry;
use crate::phase::MemoryPhase;

const TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES: u64 = 28_991_029_248;
const FOUR_BIT_LARGEST_COMPLETE_EXPERT_LAYER_BYTES: u64 = 855_638_016;

fn four_bit_resident_geometry(model_core_payload_bytes: u64) -> MlxRamBudgetModelGeometry {
    MlxRamBudgetModelGeometry {
        model_core_payload_bytes,
        complete_expert_payload_bytes: 19_595_788_288,
        largest_complete_expert_layer_bytes: FOUR_BIT_LARGEST_COMPLETE_EXPERT_LAYER_BYTES,
        largest_routed_expert_page_bytes: 3_342_336,
        sequence_state_bytes_per_token: 0,
    }
}

fn twenty_seven_gigabyte_budget(model_core_payload_bytes: u64) -> MlxRamBudget {
    MlxRamBudget::with_bootstrap_context_window_reserve_bytes(
        TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES,
        four_bit_resident_geometry(model_core_payload_bytes),
        0,
    )
    .expect("a nonzero ceiling with valid geometry should construct")
}

fn resident_prefill_context(forward_token_count: usize) -> AdaptiveRamGrowthContext {
    AdaptiveRamGrowthContext::prefill(
        forward_token_count,
        0,
        false,
        AdaptiveRamGrowthExecutionProfile::Resident,
    )
}

#[test]
fn should_give_the_guard_the_same_prefill_promise_request_admission_made() {
    let mlx_ram_budget = twenty_seven_gigabyte_budget(2_568_911_104);

    let guard_floor_bytes = mlx_ram_budget
        .adaptive_growth_unobserved_scope_transient_floor_bytes(resident_prefill_context(2_048));

    assert_eq!(guard_floor_bytes, 2_566_914_048);
    assert_eq!(
        u64::try_from(guard_floor_bytes).expect("the floor fits u64"),
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Prefill, 2_048)
    );
}

#[test]
fn should_give_decode_forwards_a_zero_floor() {
    let mlx_ram_budget = twenty_seven_gigabyte_budget(2_568_911_104);

    assert_eq!(
        mlx_ram_budget.adaptive_growth_unobserved_scope_transient_floor_bytes(
            AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Resident)
        ),
        0
    );
}

#[test]
fn should_cap_the_guard_floor_at_what_the_model_core_can_coexist_with() {
    let mlx_ram_budget =
        twenty_seven_gigabyte_budget(TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES - 1_000);

    assert_eq!(
        mlx_ram_budget.adaptive_growth_unobserved_scope_transient_floor_bytes(
            resident_prefill_context(2_048)
        ),
        1_000
    );
}

#[test]
fn should_replace_the_static_three_layer_prefill_floor_with_the_diagnostic_floor() {
    let mut mlx_ram_budget = twenty_seven_gigabyte_budget(2_568_911_104);

    mlx_ram_budget.set_diagnostic_prefill_activation_floor_bytes(1_241_335_884);

    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Prefill, 2_048),
        1_241_335_884
    );
}

#[test]
fn should_reserve_nothing_for_an_unobserved_prefill_when_the_diagnostic_floor_is_zero() {
    let mut mlx_ram_budget = twenty_seven_gigabyte_budget(2_568_911_104);

    mlx_ram_budget.set_diagnostic_prefill_activation_floor_bytes(0);

    assert_eq!(
        mlx_ram_budget.adaptive_growth_unobserved_scope_transient_floor_bytes(
            resident_prefill_context(2_048)
        ),
        0
    );
}

#[test]
fn should_keep_the_static_three_layer_prefill_floor_without_a_diagnostic_floor() {
    let mlx_ram_budget = twenty_seven_gigabyte_budget(2_568_911_104);

    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MemoryPhase::Prefill, 2_048),
        FOUR_BIT_LARGEST_COMPLETE_EXPERT_LAYER_BYTES * 3
    );
}

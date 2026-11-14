use super::MlxRamBudget;
use crate::budget::ram_values::{
    MlxRamBudgetError, MlxRamBudgetModelGeometry, MlxRamBudgetSnapshot,
};
use crate::ledger::{LedgerPhase, MemoryGovernor, ModelMemoryShape};

const MODEL_GEOMETRY: MlxRamBudgetModelGeometry = MlxRamBudgetModelGeometry {
    model_core_payload_bytes: 3_000,
    complete_expert_payload_bytes: 20_000,
    largest_complete_expert_layer_bytes: 1_000,
    largest_routed_expert_page_bytes: 100,
    sequence_state_bytes_per_token: 4,
};

#[test]
fn should_carry_the_limit_and_the_geometry_it_was_given() {
    let ram_budget = MlxRamBudget::new(16_000, MODEL_GEOMETRY).expect("the limit is not zero");
    assert_eq!(ram_budget.mlx_active_memory_ceiling_bytes(), 16_000);
    assert_eq!(ram_budget.model_geometry(), MODEL_GEOMETRY);
}

#[test]
fn should_refuse_a_zero_limit() {
    assert_eq!(
        MlxRamBudget::new(0, MODEL_GEOMETRY),
        Err(MlxRamBudgetError::InvalidCeiling)
    );
}

#[test]
fn should_take_a_new_limit_of_one_byte() {
    let mut ram_budget = MlxRamBudget::new(16_000, MODEL_GEOMETRY).expect("the limit is not zero");
    ram_budget
        .update_mlx_active_memory_ceiling_bytes(1)
        .expect("one byte is a limit");
    assert_eq!(ram_budget.mlx_active_memory_ceiling_bytes(), 1);
}

#[test]
fn should_keep_the_current_limit_when_asked_to_change_it_to_zero() {
    let mut ram_budget = MlxRamBudget::new(16_000, MODEL_GEOMETRY).expect("the limit is not zero");
    assert_eq!(
        ram_budget.update_mlx_active_memory_ceiling_bytes(0),
        Err(MlxRamBudgetError::InvalidCeiling)
    );
    assert_eq!(ram_budget.mlx_active_memory_ceiling_bytes(), 16_000);
}

fn accounts_with_everything_in_use() -> crate::ledger::LedgerAccounts {
    let mut memory_governor = MemoryGovernor::new(
        10_000,
        ModelMemoryShape {
            core_weight_bytes: 3_000,
            streamable_weight_bytes: 20_000,
            reading_prompt_stream_slot_bytes: 500,
            writing_reply_stream_slot_bytes: 50,
        },
    )
    .expect("the limit is not zero");
    let reading_prompt = LedgerPhase::ReadingPrompt {
        span_token_count: 2_048,
    };
    let _ = memory_governor.record_probe(reading_prompt, 1_000);
    let _ = memory_governor.start_request(reading_prompt, 2_000);
    memory_governor.accounts()
}

#[test]
fn should_show_the_ledger_accounts_as_one_snapshot_that_adds_up_to_the_limit() {
    let accounts = accounts_with_everything_in_use();
    let budget_snapshot = MlxRamBudgetSnapshot::from_ledger_accounts(&accounts);
    assert_eq!(budget_snapshot.mlx_active_memory_ceiling_bytes, 10_000);
    assert_eq!(budget_snapshot.model_core_payload_bytes, 3_000);
    assert_eq!(budget_snapshot.context_window_reserve_bytes, 2_000);
    // Working memory 1,000 and the margin of a fiftieth, 200.
    assert_eq!(budget_snapshot.activation_headroom_bytes, 1_200);
    assert_eq!(budget_snapshot.complete_layer_stream_slot_bytes, 500);
    assert_eq!(budget_snapshot.retained_expert_budget_bytes, 3_300);
    assert_eq!(budget_snapshot.other_fixed_bytes, 0);
    assert_eq!(accounts.set_aside_bytes(), 6_700);
}

#[test]
fn should_set_only_the_weights_aside_before_anything_is_measured() {
    let budget_snapshot = MlxRamBudgetSnapshot::with_only_weights_set_aside(10_000, 3_000);
    assert_eq!(budget_snapshot.model_core_payload_bytes, 3_000);
    assert_eq!(budget_snapshot.other_fixed_bytes, 7_000);
    assert_eq!(budget_snapshot.retained_expert_budget_bytes, 0);
}

#[test]
fn should_leave_nothing_unowned_when_the_weights_exceed_the_limit() {
    let budget_snapshot = MlxRamBudgetSnapshot::with_only_weights_set_aside(1_000, 3_000);
    assert_eq!(budget_snapshot.other_fixed_bytes, 0);
}

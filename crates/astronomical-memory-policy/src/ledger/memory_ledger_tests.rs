use super::{MemoryLedger, MemoryLedgerError};
use crate::ledger::accounts::LedgerAccounts;
use crate::ledger::lease::LeasePurpose;
use crate::ledger::model_shape::ModelMemoryShape;
use crate::ledger::phase::LedgerPhase;
use crate::ledger::reconciliation::HeldMemory;

const GIBIBYTE: u64 = 1 << 30;
const MEBIBYTE: u64 = 1 << 20;
const READING: LedgerPhase = LedgerPhase::ReadingPrompt {
    span_token_count: 4_096,
};
const MODEL_SHAPE: ModelMemoryShape = ModelMemoryShape {
    core_weight_bytes: 3 * GIBIBYTE,
    streamable_weight_bytes: 20 * GIBIBYTE,
    reading_prompt_stream_slot_bytes: 2 * GIBIBYTE,
    writing_reply_stream_slot_bytes: GIBIBYTE / 8,
};

fn ledger_at_16_gibibytes() -> MemoryLedger {
    MemoryLedger::new(16 * GIBIBYTE, MODEL_SHAPE).expect("a nonzero limit opens a ledger")
}

fn assert_accounts_add_up(accounts: LedgerAccounts) {
    assert_eq!(
        accounts.needed_bytes() + accounts.kept_weight_capacity_bytes + accounts.unassigned_bytes,
        accounts.limit_bytes + accounts.shortfall_bytes,
        "the accounts must add up to the limit: {accounts:?}"
    );
}

#[test]
fn should_refuse_a_zero_limit_at_open_and_on_change() {
    assert_eq!(
        MemoryLedger::new(0, MODEL_SHAPE),
        Err(MemoryLedgerError::ZeroLimit)
    );
    let mut ledger = ledger_at_16_gibibytes();
    assert_eq!(ledger.change_limit(0), Err(MemoryLedgerError::ZeroLimit));
    assert_eq!(ledger.accounts().limit_bytes, 16 * GIBIBYTE);
}

#[test]
fn should_open_idle_with_no_context_and_the_remainder_for_kept_weights() {
    let ledger = ledger_at_16_gibibytes();
    let accounts = ledger.accounts();
    assert_eq!(ledger.phase(), LedgerPhase::Idle);
    assert_eq!(accounts.context_bytes, 0);
    assert_eq!(accounts.mlx_working_buffer_bytes, 0);
    assert_eq!(
        accounts.kept_weight_capacity_bytes,
        16 * GIBIBYTE - 3 * GIBIBYTE - accounts.margin_bytes
    );
    assert_accounts_add_up(accounts);
}

#[test]
fn should_keep_nothing_while_reading_until_a_layer_step_is_measured() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.start_request(GIBIBYTE / 2);
    let unmeasured = ledger.enter_phase(READING);
    assert_eq!(unmeasured.kept_weight_capacity_bytes, 0);
    assert!(!unmeasured.mlx_working_buffers_are_measured);
    let measured = ledger.record_mlx_working_buffers(READING, GIBIBYTE);
    assert!(measured.mlx_working_buffers_are_measured);
    assert!(measured.kept_weight_capacity_bytes > 9 * GIBIBYTE);
    assert_accounts_add_up(measured);
}

#[test]
fn should_hold_capacity_still_between_events() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.start_request(GIBIBYTE / 2);
    ledger.enter_phase(READING);
    let after_first_measurement = ledger.record_mlx_working_buffers(READING, GIBIBYTE);
    // Smaller and equal measurements are not events: nothing may move.
    assert_eq!(
        ledger.record_mlx_working_buffers(READING, GIBIBYTE / 2),
        after_first_measurement
    );
    assert_eq!(
        ledger.record_mlx_working_buffers(READING, GIBIBYTE),
        after_first_measurement
    );
    assert_eq!(ledger.accounts(), after_first_measurement);
}

#[test]
fn should_shrink_capacity_by_exactly_what_a_larger_measurement_adds() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.start_request(GIBIBYTE / 2);
    ledger.enter_phase(READING);
    let before = ledger.record_mlx_working_buffers(READING, GIBIBYTE);
    let after = ledger.record_mlx_working_buffers(READING, GIBIBYTE + 300 * MEBIBYTE);
    assert_eq!(
        before.kept_weight_capacity_bytes - after.kept_weight_capacity_bytes,
        300 * MEBIBYTE
    );
}

#[test]
fn should_give_more_capacity_for_writing_than_for_reading() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.start_request(GIBIBYTE / 2);
    ledger.record_mlx_working_buffers(READING, GIBIBYTE);
    ledger.record_mlx_working_buffers(LedgerPhase::WritingReply, 64 * MEBIBYTE);
    let reading = ledger.enter_phase(READING);
    let writing = ledger.enter_phase(LedgerPhase::WritingReply);
    assert!(writing.kept_weight_capacity_bytes > reading.kept_weight_capacity_bytes);
    assert_accounts_add_up(writing);
}

#[test]
fn should_take_context_growth_out_of_capacity_byte_for_byte() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.start_request(GIBIBYTE / 2);
    ledger.record_mlx_working_buffers(LedgerPhase::WritingReply, 64 * MEBIBYTE);
    let before = ledger.enter_phase(LedgerPhase::WritingReply);
    let after = ledger.grow_context(128 * MEBIBYTE);
    assert_eq!(
        before.kept_weight_capacity_bytes - after.kept_weight_capacity_bytes,
        128 * MEBIBYTE
    );
    assert_eq!(after.context_bytes, GIBIBYTE / 2 + 128 * MEBIBYTE);
}

#[test]
fn should_return_to_idle_and_keep_only_the_retained_context_when_a_request_ends() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.start_request(GIBIBYTE);
    ledger.enter_phase(LedgerPhase::WritingReply);
    let finished = ledger.finish_request(GIBIBYTE / 4);
    assert_eq!(ledger.phase(), LedgerPhase::Idle);
    assert_eq!(finished.context_bytes, GIBIBYTE / 4);
    assert_eq!(finished.mlx_working_buffer_bytes, 0);
    assert_accounts_add_up(finished);
}

#[test]
fn should_keep_measurements_when_the_limit_moves() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.start_request(GIBIBYTE / 2);
    ledger.enter_phase(READING);
    let at_16 = ledger.record_mlx_working_buffers(READING, GIBIBYTE);
    let at_12 = ledger
        .change_limit(12 * GIBIBYTE)
        .expect("a nonzero limit is accepted");
    assert!(at_12.mlx_working_buffers_are_measured);
    assert_eq!(at_12.mlx_working_buffer_bytes, GIBIBYTE);
    assert!(at_12.kept_weight_capacity_bytes < at_16.kept_weight_capacity_bytes);
    assert_accounts_add_up(at_12);
}

#[test]
fn should_take_a_lease_out_of_capacity_and_return_it_when_given_back() {
    let mut ledger = ledger_at_16_gibibytes();
    let before = ledger.accounts();
    let (vision_lease, with_lease) = ledger.grant_lease(LeasePurpose::VisionTower, GIBIBYTE);
    assert_eq!(
        before.kept_weight_capacity_bytes - with_lease.kept_weight_capacity_bytes,
        GIBIBYTE
    );
    assert_eq!(ledger.give_back_lease(vision_lease), Ok(before));
    assert_eq!(
        ledger.give_back_lease(vision_lease),
        Err(MemoryLedgerError::UnknownLease)
    );
}

#[test]
fn should_settle_a_redone_step_without_counting_its_headroom_and_measurement_together() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.start_request(GIBIBYTE / 2);
    ledger.enter_phase(READING);
    let before_overflow = ledger.record_mlx_working_buffers(READING, GIBIBYTE);
    let (headroom_lease, during_redo) = ledger.grant_lease(LeasePurpose::RedoHeadroom, GIBIBYTE);
    assert_eq!(
        before_overflow.kept_weight_capacity_bytes - during_redo.kept_weight_capacity_bytes,
        GIBIBYTE
    );
    let settled = ledger
        .settle_redone_step(headroom_lease, READING, 2 * GIBIBYTE)
        .expect("the headroom lease is held");
    assert_eq!(settled.lease_bytes, 0);
    assert_eq!(settled.mlx_working_buffer_bytes, 2 * GIBIBYTE);
    assert_eq!(
        settled.kept_weight_capacity_bytes, during_redo.kept_weight_capacity_bytes,
        "twice the MLX working buffers with no lease is the capacity the redo ran at"
    );
    assert_accounts_add_up(settled);
}

#[test]
fn should_refuse_to_settle_a_redone_step_with_a_lease_that_is_not_held() {
    let mut ledger = ledger_at_16_gibibytes();
    let (headroom_lease, _) = ledger.grant_lease(LeasePurpose::RedoHeadroom, GIBIBYTE);
    ledger
        .give_back_lease(headroom_lease)
        .expect("the headroom lease is held");
    let before = ledger.accounts();
    assert_eq!(
        ledger.settle_redone_step(headroom_lease, READING, 2 * GIBIBYTE),
        Err(MemoryLedgerError::UnknownLease)
    );
    assert_eq!(
        ledger.accounts(),
        before,
        "a refused settlement changes nothing"
    );
}

#[test]
fn should_agree_with_a_reading_that_matches_what_is_held() {
    let ledger = ledger_at_16_gibibytes();
    let held_memory = HeldMemory {
        context_bytes: GIBIBYTE / 2,
        kept_weight_bytes: 8 * GIBIBYTE,
        stream_slot_bytes: GIBIBYTE,
    };
    let reconciliation = ledger.reconcile(held_memory, 12 * GIBIBYTE + GIBIBYTE / 2);
    assert_eq!(reconciliation.unaccounted_bytes, 0);
    assert_eq!(reconciliation.overcounted_bytes, 0);
    assert_eq!(reconciliation.disagreement_bytes(), 0);
    assert_eq!(reconciliation.kept_over_capacity_bytes, 0);
}

#[test]
fn should_report_memory_in_use_that_no_account_names() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.grant_lease(LeasePurpose::AstroSsdPromptCachePublication, 100 * MEBIBYTE);
    let held_memory = HeldMemory {
        context_bytes: GIBIBYTE / 2,
        kept_weight_bytes: 8 * GIBIBYTE,
        stream_slot_bytes: 0,
    };
    let expected_active_bytes = 3 * GIBIBYTE + GIBIBYTE / 2 + 8 * GIBIBYTE + 100 * MEBIBYTE;
    let reconciliation = ledger.reconcile(held_memory, expected_active_bytes + 860 * MEBIBYTE);
    assert_eq!(reconciliation.expected_active_bytes, expected_active_bytes);
    assert_eq!(reconciliation.unaccounted_bytes, 860 * MEBIBYTE);
    assert_eq!(reconciliation.overcounted_bytes, 0);
    assert_eq!(reconciliation.disagreement_bytes(), 860 * MEBIBYTE);
}

#[test]
fn should_not_expect_a_redo_headroom_lease_to_be_memory_in_use() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.grant_lease(LeasePurpose::RedoHeadroom, GIBIBYTE);
    let held_memory = HeldMemory {
        context_bytes: 0,
        kept_weight_bytes: 4 * GIBIBYTE,
        stream_slot_bytes: 0,
    };
    let reconciliation = ledger.reconcile(held_memory, 7 * GIBIBYTE);
    assert_eq!(reconciliation.expected_active_bytes, 7 * GIBIBYTE);
    assert_eq!(reconciliation.disagreement_bytes(), 0);
}

#[test]
fn should_report_an_account_that_names_memory_not_in_use() {
    let ledger = ledger_at_16_gibibytes();
    let held_memory = HeldMemory {
        context_bytes: 0,
        kept_weight_bytes: 8 * GIBIBYTE,
        stream_slot_bytes: 0,
    };
    let reconciliation = ledger.reconcile(held_memory, 10 * GIBIBYTE);
    assert_eq!(reconciliation.unaccounted_bytes, 0);
    assert_eq!(reconciliation.overcounted_bytes, GIBIBYTE);
    assert_eq!(reconciliation.disagreement_bytes(), GIBIBYTE);
}

#[test]
fn should_report_weights_kept_beyond_the_capacity() {
    let mut ledger = ledger_at_16_gibibytes();
    ledger.start_request(GIBIBYTE / 2);
    ledger.record_mlx_working_buffers(READING, GIBIBYTE);
    let capacity_bytes = ledger.enter_phase(READING).kept_weight_capacity_bytes;
    let held_memory = HeldMemory {
        context_bytes: GIBIBYTE / 2,
        kept_weight_bytes: capacity_bytes + 900 * MEBIBYTE,
        stream_slot_bytes: 0,
    };
    let reconciliation = ledger.reconcile(held_memory, 14 * GIBIBYTE);
    assert_eq!(reconciliation.kept_over_capacity_bytes, 900 * MEBIBYTE);
}

/// A linear congruential generator: the same sequence on every run, so a
/// failure names a reproducible step.
struct EventDice(u64);

impl EventDice {
    fn roll(&mut self, side_count: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % side_count
    }
}

#[test]
fn should_keep_the_accounts_adding_up_through_any_sequence_of_events() {
    let mut ledger = ledger_at_16_gibibytes();
    let mut event_dice = EventDice(2_221);
    let mut held_leases = Vec::new();
    for step_index in 0..5_000 {
        match event_dice.roll(8) {
            0 => drop(ledger.start_request(event_dice.roll(4 * GIBIBYTE))),
            1 => drop(ledger.grow_context(event_dice.roll(256 * MEBIBYTE))),
            2 => drop(ledger.finish_request(event_dice.roll(GIBIBYTE))),
            3 => drop(ledger.enter_phase(READING)),
            4 => drop(ledger.enter_phase(LedgerPhase::WritingReply)),
            5 => drop(ledger.record_mlx_working_buffers(READING, event_dice.roll(6 * GIBIBYTE))),
            6 => {
                let (lease_id, _) = ledger.grant_lease(
                    LeasePurpose::AstroSsdPromptCacheRestore,
                    event_dice.roll(GIBIBYTE),
                );
                held_leases.push(lease_id);
            }
            _ => {
                if let Some(lease_id) = held_leases.pop() {
                    ledger
                        .give_back_lease(lease_id)
                        .expect("a held lease can be given back");
                } else {
                    let new_limit_bytes = (1 + event_dice.roll(48)) * GIBIBYTE;
                    ledger
                        .change_limit(new_limit_bytes)
                        .expect("a nonzero limit is accepted");
                }
            }
        }
        let accounts = ledger.accounts();
        assert_eq!(
            accounts.needed_bytes()
                + accounts.kept_weight_capacity_bytes
                + accounts.unassigned_bytes,
            accounts.limit_bytes + accounts.shortfall_bytes,
            "step {step_index} broke the accounts: {accounts:?}"
        );
        assert!(
            accounts.kept_weight_capacity_bytes <= MODEL_SHAPE.streamable_weight_bytes,
            "step {step_index} offered more capacity than there are weights to keep"
        );
        assert!(
            accounts.shortfall_bytes == 0 || accounts.kept_weight_capacity_bytes == 0,
            "step {step_index} kept weights while short of the limit: {accounts:?}"
        );
    }
}

#[test]
fn should_move_the_system_wired_reserve_with_each_reading_and_take_it_from_capacity() {
    let mut memory_ledger = ledger_at_16_gibibytes();
    let capacity_before_bytes = memory_ledger.accounts().kept_weight_capacity_bytes;
    let system_wire_limit_bytes = 100 * GIBIBYTE;

    let accounts_over_the_limit = memory_ledger
        .observe_system_wired(system_wire_limit_bytes + GIBIBYTE, system_wire_limit_bytes);
    assert_eq!(accounts_over_the_limit.system_wired_reserve_bytes, GIBIBYTE);
    assert_eq!(
        accounts_over_the_limit.kept_weight_capacity_bytes,
        capacity_before_bytes - GIBIBYTE
    );

    // Well under the limit: everything beyond the 1 GiB slack is given back.
    let accounts_under_the_limit = memory_ledger.observe_system_wired(
        system_wire_limit_bytes - 3 * GIBIBYTE,
        system_wire_limit_bytes,
    );
    assert_eq!(accounts_under_the_limit.system_wired_reserve_bytes, 0);
    assert_eq!(
        accounts_under_the_limit.kept_weight_capacity_bytes,
        capacity_before_bytes
    );
}

#[test]
fn should_never_carry_a_system_wired_reserve_larger_than_the_limit() {
    let mut memory_ledger = ledger_at_16_gibibytes();
    let system_wire_limit_bytes = 100 * GIBIBYTE;

    let _ = memory_ledger.observe_system_wired(
        system_wire_limit_bytes + 60 * GIBIBYTE,
        system_wire_limit_bytes,
    );
    // One reading at the slack's edge keeps the reserve; had 60 GiB been
    // carried, the next reading under the limit could not bring it down.
    let accounts = memory_ledger.observe_system_wired(
        system_wire_limit_bytes - 18 * GIBIBYTE,
        system_wire_limit_bytes,
    );

    assert_eq!(accounts.system_wired_reserve_bytes, 0);
    assert_eq!(accounts.shortfall_bytes, 0);
}

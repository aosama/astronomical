//! The governor's events: each answers with the capacity in force, a
//! measured step raises what is known and is reconciled, and a step that
//! hit the limit is redone with leased room that is settled or given back.

use super::{MAXIMUM_REDO_ATTEMPT_COUNT, MeasuredStep, MemoryGovernor};
use crate::ledger::{HeldMemory, LeasePurpose, LedgerPhase, MemoryLedgerError, ModelMemoryShape};

const GIBIBYTE: u64 = 1 << 30;
const LIMIT_BYTES: u64 = 16 * GIBIBYTE;
const READING: LedgerPhase = LedgerPhase::ReadingPrompt {
    span_token_count: 4_096,
};

fn model_shape() -> ModelMemoryShape {
    ModelMemoryShape {
        core_weight_bytes: 3 * GIBIBYTE,
        streamable_weight_bytes: 20 * GIBIBYTE,
        reading_prompt_stream_slot_bytes: GIBIBYTE,
        writing_reply_stream_slot_bytes: GIBIBYTE / 8,
    }
}

fn governor() -> MemoryGovernor {
    MemoryGovernor::new(LIMIT_BYTES, model_shape()).expect("the limit is not zero")
}

/// A step that held `held_bytes` around it, read `in_flight_weight_bytes`
/// and needed `working_memory_bytes` beyond both.
fn step(phase: LedgerPhase, kept_weight_bytes: u64, working_memory_bytes: u64) -> MeasuredStep {
    let held_bytes = 3 * GIBIBYTE + kept_weight_bytes;
    MeasuredStep {
        phase,
        held_memory: HeldMemory {
            context_bytes: 0,
            kept_weight_bytes,
            stream_slot_bytes: 0,
        },
        in_flight_weight_bytes: GIBIBYTE / 2,
        active_bytes_before_step: held_bytes,
        active_bytes_after_step: held_bytes,
        peak_bytes_during_step: held_bytes + GIBIBYTE / 2 + working_memory_bytes,
    }
}

#[test]
fn should_refuse_a_zero_limit() {
    assert_eq!(
        MemoryGovernor::new(0, model_shape()).err(),
        Some(MemoryLedgerError::ZeroLimit)
    );
}

#[test]
fn should_keep_nothing_while_reading_until_a_step_has_been_measured() {
    let mut governor = governor();
    assert_eq!(governor.start_request(READING, GIBIBYTE).bytes(), 0);
}

#[test]
fn should_give_the_remainder_once_a_probe_has_measured_a_step() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let capacity = governor.start_request(READING, GIBIBYTE);
    let accounts = governor.accounts();
    assert_eq!(capacity.bytes(), accounts.kept_weight_capacity_bytes);
    assert_eq!(
        accounts.needed_bytes() + capacity.bytes(),
        LIMIT_BYTES,
        "the accounts add up to the limit"
    );
}

#[test]
fn should_take_the_weights_in_flight_out_of_a_step_working_memory() {
    let measured_step = step(READING, 0, GIBIBYTE);
    assert_eq!(
        measured_step.peak_beyond_held_bytes(),
        GIBIBYTE + GIBIBYTE / 2
    );
    assert_eq!(measured_step.working_memory_bytes(), GIBIBYTE);
}

#[test]
fn should_measure_no_working_memory_when_the_peak_never_rose() {
    let mut quiet_step = step(READING, 0, 0);
    quiet_step.peak_bytes_during_step = quiet_step.active_bytes_after_step;
    assert_eq!(quiet_step.working_memory_bytes(), 0);
}

#[test]
fn should_shrink_the_capacity_only_when_a_step_needs_more_than_was_known() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let capacity_before = governor.start_request(READING, 0);
    let lighter_verdict = governor.complete_step(step(READING, 0, GIBIBYTE / 2));
    assert_eq!(lighter_verdict.capacity, capacity_before);
    let heavier_verdict = governor.complete_step(step(READING, 0, 2 * GIBIBYTE));
    assert_eq!(
        capacity_before.bytes() - heavier_verdict.capacity.bytes(),
        GIBIBYTE,
        "the capacity gives up exactly what the step proved it needs"
    );
}

#[test]
fn should_report_memory_no_account_names() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let _ = governor.start_request(READING, 0);
    let mut leaking_step = step(READING, 0, GIBIBYTE);
    leaking_step.active_bytes_after_step += 300;
    leaking_step.peak_bytes_during_step += 300;
    let verdict = governor.complete_step(leaking_step);
    assert_eq!(verdict.reading.reconciliation.unaccounted_bytes, 300);
    assert_eq!(verdict.reading.largest_unaccounted_bytes, 300);
    assert_eq!(governor.latest_reading(), Some(verdict.reading));
}

#[test]
fn should_remember_the_emptiest_the_kept_weights_were_while_reading() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let capacity = governor.start_request(READING, 0);
    let _ = governor.complete_step(step(READING, capacity.bytes(), GIBIBYTE));
    let _ = governor.complete_step(step(READING, capacity.bytes() / 4, GIBIBYTE));
    let verdict = governor.complete_step(step(READING, capacity.bytes(), GIBIBYTE));
    assert_eq!(
        verdict.reading.lowest_kept_share_while_reading_permille,
        Some(u32::try_from(capacity.bytes() / 4 * 1_000 / capacity.bytes()).expect("a share fits")),
        "the quarter-full step is the one remembered"
    );
}

#[test]
fn should_start_each_request_with_fresh_worst_readings_and_keep_the_totals() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let _ = governor.start_request(READING, 0);
    let _ = governor.complete_step(step(READING, 0, GIBIBYTE));
    let (redo_attempt, _) = governor
        .begin_redo(READING, None, 0)
        .expect("the first redo is always allowed");
    let _ = governor.abandon_redo(redo_attempt);
    let _ = governor.finish_request(0);
    let _ = governor.start_request(READING, 0);
    let verdict = governor.complete_step(step(READING, 0, GIBIBYTE));
    assert_eq!(verdict.reading.observed_step_count, 1);
    assert_eq!(verdict.reading.overflow_count, 1);
    assert_eq!(verdict.reading.redo_attempt_count, 1);
}

#[test]
fn should_lease_as_much_room_again_for_a_first_redo_and_take_it_from_the_capacity() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let capacity_before = governor.start_request(READING, 0);
    let (redo_attempt, capacity_during_redo) = governor
        .begin_redo(READING, None, 0)
        .expect("the first redo is always allowed");
    assert_eq!(redo_attempt.attempt_number(), 1);
    assert_eq!(redo_attempt.headroom_bytes(), GIBIBYTE);
    assert_eq!(
        capacity_before.bytes() - capacity_during_redo.bytes(),
        GIBIBYTE,
        "only the leased room is given back, not every kept weight"
    );
    let _ = governor.abandon_redo(redo_attempt);
}

#[test]
fn should_say_what_the_capacity_would_be_while_reading_without_changing_anything() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let idle_capacity = governor.kept_weight_capacity();
    let capacity_while_reading = governor.capacity_while(READING);
    assert!(
        capacity_while_reading.bytes() < idle_capacity.bytes(),
        "reading a prompt sets working memory and layer slots aside"
    );
    assert_eq!(governor.phase(), LedgerPhase::Idle);
    assert_eq!(governor.kept_weight_capacity(), idle_capacity);
    assert_eq!(
        governor.start_request(READING, 0),
        capacity_while_reading,
        "the answer is the capacity that phase then has"
    );
}

#[test]
fn should_answer_the_capacity_in_force_when_asked_about_the_current_phase() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let capacity = governor.start_request(READING, GIBIBYTE);
    assert_eq!(governor.capacity_while(READING), capacity);
}

#[test]
fn should_answer_no_capacity_for_a_phase_nothing_has_measured() {
    let governor = governor();
    assert_eq!(governor.capacity_while(READING).bytes(), 0);
}

#[test]
fn should_take_a_stored_state_that_grew_shrank_or_emptied_as_it_is() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let capacity_with_two = governor.settle_context(2 * GIBIBYTE);
    assert_eq!(governor.accounts().context_bytes, 2 * GIBIBYTE);
    let capacity_with_one = governor.settle_context(GIBIBYTE);
    assert_eq!(governor.accounts().context_bytes, GIBIBYTE);
    assert_eq!(
        capacity_with_one.bytes(),
        capacity_with_two.bytes() + GIBIBYTE,
        "state that went away is room again"
    );
    let _ = governor.settle_context(0);
    assert_eq!(governor.accounts().context_bytes, 0);
}

#[test]
fn should_show_a_step_that_fits_as_having_no_shortfall() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let accounts = governor.accounts_while(READING, GIBIBYTE);
    assert_eq!(accounts.shortfall_bytes, 0);
    assert_eq!(accounts.stable_shortfall_bytes(), 0);
    assert_eq!(accounts.context_bytes, GIBIBYTE);
    assert_eq!(
        governor.accounts().context_bytes,
        0,
        "asking changes nothing"
    );
}

#[test]
fn should_tell_stored_state_that_cannot_fit_from_working_memory_that_cannot() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, 2 * GIBIBYTE);
    // 3 core + 12 stored + 2 working + 1 slots is over the 16 limit, but
    // what stays after the step (3 + 12 + the margin) is not.
    let peak_only = governor.accounts_while(READING, 12 * GIBIBYTE);
    assert!(peak_only.shortfall_bytes > 0);
    assert_eq!(peak_only.stable_shortfall_bytes(), 0);
    let stable_too = governor.accounts_while(READING, 14 * GIBIBYTE);
    assert!(stable_too.stable_shortfall_bytes() > 0);
}

#[test]
fn should_report_no_working_memory_shortfall_for_a_phase_nothing_has_measured() {
    let governor = governor();
    let accounts = governor.accounts_while(READING, 11 * GIBIBYTE);
    assert!(!accounts.working_memory_is_measured);
    assert_eq!(accounts.shortfall_bytes, 0);
}

#[test]
fn should_report_a_shortfall_for_a_model_with_nothing_to_stream() {
    let everything_fixed = ModelMemoryShape {
        core_weight_bytes: 10 * GIBIBYTE,
        streamable_weight_bytes: 0,
        ..model_shape()
    };
    let mut governor =
        MemoryGovernor::new(LIMIT_BYTES, everything_fixed).expect("the limit is not zero");
    let _ = governor.record_probe(READING, 2 * GIBIBYTE);
    assert_eq!(
        governor.accounts_while(READING, GIBIBYTE).shortfall_bytes,
        0
    );
    let over_the_limit = governor.accounts_while(READING, 5 * GIBIBYTE);
    assert!(over_the_limit.shortfall_bytes > 0);
    assert_eq!(over_the_limit.stable_shortfall_bytes(), 0);
}

#[test]
fn should_lease_the_known_shortfall_when_it_is_more_than_as_much_again() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let _ = governor.start_request(READING, 0);
    let (redo_attempt, _) = governor
        .begin_redo(READING, None, 3 * GIBIBYTE)
        .expect("the first redo is always allowed");
    assert_eq!(redo_attempt.headroom_bytes(), 3 * GIBIBYTE);
    let _ = governor.abandon_redo(redo_attempt);
}

#[test]
fn should_ignore_a_known_shortfall_smaller_than_as_much_again() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let _ = governor.start_request(READING, 0);
    let (redo_attempt, _) = governor
        .begin_redo(READING, None, GIBIBYTE / 4)
        .expect("the first redo is always allowed");
    assert_eq!(redo_attempt.headroom_bytes(), GIBIBYTE);
    let _ = governor.abandon_redo(redo_attempt);
}

#[test]
fn should_lease_a_later_attempt_the_larger_of_double_and_the_known_shortfall() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let _ = governor.start_request(READING, 0);
    let (first_attempt, _) = governor
        .begin_redo(READING, None, 0)
        .expect("the first redo is always allowed");
    let (second_attempt, _) = governor
        .begin_redo(READING, Some(first_attempt), 5 * GIBIBYTE)
        .expect("the second redo is allowed");
    assert_eq!(second_attempt.headroom_bytes(), 5 * GIBIBYTE);
    let _ = governor.abandon_redo(second_attempt);
}

#[test]
fn should_settle_a_redone_step_at_what_it_measured_and_return_the_room() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let capacity_before = governor.start_request(READING, 0);
    let (redo_attempt, _) = governor
        .begin_redo(READING, None, 0)
        .expect("the first redo is always allowed");
    let verdict =
        governor.complete_redone_step(redo_attempt, step(READING, 0, GIBIBYTE + GIBIBYTE / 2));
    assert_eq!(
        capacity_before.bytes() - verdict.capacity.bytes(),
        GIBIBYTE / 2,
        "afterwards the capacity is short by the misjudgment alone"
    );
    assert_eq!(governor.accounts().lease_bytes, 0);
}

#[test]
fn should_double_the_room_on_each_later_attempt_and_stop_after_the_last() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let _ = governor.start_request(READING, 0);
    let mut failed_attempt = None;
    let mut leased_room_bytes = Vec::new();
    while let Some((redo_attempt, _)) = governor.begin_redo(READING, failed_attempt.take(), 0) {
        leased_room_bytes.push(redo_attempt.headroom_bytes());
        assert_eq!(
            governor.accounts().lease_bytes,
            redo_attempt.headroom_bytes(),
            "the failed attempt's room went back before the next was leased"
        );
        failed_attempt = Some(redo_attempt);
    }
    assert_eq!(
        leased_room_bytes,
        vec![GIBIBYTE, 2 * GIBIBYTE, 4 * GIBIBYTE, 8 * GIBIBYTE]
    );
    assert_eq!(leased_room_bytes.len() as u32, MAXIMUM_REDO_ATTEMPT_COUNT);
    assert_eq!(governor.accounts().lease_bytes, 0, "nothing stays leased");
    assert_eq!(governor.overflow_count(), 1);
    assert_eq!(governor.redo_attempt_count(), 4);
}

#[test]
fn should_lease_one_byte_of_room_when_nothing_was_set_aside() {
    let mut governor = governor();
    let _ = governor.record_probe(LedgerPhase::WritingReply, 0);
    let _ = governor.start_request(LedgerPhase::WritingReply, 0);
    let (redo_attempt, _) = governor
        .begin_redo(LedgerPhase::WritingReply, None, 0)
        .expect("the first redo is always allowed");
    assert_eq!(redo_attempt.headroom_bytes(), 1);
    let _ = governor.abandon_redo(redo_attempt);
}

#[test]
fn should_make_room_for_a_lease_and_take_it_back() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let capacity_before = governor.start_request(READING, 0);
    let (lease_id, capacity_with_lease) = governor.grant_lease(LeasePurpose::VisionTower, GIBIBYTE);
    assert_eq!(
        capacity_before.bytes() - capacity_with_lease.bytes(),
        GIBIBYTE
    );
    assert_eq!(governor.give_back_lease(lease_id), Ok(capacity_before));
    assert_eq!(
        governor.give_back_lease(lease_id),
        Err(MemoryLedgerError::UnknownLease)
    );
}

#[test]
fn should_follow_the_limit_and_refuse_a_zero_one() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, GIBIBYTE);
    let capacity_before = governor.start_request(READING, 0);
    let capacity_after = governor
        .change_limit(LIMIT_BYTES - 2 * GIBIBYTE)
        .expect("the new limit is not zero");
    assert!(capacity_after.bytes() < capacity_before.bytes());
    assert_eq!(governor.change_limit(0), Err(MemoryLedgerError::ZeroLimit));
    assert_eq!(governor.kept_weight_capacity(), capacity_after);
}

#[test]
fn should_gain_capacity_on_entering_the_reply_and_lose_it_as_the_context_grows() {
    let mut governor = governor();
    let _ = governor.record_probe(READING, 2 * GIBIBYTE);
    let _ = governor.record_probe(LedgerPhase::WritingReply, GIBIBYTE / 4);
    let reading_capacity = governor.start_request(READING, GIBIBYTE);
    let reply_capacity = governor.enter_phase(LedgerPhase::WritingReply);
    assert!(reply_capacity.bytes() > reading_capacity.bytes());
    let grown_capacity = governor.grow_context(GIBIBYTE / 16);
    assert_eq!(
        reply_capacity.bytes() - grown_capacity.bytes(),
        GIBIBYTE / 16
    );
    let idle_capacity = governor.finish_request(0);
    assert!(idle_capacity.bytes() > reply_capacity.bytes());
}

#[test]
fn should_carry_the_probe_and_the_step_rise_in_the_reading() {
    let mut governor = governor();
    assert_eq!(governor.latest_reading(), None);
    let _ = governor.record_probe(READING, GIBIBYTE);
    let _ = governor.record_probe(READING, GIBIBYTE / 2);
    let _ = governor.start_request(READING, 0);
    let verdict = governor.complete_step(step(READING, 0, GIBIBYTE / 4));
    assert_eq!(verdict.reading.probed_working_memory_bytes, Some(GIBIBYTE));
    assert_eq!(
        verdict.reading.step_peak_beyond_held_bytes,
        GIBIBYTE / 2 + GIBIBYTE / 4
    );
    assert_eq!(
        verdict.reading.largest_reading_step_peak_beyond_held_bytes,
        GIBIBYTE / 2 + GIBIBYTE / 4
    );
    assert_eq!(
        verdict.reading.largest_writing_step_peak_beyond_held_bytes,
        0
    );
}

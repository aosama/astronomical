use super::{WorkingMemoryMeasurements, WorkingMemoryReserve};
use crate::ledger::phase::LedgerPhase;

const fn reading(span_token_count: u32) -> LedgerPhase {
    LedgerPhase::ReadingPrompt { span_token_count }
}

#[test]
fn should_need_no_working_memory_while_idle() {
    let measurements = WorkingMemoryMeasurements::default();
    assert_eq!(
        measurements.reserve_for(LedgerPhase::Idle),
        WorkingMemoryReserve::Measured(0)
    );
}

#[test]
fn should_be_unmeasured_before_any_layer_step_was_measured() {
    let measurements = WorkingMemoryMeasurements::default();
    assert_eq!(
        measurements.reserve_for(reading(4_096)),
        WorkingMemoryReserve::Unmeasured
    );
    assert_eq!(
        measurements.reserve_for(LedgerPhase::WritingReply),
        WorkingMemoryReserve::Unmeasured
    );
    assert!(!measurements.covers(reading(4_096)));
}

#[test]
fn should_reserve_exactly_what_the_same_span_measured() {
    let mut measurements = WorkingMemoryMeasurements::default();
    assert!(measurements.record(reading(4_096), 3_500));
    assert_eq!(
        measurements.reserve_for(reading(4_096)),
        WorkingMemoryReserve::Measured(3_500)
    );
}

#[test]
fn should_cover_a_smaller_span_with_the_next_larger_measurement() {
    let mut measurements = WorkingMemoryMeasurements::default();
    measurements.record(reading(4_096), 3_500);
    measurements.record(reading(8_192), 7_000);
    assert_eq!(
        measurements.reserve_for(reading(2_048)),
        WorkingMemoryReserve::Measured(3_500)
    );
}

#[test]
fn should_not_extrapolate_to_a_span_larger_than_any_measured() {
    let mut measurements = WorkingMemoryMeasurements::default();
    measurements.record(reading(1), 11);
    measurements.record(reading(1_024), 700);
    assert_eq!(
        measurements.reserve_for(reading(2_048)),
        WorkingMemoryReserve::Unmeasured
    );
}

#[test]
fn should_never_reserve_less_than_a_smaller_span_measured() {
    let mut measurements = WorkingMemoryMeasurements::default();
    measurements.record(reading(1_024), 900);
    measurements.record(reading(4_096), 800);
    assert_eq!(
        measurements.reserve_for(reading(4_096)),
        WorkingMemoryReserve::Measured(900)
    );
}

#[test]
fn should_only_raise_what_is_known_and_say_when_it_did() {
    let mut measurements = WorkingMemoryMeasurements::default();
    assert!(measurements.record(LedgerPhase::WritingReply, 40));
    assert!(!measurements.record(LedgerPhase::WritingReply, 30));
    assert!(!measurements.record(LedgerPhase::WritingReply, 40));
    assert!(measurements.record(LedgerPhase::WritingReply, 41));
    assert_eq!(
        measurements.reserve_for(LedgerPhase::WritingReply),
        WorkingMemoryReserve::Measured(41)
    );
}

#[test]
fn should_count_a_zero_byte_measurement_as_measured() {
    let mut measurements = WorkingMemoryMeasurements::default();
    assert!(!measurements.record(LedgerPhase::WritingReply, 0));
    assert_eq!(
        measurements.reserve_for(LedgerPhase::WritingReply),
        WorkingMemoryReserve::Measured(0)
    );
}

#[test]
fn should_ignore_a_measurement_reported_while_idle() {
    let mut measurements = WorkingMemoryMeasurements::default();
    assert!(!measurements.record(LedgerPhase::Idle, 9_999));
    assert_eq!(measurements, WorkingMemoryMeasurements::default());
}
